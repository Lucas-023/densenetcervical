from typing import Optional, Callable, List, Tuple
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

try:
    import torchmetrics
    from torchmetrics.classification import MulticlassRecall, MulticlassAccuracy
except Exception:
    torchmetrics = None


# ---------------------
# Building blocks
# ---------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (channel attention)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------------
# Utilities to modify DenseNet
# ---------------------

def replace_stem(model: models.DenseNet):
    """Replace DenseNet's stem conv7x7 + pool with a 3x3 conv stride1 + 2x2 pool."""
    # Original DenseNet has conv0, norm0, relu0, pool0
    conv0 = model.features.conv0
    in_ch = conv0.in_channels
    out_ch = conv0.out_channels
    # new conv same out channels but smaller kernel/stride
    new_conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
    # initialize weights from old conv (center crop if needed)
    with torch.no_grad():
        # if old kernel 7x7 exists, try to center-crop weights
        if conv0.weight.shape[2] >= 3:
            old_w = conv0.weight
            k_old = old_w.shape[2]
            start = (k_old - 3) // 2
            new_conv0.weight.copy_(old_w[:, :, start:start+3, start:start+3])
    model.features.conv0 = new_conv0
    # replace pool0 (was 3x3 stride 2) with 2x2 stride2 maxpool
    model.features.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
    return model


def insert_se_after_blocks(model: models.DenseNet, reduction: int = 16):
    """Insert an SEBlock after each denseblock (in-place insertion).
    We add the SE as an attribute named 'se{idx}' and call it in forward by wrapping features.
    For simplicity, we'll build a wrapper module that runs existing features sequentially with SE inserted.
    """
    # We'll build a new nn.Sequential that reproduces behaviour: conv0, norm0, relu0, pool0,
    # (denseblock1, se1, transition1), (denseblock2,se2,transition2), ... final norm
    seq = []
    f = model.features
    seq.append(f.conv0)
    seq.append(f.norm0)
    seq.append(f.relu0)
    seq.append(f.pool0)

    num_features_list = [
        f.transition1.conv.in_channels, # Após denseblock1
        f.transition2.conv.in_channels, # Após denseblock2
        f.transition3.conv.in_channels, # Após denseblock3
        f.norm5.num_features            # Após denseblock4
    ]
    # denseblocks and transitions are named denseblock1..4 and transition1..3
    se_blocks = {}
    for i in range(1, 5):
        db = getattr(f, f'denseblock{i}')
        seq.append(db)
        # add SE block
        num_features = num_features_list[i-1]
        # db.num_features not always present; fallback to model.num_features per block estimate
        # We'll create SE with channels = model.features.denseblock{i}.dense_layers[-1].conv2.out_channels estimate
        # Simpler: use current model.num_features (updates inside DenseNet) -> create SE with that



        se = SEBlock(num_features, reduction=reduction)
        seq.append(se)
        setattr(model, f'se{i}', se)
        if i < 4:
            tr = getattr(f, f'transition{i}')
            seq.append(tr)

    seq.append(f.norm5)  # final batchnorm
    # replace features with our new sequential
    model.features = nn.Sequential(*seq)
    return model


def apply_atrous_to_denseblocks(model: models.DenseNet, dilations: List[int] = [1,2,3]):
    """Modify each DenseBlock so that its last len(dilations) _DenseLayer(s) use dilated convs
    on their second convolution (conv2). We set dilation rates counting from the last layer backwards
    using the provided dilations list.

    This mutates the internal _DenseLayer.conv2 modules. Works for torchvision implementation.
    """
    # torchvision's DenseBlock contains _DenseLayer modules in an _DenseBlock, accessible via .children() or .denselayer
    features = model.features
    # find denseblocks by attribute names
    for i in range(1,5):
        db = getattr(features, f'denseblock{i}', None)
        if db is None:
            # if we replaced features earlier with a sequential, find the denseblock inside
            for module in features:
                if module.__class__.__name__.startswith('_DenseBlock'):
                    db = module
                    break
        if db is None:
            continue
        # collect dense layers
        layers = [m for m in db.children() if m.__class__.__name__ == '_DenseLayer']
        n = len(layers)
        for idx, d in enumerate(reversed(dilations), start=1):
            layer = layers[-idx]
            # layer has conv2: nn.Conv2d(bottleneck_features, growth_rate, kernel_size=3, padding=1)
            # replace conv2 with dilation d. Need to adjust padding accordingly to keep same output size
            conv2 = layer.conv2
            in_ch = conv2.in_channels
            out_ch = conv2.out_channels
            kernel_size = conv2.kernel_size
            k = kernel_size[0]
            padding = d * (k // 2)
            new_conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=padding, dilation=d, bias=False)
            # copy weights where possible (center) if dilation==1
            with torch.no_grad():
                if d == 1:
                    new_conv2.weight.copy_(conv2.weight)
                else:
                    # for dilated convs we cannot directly reuse weights; initialize from kaiming
                    nn.init.kaiming_normal_(new_conv2.weight, mode='fan_out', nonlinearity='relu')
            layer.conv2 = new_conv2
    return model


# ---------------------
# Full model (A2SDNet121)
# ---------------------
class A2SDNet121(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, reduction: int = 16):
        super().__init__()
        # load torchvision densenet121
        backbone = models.densenet121(pretrained=pretrained)
        # replace stem
        backbone = replace_stem(backbone)
        # apply atrous changes
        backbone = apply_atrous_to_denseblocks(backbone, dilations=[1,2,3])
        # insert SE blocks
        backbone = insert_se_after_blocks(backbone, reduction=reduction)
        self.backbone = backbone
        in_features = backbone.classifier.in_features if hasattr(backbone, 'classifier') else backbone.num_features
        # remove classifier from backbone if present
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        # build head
        self.head = nn.Sequential(
            #nn.AdaptiveAvgPool2d((1,1)),
            #nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# ---------------------
# Losses & metrics
# ---------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce)
        loss = ((1 - p_t) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ---------------------
# DataModule for SIPaKMeD (classification)
# ---------------------
class SIPaKMeDDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4, val_split: float = 0.15):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        # Expect data_dir/train and data_dir/val (if not, will split train into val)
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        test_dir = os.path.join(self.data_dir, 'test')

        if os.path.exists(train_dir) and os.path.exists(val_dir):
            self.train_dataset = ImageFolder(train_dir, transform=self.train_transforms)
            self.val_dataset = ImageFolder(val_dir, transform=self.val_transforms)
        else:
            # fallback: use ImageFolder on data_dir and split
            full = ImageFolder(self.data_dir, transform=self.train_transforms)
            n = len(full)
            n_val = int(math.floor(self.val_split * n))
            n_train = n - n_val
            train_set, val_set = torch.utils.data.random_split(full, [n_train, n_val])
            # ensure val uses val transforms
            train_set.dataset.transform = self.train_transforms
            val_set.dataset.transform = self.val_transforms
            self.train_dataset = train_set
            self.val_dataset = val_set

        if os.path.exists(test_dir):
            self.test_dataset = ImageFolder(test_dir, transform=self.val_transforms)
        else:
            self.test_dataset = None

        # compute class weights for imbalance (per dataset)
        if isinstance(self.train_dataset, torch.utils.data.Dataset) and hasattr(self.train_dataset, 'samples'):
            labels = [s[1] for s in self.train_dataset.samples]
        else:
            # for Subset case
            labels = [full.samples[i][1] for i in (self.train_dataset.indices if hasattr(self.train_dataset, 'indices') else range(len(self.train_dataset)))]
        from collections import Counter
        counts = Counter(labels)
        num_classes = len(self.train_dataset.classes)
        freq = torch.tensor([counts[i] for i in range(num_classes)], dtype=torch.float32)
        self.class_weights = (freq.sum() / (freq + 1e-8))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)


# ---------------------
# LightningModule
# ---------------------
# ---------------------
# LightningModule
# ---------------------
class A2SDLightning(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-3, weight_decay: float = 1e-4, pretrained: bool = True, focal_gamma: float = 2.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = A2SDNet121(num_classes=num_classes, pretrained=pretrained)
        self.register_buffer('class_weights', torch.ones(num_classes))
        self.criterion = FocalLoss(gamma=focal_gamma, weight=None)

        if torchmetrics is not None:
            self.val_recall = MulticlassRecall(num_classes=num_classes, average=None)
            self.train_recall = MulticlassRecall(num_classes=num_classes, average=None)

            # --- 👇 ADIÇÃO: Métricas de Teste ---
            # Vamos calcular 'macro' (média simples) e 'None' (por classe)
            self.test_recall = MulticlassRecall(num_classes=num_classes, average=None)
            self.test_recall_macro = MulticlassRecall(num_classes=num_classes, average='macro')
            self.test_acc = MulticlassAccuracy(num_classes=num_classes, average=None)
            self.test_acc_macro = MulticlassAccuracy(num_classes=num_classes, average='macro')
            # --- Fim da Adição ---

        else:
            self.val_recall = None
            self.train_recall = None

            # --- 👇 ADIÇÃO: Métricas de Teste ---
            self.test_recall = None
            self.test_recall_macro = None
            self.test_acc = None
            self.test_acc_macro = None
            # --- Fim da Adição ---

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self):
        dm = getattr(self.trainer, 'datamodule', None)
        if dm is not None and hasattr(dm, 'class_weights'):
            w = dm.class_weights
            if isinstance(w, torch.Tensor):
                self.class_weights = w.to(self.device)
                self.criterion.weight = self.class_weights

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        if self.train_recall is not None:
            self.train_recall.update(preds, y)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.train_recall is not None:
            recalls = self.train_recall.compute()
            for i, r in enumerate(recalls):
                self.log(f'train/recall_class_{i}', r, prog_bar=False)
            self.train_recall.reset()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        if self.val_recall is not None:
            self.val_recall.update(preds, y)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def on_validation_epoch_end(self):
        if self.val_recall is not None:
            recalls = self.val_recall.compute()
            for i, r in enumerate(recalls):
                self.log(f'val/recall_class_{i}', r, prog_bar=True)
            self.val_recall.reset()

    # --- 👇 MODIFICAÇÃO: test_step agora atualiza as métricas ---
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Log da loss
        self.log('test/loss', loss, on_epoch=True, prog_bar=True)

        # Atualiza métricas
        if self.test_recall is not None:
            self.test_recall.update(preds, y)
            self.test_recall_macro.update(preds, y)
            self.test_acc.update(preds, y)
            self.test_acc_macro.update(preds, y)

        return {'loss': loss}

    # --- 👇 ADIÇÃO: Novo hook para logar as métricas no final do teste ---
    def on_test_epoch_end(self):
        if self.test_recall is not None:
            # Log Acurácia Média
            acc_macro = self.test_acc_macro.compute()
            self.log('test/accuracy_macro', acc_macro, prog_bar=True)

            # Log Recall Médio
            recall_macro = self.test_recall_macro.compute()
            self.log('test/recall_macro', recall_macro, prog_bar=True)

            # Log Acurácia por Classe (opcional, pode poluir o log)
            # recalls_per_class = self.test_recall.compute()
            # for i, r in enumerate(recalls_per_class):
            #     self.log(f'test/recall_class_{i}', r)

            # Resetar métricas
            self.test_recall.reset()
            self.test_recall_macro.reset()
            self.test_acc.reset()
            self.test_acc_macro.reset()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10)
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }



# ---------------------
# Example quick-run helper
# ---------------------

if __name__ == "__main__":

    # --- Configurações ---
    DATA_DIR = 'dataset'
    MAX_EPOCHS = 300
    BATCH_SIZE = 16
    NUM_WORKERS = 0 # 0 é mais seguro no Windows, pode aumentar no Linux

    # Otimização para GPUs modernas (RTX 3000/4000, A-Series)
    if torch.cuda.is_available():
        print(f"GPU encontrada: {torch.cuda.get_device_name(0)}")
        torch.set_float32_matmul_precision('medium')
    else:
        print("AVISO: GPU não encontrada, rodando em CPU.")

    print("--- 1. CONFIGURANDO DATAMODULE E MODELO ---")
    
    # 1. Configurar o DataModule
    # (Usa a sua classe SIPaKMeDDataModule)
    dm = SIPaKMeDDataModule(
        data_dir=DATA_DIR, 
        img_size=224, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    # 2. Configurar o Modelo
    # (Usa a sua classe A2SDLightning)
    # Precisamos rodar setup() aqui para descobrir o num_classes
    dm.setup(stage='fit') 
    
    # Pega o num_classes do dataset (funciona com ImageFolder ou Subset)
    num_classes = 0
    if hasattr(dm.train_dataset, 'classes'):
        num_classes = len(dm.train_dataset.classes)
    elif hasattr(dm.train_dataset, 'dataset'):
        num_classes = len(dm.train_dataset.dataset.classes)
    else:
        raise Exception("Não foi possível determinar o número de classes do dataset.")

    print(f"Dataset encontrado com {num_classes} classes.")
    
    model = A2SDLightning(
        num_classes=num_classes, 
        lr=1e-3, 
        weight_decay=1e-4, 
        pretrained=True
    )
    
    # Passar os pesos de classe (lógica da sua example_train)
    model.class_weights = dm.class_weights
    if dm.class_weights is not None:
         model.criterion.weight = model.class_weights

    # 3. Configurar os Callbacks
    # (Lógica da sua example_train)
    # Salva o melhor modelo baseado na perda de validação
    ckpt = ModelCheckpoint(
        monitor='val/loss', 
        mode='min', 
        save_top_k=1,
        filename='best-model-{epoch:02d}-{val_loss:.2f}' # Nome de arquivo claro
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 4. Configurar o Trainer
    # (Lógica da sua example_train)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[ckpt, lr_monitor],
        precision="16-mixed", # Recomendado para GPUs modernas
        accumulate_grad_batches=4
    )

    print("--- 2. INICIANDO TREINAMENTO ---")
    
    # 5. Treinar o modelo
    trainer.fit(model, datamodule=dm)

    print("--- 3. TREINAMENTO CONCLUÍDO. INICIANDO TESTE ---")
    
    # 6. Testar o modelo
    # O Lightning é inteligente: ckpt_path="best" automaticamente
    # encontra o caminho do melhor modelo salvo pelo callback 'ckpt'.
    trainer.test(datamodule=dm, ckpt_path="best")
    
    print("--- 4. PROCESSO CONCLUÍDO ---")
    print(f"O melhor modelo foi salvo em: {ckpt.best_model_path}")