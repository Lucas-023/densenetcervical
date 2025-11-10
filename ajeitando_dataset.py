#!/usr/bin/env python3

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

SRC_ROOT = Path("archive")
DST_ROOT = Path("dataset")
RANDOM_SEED = 42

# só pegar BMP pq são as imagens recortadas
ALLOWED_EXTS = {".bmp"}

RATIOS = {"train":0.7, "val":0.15, "test":0.15}
CLEAR_DEST = True

random.seed(RANDOM_SEED)

def gather_cropped_images(class_dir: Path):
    cropped_dir = class_dir.rglob("CROPPED")
    imgs = []
    for cdir in cropped_dir:
        for p in cdir.iterdir():
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                imgs.append(p)
    return imgs

def safe_makedirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    if not SRC_ROOT.exists():
        raise SystemExit(f"Fonte '{SRC_ROOT}' não existe.")

    if CLEAR_DEST and DST_ROOT.exists():
        print(f"Removendo {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    for split in RATIOS:
        (DST_ROOT / split).mkdir(parents=True, exist_ok=True)

    classes = [d for d in SRC_ROOT.iterdir() if d.is_dir()]
    print("Classes detectadas:", [c.name for c in classes])

    for cls in tqdm(classes):
        cls_name = cls.name
        imgs = gather_cropped_images(cls)

        imgs = sorted(imgs)
        random.shuffle(imgs)
        n = len(imgs)

        if n == 0:
            print(f"ATENÇÃO: {cls_name} sem imagens CROPPED!")
            continue

        n_train = int(n*RATIOS["train"])
        n_val = int(n*RATIOS["val"])
        n_test = n - n_train - n_val

        splits = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train+n_val],
            "test": imgs[n_train+n_val:]
        }

        for split_name, files in splits.items():
            out_dir = DST_ROOT/split_name/cls_name
            safe_makedirs(out_dir)
            for f in files:
                shutil.copy(f, out_dir / f.name)

    print("\nDONE.")
    print("dataset final gerado em:", DST_ROOT.resolve())

if __name__ == "__main__":
    main()
