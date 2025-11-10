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

from artigo import A2SDLightning, A2SDNet121