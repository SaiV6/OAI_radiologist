import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    data_root: str = "./data/OAIZIB-CM"
    weights_path: str = "./weights/monai_knee_ft_best.pth"
    log_dir: str = "./logs"
    out_pred_dir: str = "./preds"
    qc_dir: str = "./qc"

    download_data: bool = True

    # training hyperparams
    target_spacing: tuple = (0.8, 0.36, 0.36)  # (z, y, x) mm
    batch_size: int = 1
    epochs: int = 350
    val_split: int = 30
    lr: float = 1e-4

    # sliding window inferer
    roi_size: tuple = (112, 224, 224)
    sw_batch_size: int = 1
    overlap: float = 0.55


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device
