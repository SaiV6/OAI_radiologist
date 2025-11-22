import numpy as np
import torch
import torch.nn.functional as F
from monai.metrics import DiceMetric

from .model_utils import to_long_device


def dice_channels(pred, gt, channel: int):
    return DiceMetric(include_background=False, reduction="mean")(
        pred[:, channel : channel + 1], gt[:, channel : channel + 1]
    ).item()


def evaluate_test(model, inferer, test_loader, device):
    model.eval()

    d_all, df_all, dt_all = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = to_long_device(batch["label"], device)

            logits = inferer(x, model)
            pred = torch.argmax(logits, dim=1, keepdim=True)

            y_oh = F.one_hot(y.squeeze(1), num_classes=3).permute(0, 4, 1, 2, 3).float()
            p_oh = F.one_hot(pred.squeeze(1), num_classes=3).permute(
                0, 4, 1, 2, 3
            ).float()

            dice_ft = DiceMetric(include_background=False, reduction="mean")(
                p_oh[:, 1:3], y_oh[:, 1:3]
            ).item()
            d_all.append(dice_ft)
            df_all.append(dice_channels(p_oh, y_oh, 1))
            dt_all.append(dice_channels(p_oh, y_oh, 2))

    print(f"Test Dice (femur+tibia): {np.mean(d_all):.4f}")
    print(f"  Femur Dice: {np.mean(df_all):.4f}")
    print(f"  Tibia Dice: {np.mean(dt_all):.4f}")
