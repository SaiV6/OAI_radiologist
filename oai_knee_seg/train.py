import numpy as np
import torch
from torch.cuda.amp import autocast

from .model_utils import one_hot, to_long_device


def train_model(
    model,
    loss_fn,
    opt,
    scaler,
    dice_metric,
    train_loader,
    val_loader,
    inferer,
    device,
    cfg,
):
    best_val = 0.0

    import os

    os.makedirs(os.path.dirname(cfg.weights_path), exist_ok=True)

    import time

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running = []

        for batch in train_loader:
            x = batch["image"].to(device)
            y = to_long_device(batch["label"], device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running.append(loss.item())

        # validation
        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = to_long_device(batch["label"], device)

                with autocast(enabled=(device.type == "cuda")):
                    logits = inferer(x, model)
                pred = torch.argmax(logits, dim=1, keepdim=True)

                y_oh = one_hot(y, 3)
                p_oh = one_hot(pred, 3)
                dices.append(dice_metric(p_oh[:, 1:3], y_oh[:, 1:3]).item())

        mean_train = float(np.mean(running))
        mean_val = float(np.mean(dices)) if dices else 0.0
        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | train loss {mean_train:.4f} | "
            f"val dice(femur+tibia) {mean_val:.4f} | {dt:.1f}s"
        )

        if mean_val > best_val:
            best_val = mean_val
            torch.save(model.state_dict(), cfg.weights_path)
            print("  ✅ Saved best weights ->", cfg.weights_path)

    print(f"Best validation dice (femur+tibia): {best_val:.4f}")
