import os
import numpy as np
import SimpleITK as sitk
import imageio.v3 as iio
import torch
from torch.cuda.amp import autocast


def itk_spacing_from_target(ts):
    z, y, x = ts
    return (x, y, z)


def save_case_qc(
    img_path,
    lab_path,
    sid,
    img_t,
    lab_t,
    device,
    inferer,
    model,
    target_spacing,
    out_pred_dir,
    qc_dir,
    verbose=True,
):
    os.makedirs(out_pred_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    try:
        if verbose:
            print(f"[QC] sid={sid}")
            print(f"[QC] img_path={img_path}")
            print(f"[QC] lab_path={lab_path}")
            print("[QC] loading & transforming...")

        x_t = img_t(img_path)
        vol_t = x_t.squeeze(0).cpu().numpy()

        gt_t = lab_t(lab_path).squeeze(0).cpu().numpy().astype(np.uint8)
        gt_t[(gt_t != 1) & (gt_t != 3)] = 0
        gt_t[gt_t == 1] = 1
        gt_t[gt_t == 3] = 2

        if verbose:
            print(
                f"[QC] vol_t shape={vol_t.shape}  "
                f"min/max=({vol_t.min():.3f},{vol_t.max():.3f})"
            )
            print(f"[QC] gt_t shape={gt_t.shape}  unique labels={np.unique(gt_t).tolist()}")

        if verbose:
            print("[QC] running inference...")
        x = x_t.unsqueeze(0).to(device)
        with torch.no_grad(), autocast(enabled=(device.type == "cuda")):
            logits = inferer(x, model)
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

        if verbose:
            print(f"[QC] pred shape={pred.shape}  unique={np.unique(pred).tolist()}")

        pred_img_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
        pred_img_sitk.SetSpacing(itk_spacing_from_target(target_spacing))
        pred_img_sitk.SetOrigin((0.0, 0.0, 0.0))
        pred_img_sitk.SetDirection(
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        )
        out_nii = os.path.join(out_pred_dir, f"{sid}_pred_transformed.nii.gz")
        sitk.WriteImage(pred_img_sitk, out_nii)
        print(f"[QC] wrote: {out_nii}")

        def colorize(mask_zyx):
            z, y, x = mask_zyx.shape
            rgb = np.zeros((z, y, x, 3), dtype=np.uint8)
            rgb[mask_zyx == 1, 0] = 255
            rgb[mask_zyx == 2, 2] = 255
            return rgb

        def blend(gray, overlay_rgb, alpha=0.35):
            g = gray.astype(np.float32)
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)
            base = np.stack([g, g, g], -1) * 255.0
            out = (1 - alpha) * base + alpha * overlay_rgb.astype(np.float32)
            return np.clip(out, 0, 255).astype(np.uint8)

        pred_rgb = colorize(pred)
        gt_rgb = colorize(gt_t)
        Z = vol_t.shape[0]

        # some PNG slices
        for z in [Z // 4, Z // 2, 3 * Z // 4]:
            gt_overlay = blend(vol_t[z], gt_rgb[z])
            pr_overlay = blend(vol_t[z], pred_rgb[z])
            canvas = np.concatenate([gt_overlay, pr_overlay], axis=1)
            p = os.path.join(qc_dir, f"{sid}_z{z:03d}.png")
            iio.imwrite(p, canvas)
            print(f"[QC] wrote PNG: {p}")

        # MP4
        frames = []
        for z in range(Z):
            gt_overlay = blend(vol_t[z], gt_rgb[z])
            pr_overlay = blend(vol_t[z], pred_rgb[z])
            frames.append(np.concatenate([gt_overlay, pr_overlay], axis=1))
        mp4 = os.path.join(qc_dir, f"{sid}_gt_vs_pred.mp4")
        iio.imwrite(mp4, frames, fps=30, codec="libx264", quality=7)
        print(f"[QC] wrote MP4: {mp4}")

    except Exception as e:
        print("[QC] ERROR:", e)
        import traceback

        traceback.print_exc()
