#!/usr/bin/env python
import os
import subprocess

import torch

from oai_knee_seg import (
    Config,
    set_seed,
    get_device,
    download_oaizib_cm,
    prepare_data_folders,
    get_transforms,
    build_datasets_and_loaders,
    build_model_and_loss,
    train_model,
    evaluate_test,
    save_case_qc,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="3D MONAI UNet training for OAIZIB-CM knee cartilage"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/OAIZIB-CM",
        help="Root directory for OAIZIB-CM dataset",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download OAIZIB-CM dataset via huggingface_hub",
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (1 typical for 3D)"
    )
    parser.add_argument(
        "--val-split", type=int, default=30, help="Last N training scans used for val"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default="./weights/monai_knee_ft_best.pth",
        help="Path to save/load best model weights",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only evaluate / run QC (requires weights_path)",
    )
    parser.add_argument(
        "--run-qc",
        action="store_true",
        help="Run QC overlays on first test or train case",
    )

    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        weights_path=args.weights_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        lr=args.lr,
    )

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.out_pred_dir, exist_ok=True)
    os.makedirs(cfg.qc_dir, exist_ok=True)

    set_seed(42)
    device = get_device()

    if args.download_data:
        download_oaizib_cm(cfg)

    IMTR, LBTR, IMTS, LBTS = prepare_data_folders(cfg)
    img_t, lab_t = get_transforms(cfg.target_spacing)

    (
        train_loader,
        val_loader,
        test_loader,
        train_files,
        val_files,
        test_files,
    ) = build_datasets_and_loaders(IMTR, LBTR, IMTS, LBTS, cfg, img_t, lab_t)

    model, loss_fn, opt, scaler, dice_metric, inferer = build_model_and_loss(
        cfg, device
    )

    if not args.skip_train:
        train_model(
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
        )
    else:
        print("Skipping training (skip-train=True).")

    if os.path.isfile(cfg.weights_path):
        print("Loading best weights from:", cfg.weights_path)
        model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
    else:
        print("WARNING: weights file not found, evaluating with current model parameters.")

    evaluate_test(model, inferer, test_loader, device)

    print("[RUN] verifying test/train file lists...")
    print("  test_files:", len(test_files))
    print("  train_files:", len(train_files))

    if args.run_qc:
        if len(test_files) > 0:
            case = test_files[0]
            img_path = case["image"]
            lab_path = case["label"]
            sid = os.path.basename(img_path).replace("_0000.nii.gz", "")
            print(f"[RUN] using TEST case: {sid}")
            save_case_qc(
                img_path,
                lab_path,
                sid,
                img_t,
                lab_t,
                device,
                inferer,
                model,
                cfg.target_spacing,
                cfg.out_pred_dir,
                cfg.qc_dir,
                verbose=True,
            )
        elif len(train_files) > 0:
            case = train_files[0]
            img_path = case["image"]
            lab_path = case["label"]
            sid = os.path.basename(img_path).replace("_0000.nii.gz", "")
            print(f"[RUN] using TRAIN case: {sid}")
            save_case_qc(
                img_path,
                lab_path,
                sid,
                img_t,
                lab_t,
                device,
                inferer,
                model,
                cfg.target_spacing,
                cfg.out_pred_dir,
                cfg.qc_dir,
                verbose=True,
            )
        else:
            print("[RUN] No files found. Printing quick directory listing:")
            for d in [
                cfg.data_root,
                os.path.join(cfg.data_root, "imagesTr"),
                os.path.join(cfg.data_root, "labelsTr"),
                os.path.join(cfg.data_root, "imagesTs"),
                os.path.join(cfg.data_root, "labelsTs"),
            ]:
                try:
                    print("DIR:", d)
                    subprocess.run(["ls", "-lh", d], check=False)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
