import glob
import os
import shutil
import zipfile

import numpy as np
from huggingface_hub import snapshot_download
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensity,
    EnsureType,
)

from .config import Config


def download_oaizib_cm(cfg: Config):
    os.makedirs(cfg.data_root, exist_ok=True)
    print(f"Downloading OAIZIB-CM into {cfg.data_root} ...")
    snapshot_download(
        repo_id="YongchengYAO/OAIZIB-CM",
        repo_type="dataset",
        local_dir=cfg.data_root,
    )
    print("Download complete.")


def prepare_data_folders(cfg: Config):
    data_root = cfg.data_root
    print(f"Preparing data under {data_root} ...")

    # 1) unzip all zips
    for z in glob.glob(os.path.join(data_root, "**", "*.zip"), recursive=True):
        print("Unzipping:", z)
        with zipfile.ZipFile(z, "r") as f:
            f.extractall(data_root)

    # 2) canonical folders
    IMTR = os.path.join(data_root, "imagesTr")
    LBTR = os.path.join(data_root, "labelsTr")
    IMTS = os.path.join(data_root, "imagesTs")
    LBTS = os.path.join(data_root, "labelsTs")
    for d in (IMTR, LBTR, IMTS, LBTS):
        os.makedirs(d, exist_ok=True)

    # 3) move NIfTIs
    for f in glob.glob(os.path.join(data_root, "**", "*.nii.gz"), recursive=True):
        base = os.path.basename(f)
        path_lower = os.path.dirname(f).lower()

        if "_0000.nii.gz" in base:  # image
            if "imagestr" in path_lower and os.path.dirname(f) != IMTR:
                shutil.move(f, os.path.join(IMTR, base))
            elif "imagests" in path_lower and os.path.dirname(f) != IMTS:
                shutil.move(f, os.path.join(IMTS, base))
        else:  # label
            if "labelstr" in path_lower and os.path.dirname(f) != LBTR:
                shutil.move(f, os.path.join(LBTR, base))
            elif "labelsts" in path_lower and os.path.dirname(f) != LBTS:
                shutil.move(f, os.path.join(LBTS, base))

    # 4) final sanity
    train_imgs = sorted(glob.glob(os.path.join(IMTR, "*.nii.gz")))
    train_labs = sorted(glob.glob(os.path.join(LBTR, "*.nii.gz")))
    test_imgs = sorted(glob.glob(os.path.join(IMTS, "*.nii.gz")))
    test_labs = sorted(glob.glob(os.path.join(LBTS, "*.nii.gz")))

    print(f"Train images: {len(train_imgs)} | Train labels: {len(train_labs)}")
    print(f"Test  images: {len(test_imgs)} | Test  labels: {len(test_labs)}")

    if len(train_imgs) != len(train_labs):
        raise RuntimeError("Mismatch train img/label counts")
    if len(test_imgs) != len(test_labs):
        raise RuntimeError("Mismatch test img/label counts")

    return IMTR, LBTR, IMTS, LBTS


def remap_labels_np(lbl: np.ndarray) -> np.ndarray:
    """
    Map OAIZIB labels {0..5} -> {0:bg, 1:femur, 2:tibia}.
    """
    a = lbl.copy()
    a[(a != 1) & (a != 3)] = 0
    a[a == 1] = 1
    a[a == 3] = 2
    return a.astype(np.uint8)


def get_transforms(target_spacing):
    img_t = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=target_spacing, mode=("bilinear",)),
            ScaleIntensity(minv=0.0, maxv=1.0),
            EnsureType(),
        ]
    )

    lab_t = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=target_spacing, mode=("nearest",)),
            EnsureType(),
        ]
    )
    return img_t, lab_t


class KneeDataset(Dataset):
    def __init__(self, items, img_t, lab_t):
        self.items = items
        self.img_t = img_t
        self.lab_t = lab_t

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        img = self.img_t(d["image"])
        lab = self.lab_t(d["label"]).astype(np.uint8)
        lab = remap_labels_np(lab)
        return {"image": img, "label": lab}


def build_datasets_and_loaders(IMTR, LBTR, IMTS, LBTS, cfg: Config, img_t, lab_t):
    train_imgs = sorted(glob.glob(os.path.join(IMTR, "*.nii.gz")))
    train_labs = [
        os.path.join(LBTR, os.path.basename(p).replace("_0000", ""))
        for p in train_imgs
    ]

    test_imgs = sorted(glob.glob(os.path.join(IMTS, "*.nii.gz")))
    test_labs = [
        os.path.join(LBTS, os.path.basename(p).replace("_0000", ""))
        for p in test_imgs
    ]

    print(f"Train images: {len(train_imgs)} | Train labels: {len(train_labs)}")
    print(f"Test  images: {len(test_imgs)} | Test  labels: {len(test_labs)}")

    if len(train_imgs) < cfg.val_split:
        raise RuntimeError("Not enough training scans for the requested VAL_SPLIT.")

    train_files = [
        {"image": i, "label": l}
        for i, l in zip(train_imgs[:-cfg.val_split], train_labs[:-cfg.val_split])
    ]
    val_files = [
        {"image": i, "label": l}
        for i, l in zip(train_imgs[-cfg.val_split:], train_labs[-cfg.val_split:])
    ]
    test_files = [{"image": i, "label": l} for i, l in zip(test_imgs, test_labs)]

    train_ds = KneeDataset(train_files, img_t, lab_t)
    val_ds = KneeDataset(val_files, img_t, lab_t)
    test_ds = KneeDataset(test_files, img_t, lab_t)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_files, val_files, test_files
