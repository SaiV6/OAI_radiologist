# OAIZIB-CM Knee Cartilage Segmentation (MONAI + 3D UNet)

This repo contains a 3D MONAI UNet pipeline for femoral and tibial cartilage segmentation on knee MRI (OAIZIB-CM / OAI-style data). It includes:

- Data download + preprocessing (HuggingFace OAIZIB-CM dataset)
- 3D UNet training with Dice + Cross-Entropy loss
- Test-time sliding-window inference
- QC visualizations (PNG slices + MP4 overlays)
- Inference utilities for new DICOM / NIfTI studies
- Optional femur–tibia minimum distance computation

---

## Install

```bash
git clone <your-repo-url>.git
cd oai-knee-seg

# create & activate your environment if desired, then:
pip install -r requirements.txt

python train_eval_qc.py --download-data --run-qc
'''

After training you can use infer_new from the import

from oai_knee_seg import (
    Config, get_device, set_seed,
    get_transforms, build_model_and_loss,
    infer_new,
)

cfg = Config()
set_seed(42)
device = get_device()

img_t, _ = get_transforms(cfg.target_spacing)
model, _, _, _, _, inferer = build_model_and_loss(cfg, device)
model.load_state_dict(torch.load(cfg.weights_path, map_location=device))
model.eval()

# input_path can be a DICOM folder, single DICOM, or NIfTI
infer_new(
    model,
    input_path="/path/to/dicom_or_nii",
    infer_xforms=img_t,
    inferer=inferer,
    device=device,
    out_pred_path="./new_pred.nii.gz",
)

