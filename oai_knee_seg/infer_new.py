import os
import SimpleITK as sitk
import torch


def dicom_to_nifti(input_path, out_nii):
    if os.path.isdir(input_path):
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(input_path)
        if len(files) == 0:
            raise RuntimeError("No DICOM slices found in folder.")
        reader.SetFileNames(files)
        img = reader.Execute()
    elif str(input_path).lower().endswith((".nii", ".nii.gz")):
        return input_path
    else:
        img = sitk.ReadImage(input_path)

    sitk.WriteImage(img, out_nii)
    return out_nii


def infer_new(
    model,
    input_path,
    infer_xforms,
    inferer,
    device,
    out_pred_path="./new_pred.nii.gz",
    tmp_case_path="./new_case.nii.gz",
):
    nii = dicom_to_nifti(input_path, tmp_case_path)
    ref = sitk.ReadImage(nii)
    x = infer_xforms(nii).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = inferer(x, model)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

    pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
    pred_img.CopyInformation(ref)
    sitk.WriteImage(pred_img, out_pred_path)
    print("Saved:", out_pred_path)
    return out_pred_path
