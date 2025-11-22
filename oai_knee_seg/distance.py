import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree


def femur_tibia_min_distance_mm(seg_path):
    img = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    sx, sy, sz = img.GetSpacing()
    spacing_zyx = np.array([sz, sy, sx], dtype=float)

    f_idx = np.argwhere(seg == 1)
    t_idx = np.argwhere(seg == 2)

    if f_idx.size == 0 or t_idx.size == 0:
        return None

    f_mm = f_idx * spacing_zyx
    t_mm = t_idx * spacing_zyx
    tree = cKDTree(t_mm)
    dists, nn = tree.query(f_mm, k=1)
    j = int(np.argmin(dists))
    return float(dists[j]), f_mm[j], t_mm[nn[j]], spacing_zyx
