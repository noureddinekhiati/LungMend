"""Bone mask from CT: HU threshold inside the body, small specks and the scanner table removed."""
import numpy as np
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label

BONE_HU = 250
MIN_BONE_VOXELS = 10
MIN_BODY_PX = 1000


def body_mask_2d(ct2d):
    """Torso and arms on one axial slice; drops everything behind the torso (table)."""
    lab, n = label(ct2d > -500)
    if n == 0:
        return np.zeros(ct2d.shape, bool)
    sizes = np.bincount(lab.ravel())
    sums = np.bincount(lab.ravel(), weights=ct2d.astype(np.float64).ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_BODY_PX and sums[i] / sizes[i] < 200]
    body = binary_fill_holes(np.isin(lab, keep))
    torso = lab == (np.argmax(sizes[1:]) + 1)
    back = np.nonzero(torso.any(axis=1))[0].max()  # axis 0 of the slice runs anterior -> posterior
    body[back + 3:] = False
    return body


def bone_mask(ct):
    """ct: (z, y, x) HU array, y running anterior -> posterior. Returns a uint8 bone mask."""
    bone = ct > BONE_HU
    lab, n = label(bone, structure=generate_binary_structure(3, 1))
    if n:
        sizes = np.bincount(lab.ravel())
        bone &= sizes[lab] >= MIN_BONE_VOXELS
    for z in range(ct.shape[0]):
        if bone[z].any():
            bone[z] &= body_mask_2d(ct[z])
    return bone.astype(np.uint8)
