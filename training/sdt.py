"""Signed distance map of a lung mask, positive inside, clipped to +-20 mm and scaled to [-1, 1]; mask = sdt > 0.

Arrays are (z, y, x): axis 0 cranio-caudal, axis 1 antero-posterior, axis 2 left-right.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

DEFAULT_CLIP_MM = 20.0
DEFAULT_TAU_MM = 2.0


def compute_sdt(mask, spacing_mm, clip_mm=DEFAULT_CLIP_MM):
    """mask: bool/uint8 array (any ndim). spacing_mm: sequence of length mask.ndim,
    in mm per voxel, axis-aligned with `mask`. Returns float32 sdt in [-1, 1],
    positive inside the mask, computed on the fly (call this after augmentation,
    never cache it from a pre-augmentation mask)."""
    mask = np.asarray(mask, dtype=bool)
    spacing_mm = tuple(float(s) for s in spacing_mm)

    if mask.all():
        sdt_mm = np.full(mask.shape, clip_mm, dtype=np.float32)
    elif not mask.any():
        sdt_mm = np.full(mask.shape, -clip_mm, dtype=np.float32)
    else:
        dist_in = distance_transform_edt(mask, sampling=spacing_mm)
        dist_out = distance_transform_edt(~mask, sampling=spacing_mm)
        sdt_mm = (dist_in - dist_out).astype(np.float32)

    sdt = np.clip(sdt_mm, -clip_mm, clip_mm) / clip_mm
    return sdt.astype(np.float32)


def sdt_to_mask(sdt):
    return sdt > 0


def soft_mask_np(sdt, clip_mm=DEFAULT_CLIP_MM, tau_mm=DEFAULT_TAU_MM):
    x = sdt * (clip_mm / tau_mm)
    return 1.0 / (1.0 + np.exp(-x))


def soft_mask_torch(sdt, clip_mm=DEFAULT_CLIP_MM, tau_mm=DEFAULT_TAU_MM):
    import torch
    return torch.sigmoid(sdt * (clip_mm / tau_mm))
