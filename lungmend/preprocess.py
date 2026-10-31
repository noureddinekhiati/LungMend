"""Mask preprocessing: component cleaning, lung/bone crop, resampling to 128^3, signed distance map and conditioning maps.

Arrays are (z, y, x): axis 0 cranio-caudal (apex first), axis 1 antero-posterior, axis 2 left-right.
"""
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label, zoom

GRID = 128
LATENT_RES = 16
SDT_CLIP_MM = 20.0
BONE_DIST_CLIP_MM = 50.0
MARGIN_MM = 15.0


def clean_lung(lung):
    """Keep the largest component and the second one if it is at least 15 % of the first; fill holes slice by slice."""
    lab, n = label(lung)
    if n == 0:
        return lung.astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    keep = [order[0]]
    if n > 1 and sizes[order[1]] >= 0.15 * sizes[order[0]]:
        keep.append(order[1])
    out = np.isin(lab, keep)
    return np.stack([binary_fill_holes(s) for s in out]).astype(np.uint8)


def bbox(mask):
    lo, hi = [], []
    for ax in range(mask.ndim):
        idx = np.nonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != ax)))[0]
        if len(idx) == 0:
            raise ValueError("empty mask")
        lo.append(idx[0])
        hi.append(idx[-1] + 1)
    return np.array(lo), np.array(hi)


def crop_box(lung, bone, spacing, margin_mm=MARGIN_MM):
    """Lung box plus margin, united with the bone box (the R231 mask may miss whole regions)."""
    lo_l, hi_l = bbox(lung)
    m = margin_mm / np.asarray(spacing)
    lo_b, hi_b = bbox(bone)
    lo = np.maximum(np.minimum(np.floor(lo_l - m).astype(int), lo_b), 0)
    hi = np.minimum(np.maximum(np.ceil(hi_l + m).astype(int), hi_b), lung.shape)
    return lo, hi


def resize(vol, shape, order):
    out = zoom(vol.astype(np.float32), [s / v for s, v in zip(shape, vol.shape)], order=order, mode="nearest")
    out = out[: shape[0], : shape[1], : shape[2]]
    if out.shape != tuple(shape):
        out = np.pad(out, [(0, s - o) for s, o in zip(shape, out.shape)], mode="edge")
    return out


def to_grid(mask):
    return (resize(mask, (GRID,) * 3, order=0) > 0.5).astype(np.uint8)


def signed_distance(mask, spacing, clip_mm=SDT_CLIP_MM):
    """Positive inside, clipped to +-clip_mm and scaled to [-1, 1]."""
    mask = mask.astype(bool)
    if mask.all() or not mask.any():
        return np.full(mask.shape, 1.0 if mask.all() else -1.0, np.float32)
    d = distance_transform_edt(mask, sampling=spacing) - distance_transform_edt(~mask, sampling=spacing)
    return (np.clip(d, -clip_mm, clip_mm) / clip_mm).astype(np.float32)


def avg_pool(vol, res=LATENT_RES):
    f = vol.shape[0] // res
    return vol.astype(np.float32).reshape(res, f, res, f, res, f).mean(axis=(1, 3, 5))


def prepare(lung, bone, spacing):
    """Returns the model inputs on the 128^3 grid and the crop needed to paste the result back."""
    lung = clean_lung(lung > 0)
    bone = bone > 0
    lo, hi = crop_box(lung, bone, spacing)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    lung_g, bone_g = to_grid(lung[sl]), to_grid(bone[sl])
    grid_spacing = (hi - lo) * np.asarray(spacing, float) / GRID
    dist = np.clip(distance_transform_edt(bone_g == 0, sampling=grid_spacing) / BONE_DIST_CLIP_MM, 0, 1)
    return dict(sdt=signed_distance(lung_g, grid_spacing), bone=avg_pool(bone_g), dist=avg_pool(dist), lo=lo, hi=hi,
                lung_clean=lung)


def paste_back(sdt_grid, lo, hi, shape):
    """Upsample the (mean) signed distance map trilinearly to the crop size, threshold at 0, paste into the full volume."""
    out = np.zeros(shape, np.uint8)
    out[tuple(slice(a, b) for a, b in zip(lo, hi))] = resize(sdt_grid, hi - lo, order=1) > 0
    return out
