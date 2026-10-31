"""Joint augmentation of lung and bone masks (small rotations, scaling, flips applied to both at once).
The caller passes the rng; this module never seeds its own.
"""
import numpy as np
from scipy.ndimage import affine_transform


def sample_augment_params(rng, rot_deg=10.0, zoom_range=(0.9, 1.1),
                           trans_frac=0.05, flip_p=0.5):
    return dict(
        angles_deg=rng.uniform(-rot_deg, rot_deg, size=3),
        zoom=float(rng.uniform(zoom_range[0], zoom_range[1])),
        translation_frac=rng.uniform(-trans_frac, trans_frac, size=3),
        flip=bool(rng.random() < flip_p),
    )


def _rotation_matrix(angles_deg):
    ax, ay, az = np.deg2rad(angles_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _apply_one(volume, params):
    vol = volume.astype(np.float32)
    if params["flip"]:
        vol = vol[:, :, ::-1]

    shape = np.array(vol.shape, dtype=np.float64)
    center = (shape - 1) / 2.0
    R = _rotation_matrix(params["angles_deg"])
    translation_vox = params["translation_frac"] * shape

    # affine_transform maps output_coord -> input_coord, so use the inverse
    # of the forward transform out = center + translation + zoom*R*(in - center)
    inv_matrix = R.T / params["zoom"]
    offset = center - inv_matrix @ (center + translation_vox)

    out = affine_transform(np.ascontiguousarray(vol), inv_matrix, offset=offset,
                            order=0, mode="constant", cval=0.0)
    return (out > 0.5).astype(np.uint8)


def augment_pair(lung, thorax, rng, rot_deg=10.0, zoom_range=(0.9, 1.1),
                  trans_frac=0.05, flip_p=0.5):
    """lung, thorax: uint8 arrays, identical shape. Returns (lung_aug, thorax_aug, params)."""
    assert lung.shape == thorax.shape, f"shape mismatch {lung.shape} vs {thorax.shape}"
    params = sample_augment_params(rng, rot_deg, zoom_range, trans_frac, flip_p)
    lung_aug = _apply_one(lung, params)
    thorax_aug = _apply_one(thorax, params)
    return lung_aug, thorax_aug, params


def augment_single(volume, rng, rot_deg=10.0, zoom_range=(0.9, 1.1),
                    trans_frac=0.05, flip_p=0.5):
    """Single-volume variant (e.g. VAE training, which only ever sees the lung
    mask — no thorax conditioning). Returns (volume_aug, params)."""
    params = sample_augment_params(rng, rot_deg, zoom_range, trans_frac, flip_p)
    return _apply_one(volume, params), params
