"""Dataset for the latent diffusion model: augmented healthy mask, a corrupted copy of it, the bone mask and the
distance-to-bone map. Everything derived (SDT, corruption, distance) is computed after augmentation, with the rng seeded per
(seed, worker, epoch, index). Validation uses one fixed corruption per patient.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset, get_worker_info

from transforms import augment_pair
from corruption_v2 import corrupt
from sdt import compute_sdt

DIST_TO_BONE_CLIP_MM = 50.0


def stable_seed(*parts):
    s = "_".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def avgpool_to_latent(volume, latent_res):
    """Exact box-filter average pool)."""
    d = volume.shape[0]
    assert d % latent_res == 0, f"volume size {d} not divisible by latent_res {latent_res}"
    factor = d // latent_res
    v = volume.astype(np.float32).reshape(latent_res, factor, latent_res, factor, latent_res, factor)
    return v.mean(axis=(1, 3, 5))


class LatentDiffusionDataset(Dataset):
    def __init__(self, npy_dir, pids, latent_res=16, sdt_clip_mm=20.0, base_seed=0,
                 fixed_corruption=False, corruption_config=None):
        self.npy_dir = Path(npy_dir)
        self.pids = list(pids)
        self.latent_res = latent_res
        self.sdt_clip_mm = sdt_clip_mm
        self.base_seed = base_seed
        self.epoch = 0
        self.fixed_corruption = fixed_corruption
        self.corruption_config = corruption_config

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.pids)

    def _seed_for(self, index):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        return (self.base_seed + worker_id * 100_003 + self.epoch * 1_000_003 + index) % (2**32 - 1)

    def __getitem__(self, index):
        pid = self.pids[index]
        lung = np.load(self.npy_dir / f"{pid}_lung.npy")
        thorax = np.load(self.npy_dir / f"{pid}_thorax.npy")
        meta = json.loads((self.npy_dir / f"{pid}_meta.json").read_text())
        spacing_mm = np.array(meta["resampled_spacing_mm"], dtype=np.float64)

        if self.fixed_corruption:
            rng = np.random.default_rng(stable_seed(pid, 0))
        else:
            rng = np.random.default_rng(self._seed_for(index))
            lung, thorax, _ = augment_pair(lung, thorax, rng)

        corrupted, info = corrupt(lung, thorax, rng, spacing_mm, config=self.corruption_config)

        sdt_clean = compute_sdt(lung, spacing_mm, clip_mm=self.sdt_clip_mm)
        sdt_corr = compute_sdt(corrupted, spacing_mm, clip_mm=self.sdt_clip_mm)

        thorax_cond = avgpool_to_latent(thorax, self.latent_res)
        bone_dist_mm = distance_transform_edt(thorax == 0, sampling=spacing_mm)
        dist_cond = np.clip(bone_dist_mm / DIST_TO_BONE_CLIP_MM, 0.0, 1.0)
        dist_cond = avgpool_to_latent(dist_cond, self.latent_res)

        return {
            "sdt_clean": torch.from_numpy(sdt_clean).unsqueeze(0),
            "sdt_corr": torch.from_numpy(sdt_corr).unsqueeze(0),
            "thorax_cond": torch.from_numpy(thorax_cond).unsqueeze(0),
            "dist_cond": torch.from_numpy(dist_cond.astype(np.float32)).unsqueeze(0),
            "spacing_mm": torch.from_numpy(spacing_mm),
            "pid": pid,
            "achieved_s": float(info["achieved_s"]),
        }
