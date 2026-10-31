"""Dataset for the VAE: augmented healthy lung mask -> signed distance map, computed on the fly.
The rng is seeded per (seed, worker, epoch, index) inside __getitem__.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from transforms import augment_single
from sdt import compute_sdt


class LungSDTDataset(Dataset):
    def __init__(self, npy_dir, pids, sdt_clip_mm=20.0, augment=True, base_seed=0):
        self.npy_dir = Path(npy_dir)
        self.pids = list(pids)
        self.sdt_clip_mm = sdt_clip_mm
        self.augment = augment
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch):
        # Caller must recreate the DataLoader (persistent_workers=False, the
        # default) after calling this, so workers are re-forked with the
        # updated epoch — otherwise already-running worker processes never
        # see the new value.
        self.epoch = epoch

    def __len__(self):
        return len(self.pids)

    def _seed_for(self, index):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        # large odd multipliers to avoid accidental seed collisions across axes
        return (self.base_seed + worker_id * 100_003 + self.epoch * 1_000_003 + index) % (2**32 - 1)

    def __getitem__(self, index):
        pid = self.pids[index]
        lung = np.load(self.npy_dir / f"{pid}_lung.npy")
        meta = json.loads((self.npy_dir / f"{pid}_meta.json").read_text())
        spacing_mm = np.array(meta["resampled_spacing_mm"], dtype=np.float64)

        if self.augment:
            rng = np.random.default_rng(self._seed_for(index))
            lung, _ = augment_single(lung, rng)

        sdt = compute_sdt(lung, spacing_mm, clip_mm=self.sdt_clip_mm)
        sdt_t = torch.from_numpy(sdt).unsqueeze(0)  # (1, D, H, W)
        return {"sdt": sdt_t, "pid": pid}
