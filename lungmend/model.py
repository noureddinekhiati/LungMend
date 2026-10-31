"""LungMend: load the VAE + latent diffusion model and correct a lung mask."""
from pathlib import Path

import numpy as np
import torch

from .diffusion import GaussianDiffusion
from .preprocess import paste_back, prepare
from .unet import UNetLatent3D
from .vae import VAE3D

LATENT_CH = 8


class LungMend:
    def __init__(self, weights_dir="weights", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        w = Path(weights_dir)
        vae_state = torch.load(w / "lungmend_vae.pt", map_location="cpu", weights_only=True)
        ldm_state = torch.load(w / "lungmend_ldm.pt", map_location="cpu", weights_only=True)

        self.vae = VAE3D(in_ch=1, stem_ch=32, widths=(32, 64, 128), latent_ch=LATENT_CH)
        self.vae.load_state_dict(vae_state["model"])
        self.latent_std = float(vae_state["latent_std"])

        # input = noisy latent + encoded R231 mask + bone occupancy + distance to bone
        self.unet = UNetLatent3D(in_ch=2 * LATENT_CH + 2, out_ch=LATENT_CH, model_channels=128, channel_mult=(1, 2, 4),
                                 num_res_blocks=2, num_heads=4)
        self.unet.load_state_dict(ldm_state["model"])
        self.vae.to(self.device).eval()
        self.unet.to(self.device).eval()
        self.diffusion = GaussianDiffusion(T=1000, device=self.device)

    def _model_fn(self, z, t, cond, drop_lung):
        c_lung = torch.zeros_like(cond["lung"]) if drop_lung else cond["lung"]
        return self.unet(torch.cat([z, c_lung, cond["bone"], cond["dist"]], 1), t)

    @torch.no_grad()
    def sample(self, inputs, n_samples=5, guidance=1.5, steps=50, seed=0):
        """Returns the n_samples decoded signed distance maps on the 128^3 grid, shape (n, 128, 128, 128)."""
        t = lambda a: torch.from_numpy(np.asarray(a, np.float32))[None, None].to(self.device)  # noqa: E731
        mu, _ = self.vae.encode(t(inputs["sdt"]))
        cond = dict(lung=mu / self.latent_std, bone=t(inputs["bone"]), dist=t(inputs["dist"]))
        out = []
        for i in range(n_samples):
            g = torch.Generator(device=self.device).manual_seed(seed + i)
            z = self.diffusion.ddim_sample(self._model_fn, cond["lung"].shape, cond, self.device, steps, guidance, g)
            out.append(self.vae.decode(z * self.latent_std)[0, 0].float().cpu().numpy())
        return np.stack(out)

    def predict(self, lung, bone, spacing=(1.0, 1.0, 1.0), n_samples=5, guidance=1.5, steps=50, seed=0,
                return_samples=False):
        """lung, bone: (z, y, x) binary arrays on the same grid. Returns the corrected lung mask (bool, same grid)."""
        inputs = prepare(lung, bone, spacing)
        samples = self.sample(inputs, n_samples, guidance, steps, seed)
        mask = paste_back(samples.mean(0), inputs["lo"], inputs["hi"], lung.shape)
        if return_samples:
            return mask.astype(bool), samples
        return mask.astype(bool)
