"""Cosine noise schedule, v-prediction and DDIM sampling with classifier-free guidance."""
import math

import torch


def cosine_alphas_cumprod(T, s=0.008):
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / f[0])[1:].clamp(min=1e-5, max=1.0)


class GaussianDiffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        self.alphas_cumprod = cosine_alphas_cumprod(T).to(device)
        self.sqrt_ac = self.alphas_cumprod.sqrt()
        self.sqrt_1m_ac = (1 - self.alphas_cumprod).sqrt()

    def _at(self, buf, t):
        return buf[t].view(-1, 1, 1, 1, 1)

    def q_sample(self, x0, t, noise):
        return self._at(self.sqrt_ac, t) * x0 + self._at(self.sqrt_1m_ac, t) * noise

    def v_target(self, x0, noise, t):
        return self._at(self.sqrt_ac, t) * noise - self._at(self.sqrt_1m_ac, t) * x0

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, cond, device, steps=50, guidance=1.5, generator=None):
        """Deterministic DDIM (eta = 0). model_fn(z_t, t, cond, drop_lung) returns the v prediction."""
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        z = torch.randn(shape, device=device, generator=generator)
        for i, t in enumerate(ts):
            tb = t.repeat(shape[0])
            v = model_fn(z, tb, cond, False)
            if guidance != 1.0:
                v_unc = model_fn(z, tb, cond, True)
                v = v_unc + guidance * (v - v_unc)
            a, b = self._at(self.sqrt_ac, tb), self._at(self.sqrt_1m_ac, tb)
            x0 = a * z - b * v
            if i == len(ts) - 1:
                return x0
            eps = b * z + a * v
            a_next = self._at(self.alphas_cumprod, ts[i + 1].repeat(shape[0]))
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return z
