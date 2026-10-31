"""Cosine noise schedule, v-prediction and DDIM sampling with classifier-free guidance.
"""
import math

import torch


def cosine_alphas_cumprod(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    return alphas_cumprod[1:].clamp(min=1e-5, max=1.0)  # length T


class GaussianDiffusion:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        alphas_cumprod = cosine_alphas_cumprod(T).to(device)
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

    def _gather(self, buf, t):
        return buf[t].view(-1, 1, 1, 1, 1)

    def q_sample(self, x0, t, noise):
        a = self._gather(self.sqrt_alphas_cumprod, t)
        b = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return a * x0 + b * noise

    def v_from_x0_noise(self, x0, noise, t):
        a = self._gather(self.sqrt_alphas_cumprod, t)
        b = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return a * noise - b * x0

    def x0_from_v(self, x_t, v, t):
        a = self._gather(self.sqrt_alphas_cumprod, t)
        b = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return a * x_t - b * v

    def noise_from_v(self, x_t, v, t):
        a = self._gather(self.sqrt_alphas_cumprod, t)
        b = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        return b * x_t + a * v

    @torch.no_grad()
    def ddim_sample(self, model_fn, shape, cond, device, steps=50, cfg_scale=1.5, generator=None):
        """Deterministic DDIM (eta=0). model_fn(z_t, t, cond, drop_cond) -> v_pred.
        `generator`: optional torch.Generator (same device as `device`) for the initial
        noise draw — opt-in, default None reproduces the original ungenerated behavior
        used by train_ldm.py's validate()/save_debug_sample_png (do not change those
        call sites)."""
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        z = torch.randn(shape, device=device, generator=generator)

        for i in range(len(ts)):
            t = ts[i]
            t_batch = t.repeat(shape[0])
            v_cond = model_fn(z, t_batch, cond, False)
            if cfg_scale != 1.0:
                v_uncond = model_fn(z, t_batch, cond, True)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            x0 = self.x0_from_v(z, v, t_batch)

            if i == len(ts) - 1:
                z = x0
                break

            noise_pred = self.noise_from_v(z, v, t_batch)
            t_next = ts[i + 1].repeat(shape[0])
            a_next = self._gather(self.alphas_cumprod, t_next)
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * noise_pred
        return z

    @torch.no_grad()
    def ddim_sample_repaint(self, model_fn, shape, cond, ref_z0, region_mask, device,
                             steps=50, cfg_scale=1.5, generator=None):
        """RePaint-style sampling (Lugmayr et al.), ADDITIVE — not used by any existing
        call site. `ref_z0`: the known/reference clean latent (e.g. mu_R231/latent_std),
        same shape as `shape`. `region_mask`: bool tensor broadcastable to `shape`
        (typically shape (1,1,Dl,Dl,Dl) at the latent grid, e.g. 16^3) — True = "let the
        network generate freely here", False = "force to the known reference's
        forward-diffused trajectory at every step". Final z at t=0 is `ref_z0` outside
        the region, the network's x0 inside it (both already true at the last DDIM step
        since ref forward-diffusion at t=0 is exactly ref_z0)."""
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        z = torch.randn(shape, device=device, generator=generator)
        region_mask = region_mask.to(device=device, dtype=torch.bool)

        for i in range(len(ts)):
            t = ts[i]
            t_batch = t.repeat(shape[0])

            # known-region forward diffusion of the reference latent at this timestep
            ref_noise = torch.randn(shape, device=device, generator=generator)
            z_known = self.q_sample(ref_z0, t_batch, ref_noise)
            z = torch.where(region_mask, z, z_known)

            v_cond = model_fn(z, t_batch, cond, False)
            if cfg_scale != 1.0:
                v_uncond = model_fn(z, t_batch, cond, True)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
            x0 = self.x0_from_v(z, v, t_batch)

            if i == len(ts) - 1:
                z = torch.where(region_mask, x0, ref_z0)
                break

            noise_pred = self.noise_from_v(z, v, t_batch)
            t_next = ts[i + 1].repeat(shape[0])
            a_next = self._gather(self.alphas_cumprod, t_next)
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * noise_pred
        return z

    def ddim_sample_dps(self, model_fn, shape, cond, ref_sdt, vae, latent_std, device,
                         steps=50, cfg_scale=1.5, lam=0.0, voxel_weight=None, generator=None):
        """DDIM with Diffusion Posterior Sampling-style guidance, ADDITIVE. At each step,
        decodes x0_pred through the frozen VAE and nudges z_t (via autograd through the
        frozen decoder) to reduce a weak symmetric soft-Dice disagreement with `ref_sdt`
        (the R231 SDT, same 128^3 shape as a decoded volume) over ALL voxels, scaled by
        `lam`. `voxel_weight` (optional, same shape as decoded volume): per-voxel
        multiplier on the guidance loss before backprop — used by the n_samples+dps combo
        to weight guidance by sample-ensemble uncertainty; None = uniform weight 1."""
        from sdt import soft_mask_torch

        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        z = torch.randn(shape, device=device, generator=generator)
        ref_soft = soft_mask_torch(ref_sdt).detach()

        for i in range(len(ts)):
            t = ts[i]
            t_batch = t.repeat(shape[0])

            with torch.no_grad():
                v_cond = model_fn(z, t_batch, cond, False)
                if cfg_scale != 1.0:
                    v_uncond = model_fn(z, t_batch, cond, True)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond
                x0 = self.x0_from_v(z, v, t_batch)

            if lam > 0:
                with torch.enable_grad():
                    x0_g = x0.detach().clone().requires_grad_(True)
                    rec_sdt = vae.decode(x0_g * latent_std)
                    pred_soft = soft_mask_torch(rec_sdt)
                    w = 1.0 if voxel_weight is None else voxel_weight
                    inter = (pred_soft * ref_soft * w).sum()
                    denom = (pred_soft * w).sum() + (ref_soft * w).sum()
                    soft_dice_loss = 1.0 - (2 * inter + 1e-6) / (denom + 1e-6)
                    grad = torch.autograd.grad(soft_dice_loss, x0_g)[0]
                x0 = x0 - lam * grad

            if i == len(ts) - 1:
                z = x0
                break

            # recompute noise consistent with the (possibly guided) x0 and the actual z_t
            # used this step — NOT noise_from_v(z, v, t), which assumes the *original*
            # unguided x0 and is stale once x0 has been nudged above.
            a_t = self._gather(self.sqrt_alphas_cumprod, t_batch)
            b_t = self._gather(self.sqrt_one_minus_alphas_cumprod, t_batch)
            noise_pred = (z - a_t * x0) / b_t
            t_next = ts[i + 1].repeat(shape[0])
            a_next = self._gather(self.alphas_cumprod, t_next)
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * noise_pred
        return z
