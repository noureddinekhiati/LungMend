"""3D U-Net denoiser in latent space. Conditioning by channel concatenation: noisy latent, encoded corrupted mask, bone
occupancy and distance to bone.
"""
import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _groups(ch, max_groups=32):
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return g


class ResBlock3D(nn.Module):
    """FiLM-style (scale-shift) time conditioning — ref/unet.py's use_scale_shift_norm."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.emb_proj(emb)[:, :, None, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(self.act(h)))
        return h + self.skip(x)


class SelfAttention3D(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.num_heads = num_heads
        self.qkv = nn.Conv3d(ch, ch * 3, 1)
        self.proj = nn.Conv3d(ch, ch, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        n = d * h * w
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, n)
        q, k, v = qkv.unbind(1)
        scale = (c // self.num_heads) ** -0.5
        attn = torch.softmax((q.transpose(-1, -2) @ k) * scale, dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(b, c, d, h, w)
        return x + self.proj(out)


class Downsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class UNetLatent3D(nn.Module):
    """in_ch: latent_ch (denoised) + cond channels. out_ch: latent_ch (v-pred)."""

    def __init__(self, in_ch, out_ch, model_channels=128, channel_mult=(1, 2, 4),
                 num_res_blocks=2, num_heads=4, dropout=0.0, attn_at_lowest_only=True):
        super().__init__()
        emb_dim = model_channels * 4
        self.model_channels = model_channels
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim),
        )

        ch = model_channels
        n_levels = len(channel_mult)
        # each entry of input_blocks/output_blocks is itself a small ModuleList
        # ("unit") run sequentially; one hs push/pop per unit, exactly mirroring
        # ref/unet.py's TimestepEmbedSequential grouping.
        self.input_blocks = nn.ModuleList([nn.ModuleList([nn.Conv3d(in_ch, ch, 3, padding=1)])])
        input_block_chans = [ch]

        for level, mult in enumerate(channel_mult):
            out_c = model_channels * mult
            is_lowest = level == n_levels - 1
            for _ in range(num_res_blocks):
                unit = nn.ModuleList([ResBlock3D(ch, out_c, emb_dim, dropout)])
                ch = out_c
                if is_lowest and attn_at_lowest_only:
                    unit.append(SelfAttention3D(ch, num_heads))
                self.input_blocks.append(unit)
                input_block_chans.append(ch)
            if not is_lowest:
                self.input_blocks.append(nn.ModuleList([Downsample3D(ch)]))
                input_block_chans.append(ch)

        self.mid_block1 = ResBlock3D(ch, ch, emb_dim, dropout)
        self.mid_attn = SelfAttention3D(ch, num_heads)
        self.mid_block2 = ResBlock3D(ch, ch, emb_dim, dropout)

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_c = model_channels * mult
            is_lowest = level == n_levels - 1
            for i in range(num_res_blocks + 1):
                skip_c = input_block_chans.pop()
                unit = nn.ModuleList([ResBlock3D(ch + skip_c, out_c, emb_dim, dropout)])
                ch = out_c
                if is_lowest and attn_at_lowest_only:
                    unit.append(SelfAttention3D(ch, num_heads))
                if level != 0 and i == num_res_blocks:
                    unit.append(Upsample3D(ch))
                self.output_blocks.append(unit)

        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_act = nn.SiLU()
        self.to_out = nn.Conv3d(ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    @staticmethod
    def _run_unit(unit, h, emb):
        for layer in unit:
            h = layer(h, emb) if isinstance(layer, ResBlock3D) else layer(h)
        return h

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hs = []
        h = x
        for unit in self.input_blocks:
            h = self._run_unit(unit, h, emb)
            hs.append(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for unit in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = unit[0](h, emb)  # ResBlock always first
            for layer in unit[1:]:
                h = layer(h)

        h = self.out_act(self.out_norm(h))
        return self.to_out(h)
