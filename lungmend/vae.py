"""3D VAE that compresses a 128^3 lung signed distance map into a 16^3 x 8 latent (decoder ends with tanh, SDT in [-1, 1])."""
import math

import torch
import torch.nn as nn

DEFAULT_BASE_WIDTHS = (32, 64, 128, 256)


def widths_for_latent_res(latent_res, input_res=128, base_widths=DEFAULT_BASE_WIDTHS):
    n_stages = int(round(math.log2(input_res / latent_res)))
    if n_stages < 1 or n_stages > len(base_widths):
        raise ValueError(f"unsupported latent_res={latent_res} for input_res={input_res}")
    if input_res / (2 ** n_stages) != latent_res:
        raise ValueError(f"latent_res={latent_res} is not input_res={input_res} / 2^k")
    return base_widths[:n_stages]


def _groups(ch, max_groups=8):
    g = min(max_groups, ch)
    while ch % g != 0:
        g -= 1
    return g


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Encoder3D(nn.Module):
    def __init__(self, in_ch, stem_ch, widths, latent_ch):
        super().__init__()
        self.stem = nn.Conv3d(in_ch, stem_ch, 3, padding=1)
        stages = []
        ch = stem_ch
        for w in widths:
            stages.append(ResBlock3D(ch, w))
            stages.append(nn.Conv3d(w, w, 4, stride=2, padding=1))
            ch = w
        self.stages = nn.ModuleList(stages)
        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_act = nn.SiLU()
        self.to_stats = nn.Conv3d(ch, 2 * latent_ch, 3, padding=1)

    def forward(self, x):
        h = self.stem(x)
        for layer in self.stages:
            h = layer(h)
        h = self.out_act(self.out_norm(h))
        mu, logvar = self.to_stats(h).chunk(2, dim=1)
        return mu, logvar


class Decoder3D(nn.Module):
    def __init__(self, out_ch, stem_ch, widths, latent_ch):
        super().__init__()
        widths_rev = list(reversed(widths))
        self.from_latent = nn.Conv3d(latent_ch, widths_rev[0], 3, padding=1)
        stages = []
        ch = widths_rev[0]
        for w in list(widths_rev[1:]) + [stem_ch]:
            stages.append(nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1))
            stages.append(ResBlock3D(ch, w))
            ch = w
        self.stages = nn.ModuleList(stages)
        self.out_norm = nn.GroupNorm(_groups(ch), ch)
        self.out_act = nn.SiLU()
        self.to_out = nn.Conv3d(ch, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.from_latent(z)
        for layer in self.stages:
            h = layer(h)
        h = self.out_act(self.out_norm(h))
        return torch.tanh(self.to_out(h))


class VAE3D(nn.Module):
    def __init__(self, in_ch=1, stem_ch=32, widths=(32, 64, 128), latent_ch=8):
        super().__init__()
        self.encoder = Encoder3D(in_ch, stem_ch, widths, latent_ch)
        self.decoder = Decoder3D(in_ch, stem_ch, widths, latent_ch)
        self.latent_ch = latent_ch

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec = self.decode(z)
        return rec, mu, logvar
