#!/usr/bin/env python3
"""Train the latent diffusion model (v-prediction, cosine schedule, lung-channel dropout for guidance) on top of a frozen VAE.

    torchrun --nproc_per_node=$NGPU train_ldm.py --config configs/ldm_stage1.yaml --run_dir runs/ldm
    # stage 2: fine-tune with medial corruptions
    torchrun --nproc_per_node=$NGPU train_ldm.py --config configs/ldm_stage2_medial.yaml --run_dir runs/ldm_medial \
        --corruption_medial --init_from runs/ldm/ckpt/last.pt
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import yaml
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, DistributedSampler

from corruption_v2 import mediastinum_proxy
from dataset_ldm import LatentDiffusionDataset
from unet_latent import UNetLatent3D
from vae3d import VAE3D, widths_for_latent_res
from diffusion import GaussianDiffusion
from sdt import sdt_to_mask
from train_vae import EMA, dice_score

DEFAULT_CONFIG = {
    "npy_dir": "data/npy128",
    "splits_json": "data/splits.json",
    "sdt_clip_mm": 20.0,
    "vae_ckpt": "runs/vae/ckpt/best.pt",
    "vae_latent_res": 16,
    "vae_latent_ch": 8,
    "vae_stem_ch": 32,
    "latent_stats_json": "runs/vae/latent_stats.json",
    "model_channels": 128,
    "channel_mult": [1, 2, 4],
    "num_res_blocks": 2,
    "num_heads": 4,
    "T": 1000,
    "cfg_dropout_p": 0.2,
    "lr": 1e-4,
    "batch_size": 8,  # per GPU
    "total_steps": 100000,
    "val_every": 2000,
    "ckpt_every": 2000,
    "log_every": 10,
    "ema_decay": 0.9999,
    "n_val_patients": 20,
    "ddim_steps": 50,
    "cfg_scale": 1.5,
    "decode_dice_every": 0,  # 0 = disabled; logging only
    "num_workers": 4,
    "seed": 0,
}


def is_ddp_run():
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def build_model(cfg):
    latent_ch = cfg["vae_latent_ch"]
    in_ch = latent_ch * 2 + 2  # z_t + c_lung + c_thorax(1) + c_dist(1)
    return UNetLatent3D(in_ch=in_ch, out_ch=latent_ch, model_channels=cfg["model_channels"],
                         channel_mult=tuple(cfg["channel_mult"]), num_res_blocks=cfg["num_res_blocks"],
                         num_heads=cfg["num_heads"])


def load_frozen_vae(cfg, device):
    widths = widths_for_latent_res(cfg["vae_latent_res"])
    vae = VAE3D(in_ch=1, stem_ch=cfg["vae_stem_ch"], widths=widths, latent_ch=cfg["vae_latent_ch"]).to(device)
    state = torch.load(cfg["vae_ckpt"], map_location=device, weights_only=False)
    vae.load_state_dict(state["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def load_latent_std(cfg):
    stats = json.loads(Path(cfg["latent_stats_json"]).read_text())
    return float(stats["latent_std"])


def load_splits(splits_json, debug=False, debug_n=2):
    splits = json.loads(Path(splits_json).read_text())
    train_pids, val_pids = list(splits["train"]), list(splits["val"])
    if debug:
        train_pids = train_pids[:debug_n]
        val_pids = val_pids[:debug_n]
    return train_pids, val_pids


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run_dir", default="runs/ldm_debug")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume model+EMA+step from <run_dir>/ckpt/last.pt (optimizer restarts fresh)")
    ap.add_argument("--channel_mult", type=int, nargs="+", default=None)
    ap.add_argument("--init_from", default=None,
                    help="fine-tune: init model+EMA+step from this ckpt (another run's last.pt; "
                         "never written to). Optimizer starts fresh (ckpts hold no optimizer state).")
    ap.add_argument("--ft_steps", type=int, default=None,
                    help="with --init_from: train this many steps past the ckpt step (total_steps = step + ft_steps)")
    ap.add_argument("--corruption_medial", action="store_true",
                    help="set corruption.medial=true (medial corruption ops, train set only)")
    for k, v in DEFAULT_CONFIG.items():
        if k == "channel_mult":
            continue
        ap.add_argument(f"--{k}", type=type(v), default=None)
    return ap.parse_args()


def resolve_config(args):
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for k in DEFAULT_CONFIG:
        if k == "channel_mult":
            continue
        val = getattr(args, k)
        if val is not None:
            cfg[k] = val
    if args.channel_mult is not None:
        cfg["channel_mult"] = args.channel_mult
    if args.corruption_medial:
        cfg["corruption"] = {**(cfg.get("corruption") or {}), "medial": True}
    if args.debug:
        cfg["total_steps"] = 5
        if args.ft_steps is not None:
            args.ft_steps = 5
        cfg["val_every"] = 5
        cfg["ckpt_every"] = 5
        cfg["log_every"] = 1
        cfg["batch_size"] = 1
        cfg["num_workers"] = 0
        cfg["ddim_steps"] = 10  # keep the debug sampling pass fast
    return cfg


def average_surface_distance_mm(pred_mask, gt_mask, spacing_mm):
    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    gt_surface = gt_mask ^ binary_erosion(gt_mask)
    if not pred_surface.any() or not gt_surface.any():
        return float("nan")
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing_mm)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing_mm)
    return float(np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]]).mean())


MEDIAL_BAND_MM = 20.0


def medial_band_surface_distance_mm(pred_mask, gt_mask, spacing_mm, band_mm=MEDIAL_BAND_MM):
    """Symmetric surface distance restricted to surface voxels (of pred and of GT)
    within band_mm of the mediastinum proxy built from the GT (clean) mask.
    NaN if the proxy or the band is empty."""
    med = mediastinum_proxy(gt_mask)
    if not med.any():
        return float("nan")
    band = distance_transform_edt(~med, sampling=spacing_mm) <= band_mm
    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    gt_surface = gt_mask ^ binary_erosion(gt_mask)
    ps, gs = pred_surface & band, gt_surface & band
    if not ps.any() or not gs.any():
        return float("nan")
    dt_gt = distance_transform_edt(~gt_surface, sampling=spacing_mm)
    dt_pred = distance_transform_edt(~pred_surface, sampling=spacing_mm)
    return float(np.concatenate([dt_gt[ps], dt_pred[gs]]).mean())


def make_model_fn(raw_model):
    def model_fn(z_t, t, cond, drop_cond):
        c_lung = torch.zeros_like(cond["c_lung"]) if drop_cond else cond["c_lung"]
        inp = torch.cat([z_t, c_lung, cond["c_thorax"], cond["c_dist"]], dim=1)
        return raw_model(inp, t)
    return model_fn


@torch.no_grad()
def validate(raw_model, vae, diffusion, val_loader, device, cfg, latent_std):
    raw_model.eval()
    dices, asds, vol_ratios, fp_vols, fn_vols, medial_asds = [], [], [], [], [], []
    model_fn = make_model_fn(raw_model)

    for batch in val_loader:
        sdt_clean = batch["sdt_clean"].to(device)
        sdt_corr = batch["sdt_corr"].to(device)
        thorax_cond = batch["thorax_cond"].to(device)
        dist_cond = batch["dist_cond"].to(device)
        spacing_mm = batch["spacing_mm"].numpy()

        mu_corr, _ = vae.encode(sdt_corr)
        c_lung = mu_corr / latent_std
        cond = {"c_lung": c_lung, "c_thorax": thorax_cond, "c_dist": dist_cond}

        z_sample = diffusion.ddim_sample(model_fn, c_lung.shape, cond, device,
                                          steps=cfg["ddim_steps"], cfg_scale=cfg["cfg_scale"])
        rec_sdt = vae.decode(z_sample * latent_std)
        pred_mask = sdt_to_mask(rec_sdt).cpu().numpy()
        gt_mask = sdt_to_mask(sdt_clean).cpu().numpy()

        for b in range(pred_mask.shape[0]):
            pm, gm, sp = pred_mask[b, 0], gt_mask[b, 0], spacing_mm[b]
            inter = int((pm & gm).sum())
            union = int(pm.sum()) + int(gm.sum())
            dices.append((2 * inter + 1e-6) / (union + 1e-6))
            vol_pred, vol_gt = int(pm.sum()), int(gm.sum())
            vol_ratios.append(vol_pred / max(vol_gt, 1))
            fp_vols.append(int((pm & ~gm).sum()))
            fn_vols.append(int((~pm & gm).sum()))
            asds.append(average_surface_distance_mm(pm, gm, sp))
            medial_asds.append(medial_band_surface_distance_mm(pm, gm, sp))

    raw_model.train()
    return {
        "dice": float(np.mean(dices)),
        "asd_mm": float(np.nanmean(asds)),
        "vol_ratio": float(np.mean(vol_ratios)),
        "fp_vox": float(np.mean(fp_vols)),
        "fn_vox": float(np.mean(fn_vols)),
        "grow_shrink_ratio": float(np.mean(fp_vols) / max(np.mean(fn_vols), 1e-6)),
        "medial_asd_mm": float(np.nanmean(medial_asds)) if np.isfinite(medial_asds).any() else float("nan"),
    }


def save_debug_sample_png(raw_model, vae, diffusion, val_ds, device, cfg, latent_std, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    item = val_ds[0]
    sdt_clean = item["sdt_clean"].unsqueeze(0).to(device)
    sdt_corr = item["sdt_corr"].unsqueeze(0).to(device)
    thorax_cond = item["thorax_cond"].unsqueeze(0).to(device)
    dist_cond = item["dist_cond"].unsqueeze(0).to(device)

    raw_model.eval()
    with torch.no_grad():
        mu_corr, _ = vae.encode(sdt_corr)
        c_lung = mu_corr / latent_std
        cond = {"c_lung": c_lung, "c_thorax": thorax_cond, "c_dist": dist_cond}
        model_fn = make_model_fn(raw_model)
        z_sample = diffusion.ddim_sample(model_fn, c_lung.shape, cond, device,
                                          steps=cfg["ddim_steps"], cfg_scale=cfg["cfg_scale"])
        rec_sdt = vae.decode(z_sample * latent_std)
    raw_model.train()

    pred_mask = sdt_to_mask(rec_sdt)[0, 0].cpu().numpy()
    corr_mask = sdt_to_mask(sdt_corr)[0, 0].cpu().numpy()
    clean_mask = sdt_to_mask(sdt_clean)[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mid = pred_mask.shape[0] // 2
    for ax, vol, title in zip(axes, [corr_mask, pred_mask, clean_mask],
                               ["corrupted input", "generated output", "clean GT"]):
        ax.imshow(vol[mid], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"pid={item['pid']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote debug sample PNG: {out_path}")


def train(cfg, run_dir, debug, resume=False, init_from=None, ft_steps=None):
    ddp = is_ddp_run()
    if ddp:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank, world_size = 0, 1
    is_main = rank == 0

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "ckpt"
    loss_log_path = run_dir / "loss_log.csv"
    val_log_path = run_dir / "val_log.csv"
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f)
        if not loss_log_path.exists():
            loss_log_path.write_text("step,loss,lr,decode_dice,elapsed_s\n")
        if not val_log_path.exists():
            val_log_path.write_text("step,val_dice,asd_mm,vol_ratio,fp_vox,fn_vox,grow_shrink_ratio,"
                                    "medial_asd_mm\n")
    # older runs' val_log.csv has no medial column: keep their format on resume
    if ddp:
        dist.barrier()
    val_has_medial = "medial_asd_mm" in val_log_path.read_text().splitlines()[0]

    train_pids, val_pids_all = load_splits(cfg["splits_json"], debug=debug, debug_n=2)
    val_pids = val_pids_all if debug else val_pids_all[: cfg["n_val_patients"]]

    train_ds = LatentDiffusionDataset(cfg["npy_dir"], train_pids, latent_res=cfg["vae_latent_res"],
                                       sdt_clip_mm=cfg["sdt_clip_mm"], base_seed=cfg["seed"],
                                       fixed_corruption=False,
                                       corruption_config=cfg.get("corruption"))
    val_ds = LatentDiffusionDataset(cfg["npy_dir"], val_pids, latent_res=cfg["vae_latent_res"],
                                     sdt_clip_mm=cfg["sdt_clip_mm"], fixed_corruption=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, min(cfg["batch_size"], len(val_ds))),
                             shuffle=False, num_workers=cfg["num_workers"])

    vae = load_frozen_vae(cfg, device)
    latent_std = load_latent_std(cfg)
    diffusion = GaussianDiffusion(T=cfg["T"], device=device)

    model = build_model(cfg).to(device)
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    ema = EMA(raw_model, cfg["ema_decay"])

    use_amp = device.type == "cuda"
    global_step = 0
    epoch = 0
    best_dice = -1.0
    if resume:
        # last.pt holds model+EMA+step only (no optimizer state): AdamW moments restart.
        state = torch.load(ckpt_dir / "last.pt", map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        global_step = int(state["step"])
        steps_per_epoch = max(1, len(train_ds) // (cfg["batch_size"] * world_size))
        epoch = global_step // steps_per_epoch
        val_rows = [l.split(",") for l in val_log_path.read_text().splitlines()[1:] if l]
        best_dice = max([float(r[1]) for r in val_rows if int(r[0]) <= global_step], default=-1.0)
        if is_main:
            # drop log rows written after last.pt so the CSVs stay monotone in step
            for pth in (loss_log_path, val_log_path):
                lines = pth.read_text().splitlines()
                keep = [lines[0]] + [l for l in lines[1:] if l and int(l.split(",")[0]) <= global_step]
                pth.write_text("\n".join(keep) + "\n")
            print(f"resumed from last.pt: step {global_step}, epoch {epoch}, best_dice {best_dice:.4f}")
    elif init_from:
        if Path(init_from).resolve().parent.parent == run_dir.resolve():
            raise ValueError("--init_from must point to another run's checkpoint")
        state = torch.load(init_from, map_location=device, weights_only=False)
        raw_model.load_state_dict(state["model"])
        ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        global_step = int(state["step"])
        steps_per_epoch = max(1, len(train_ds) // (cfg["batch_size"] * world_size))
        epoch = global_step // steps_per_epoch
        cfg["total_steps"] = global_step + int(ft_steps)
        cfg["init_from"], cfg["init_step"] = str(init_from), global_step
        if is_main:
            with open(run_dir / "config.yaml", "w") as f:
                yaml.safe_dump(cfg, f)
            print(f"fine-tune init from {init_from}: step {global_step} -> {cfg['total_steps']} "
                  f"(epoch {epoch}); optimizer fresh")
    if (resume or init_from) and cfg["total_steps"] <= global_step:
        raise ValueError(f"total_steps {cfg['total_steps']} <= start step {global_step}")
    start_time = time.time()

    def run_val(step):
        nonlocal best_dice
        metrics = validate(raw_model, vae, diffusion, val_loader, device, cfg, latent_std)
        if not is_main:
            return
        print(f"[val] step {step} dice {metrics['dice']:.4f} "
              f"asd_mm {metrics['asd_mm']:.2f} vol_ratio {metrics['vol_ratio']:.3f} "
              f"grow_shrink {metrics['grow_shrink_ratio']:.3f} medial_asd_mm {metrics['medial_asd_mm']:.2f}")
        with open(val_log_path, "a") as f:
            f.write(f"{step},{metrics['dice']:.6f},{metrics['asd_mm']:.4f},"
                    f"{metrics['vol_ratio']:.6f},{metrics['fp_vox']:.2f},"
                    f"{metrics['fn_vox']:.2f},{metrics['grow_shrink_ratio']:.6f}"
                    + (f",{metrics['medial_asd_mm']:.4f}" if val_has_medial else "") + "\n")
        return metrics

    if init_from and not resume:
        run_val(global_step)  # baseline of the init checkpoint on the same seeded val set

    while global_step < cfg["total_steps"]:
        train_ds.set_epoch(epoch)
        if ddp:
            sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                          shuffle=True, seed=cfg["seed"] + epoch)
            loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                                 num_workers=cfg["num_workers"], drop_last=True)
        else:
            loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                 num_workers=cfg["num_workers"],
                                 drop_last=(len(train_ds) > cfg["batch_size"]))

        for batch in loader:
            sdt_clean = batch["sdt_clean"].to(device, non_blocking=True)
            sdt_corr = batch["sdt_corr"].to(device, non_blocking=True)
            thorax_cond = batch["thorax_cond"].to(device, non_blocking=True)
            dist_cond = batch["dist_cond"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                with torch.no_grad():
                    mu_clean, _ = vae.encode(sdt_clean)
                    z0 = mu_clean / latent_std
                    mu_corr, _ = vae.encode(sdt_corr)
                    c_lung = mu_corr / latent_std

                b = z0.shape[0]
                drop_mask = (torch.rand(b, device=device) < cfg["cfg_dropout_p"]).view(b, 1, 1, 1, 1)
                c_lung_used = torch.where(drop_mask, torch.zeros_like(c_lung), c_lung)

                t = torch.randint(0, diffusion.T, (b,), device=device)
                noise = torch.randn_like(z0)
                z_t = diffusion.q_sample(z0, t, noise)
                v_target = diffusion.v_from_x0_noise(z0, noise, t)

                inp = torch.cat([z_t, c_lung_used, thorax_cond, dist_cond], dim=1)
                v_pred = model(inp, t)
                loss = F.mse_loss(v_pred, v_target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(raw_model)

            global_step += 1

            decode_dice_val = float("nan")
            if cfg["decode_dice_every"] > 0 and global_step % cfg["decode_dice_every"] == 0:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    with torch.no_grad():
                        x0_pred = diffusion.x0_from_v(z_t, v_pred, t)
                        rec_sdt = vae.decode(x0_pred * latent_std)
                        decode_dice_val = dice_score(sdt_to_mask(rec_sdt), sdt_to_mask(sdt_clean))

            if is_main and (global_step % cfg["log_every"] == 0 or global_step == 1):
                elapsed = time.time() - start_time
                with open(loss_log_path, "a") as f:
                    f.write(f"{global_step},{loss.item():.6f},{cfg['lr']:.6e},"
                            f"{decode_dice_val:.4f},{elapsed:.1f}\n")
                print(f"step {global_step}/{cfg['total_steps']} loss {loss.item():.4f}")

            do_val = global_step % cfg["val_every"] == 0 or global_step == cfg["total_steps"]
            if do_val:
                metrics = run_val(global_step)
                if is_main:
                    ckpt = {"model": raw_model.state_dict(), "ema": ema.state_dict(),
                            "step": global_step, "cfg": cfg, "val_dice": metrics["dice"]}
                    torch.save(ckpt, ckpt_dir / "last.pt")
                    if metrics["dice"] > best_dice:
                        best_dice = metrics["dice"]
                        torch.save(ckpt, ckpt_dir / "best.pt")
            elif global_step % cfg["ckpt_every"] == 0 and is_main:
                torch.save({"model": raw_model.state_dict(), "ema": ema.state_dict(),
                            "step": global_step, "cfg": cfg}, ckpt_dir / "last.pt")

            if global_step >= cfg["total_steps"]:
                break
        epoch += 1

    if debug and is_main:
        save_debug_sample_png(raw_model, vae, diffusion, val_ds, device, cfg, latent_std,
                               run_dir / "debug_sample.png")

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print(f"training complete: {global_step} steps, best val dice {best_dice:.4f}")


def main():
    args = parse_args()
    cfg = resolve_config(args)
    if args.init_from and args.ft_steps is None:
        raise SystemExit("--init_from requires --ft_steps")
    train(cfg, args.run_dir, args.debug, resume=args.resume, init_from=args.init_from, ft_steps=args.ft_steps)


if __name__ == "__main__":
    main()
