#!/usr/bin/env python3
"""Train the 3D VAE on healthy lung signed distance maps (L1 + soft Dice + small KL).

    torchrun --nproc_per_node=$NGPU train_vae.py --config configs/vae.yaml --run_dir runs/vae
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
from torch.utils.data import DataLoader, DistributedSampler

from dataset_vae import LungSDTDataset
from vae3d import VAE3D, widths_for_latent_res
from sdt import soft_mask_torch, sdt_to_mask, compute_sdt

DEFAULT_CONFIG = {
    "npy_dir": "data/npy128",
    "splits_json": "data/splits.json",
    "sdt_clip_mm": 20.0,
    "latent_ch": 8,
    "latent_res": 16,
    "stem_ch": 32,
    "lr": 2e-4,
    "batch_size": 4,  # per GPU
    "total_steps": 30000,
    "val_every": 1000,
    "ckpt_every": 2000,
    "log_every": 10,
    "ema_decay": 0.999,
    "lambda_dice": 1.0,
    "beta_kl": 1e-6,
    "tau_mm": 2.0,
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


def dice_loss(pred_soft, target_soft, eps=1e-6):
    dims = (1, 2, 3, 4)
    inter = (pred_soft * target_soft).sum(dims)
    union = pred_soft.sum(dims) + target_soft.sum(dims)
    return 1.0 - ((2 * inter + eps) / (union + eps)).mean()


def dice_score(pred_mask, target_mask, eps=1e-6):
    dims = (1, 2, 3, 4)
    inter = (pred_mask & target_mask).sum(dims).float()
    union = pred_mask.sum(dims).float() + target_mask.sum(dims).float()
    return ((2 * inter + eps) / (union + eps)).mean().item()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow


def build_model(cfg):
    widths = widths_for_latent_res(cfg["latent_res"])
    return VAE3D(in_ch=1, stem_ch=cfg["stem_ch"], widths=widths, latent_ch=cfg["latent_ch"])


def load_splits(splits_json, debug=False, debug_n=2):
    splits = json.loads(Path(splits_json).read_text())
    train_pids, val_pids = list(splits["train"]), list(splits["val"])
    if debug:
        train_pids = train_pids[:debug_n]
        val_pids = val_pids[:debug_n]
    return train_pids, val_pids


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="yaml file, overrides DEFAULT_CONFIG")
    ap.add_argument("--run_dir", default="runs/vae_debug")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--encode_all", action="store_true")
    ap.add_argument("--ckpt", default=None, help="checkpoint to load for --encode_all")
    for k, v in DEFAULT_CONFIG.items():
        ap.add_argument(f"--{k}", type=type(v), default=None)
    return ap.parse_args()


def resolve_config(args):
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    for k in DEFAULT_CONFIG:
        val = getattr(args, k)
        if val is not None:
            cfg[k] = val
    if args.debug:
        cfg["total_steps"] = 5
        cfg["val_every"] = 5
        cfg["ckpt_every"] = 5
        cfg["log_every"] = 1
        cfg["batch_size"] = 1
        cfg["num_workers"] = 0
    return cfg


@torch.no_grad()
def validate(raw_model, val_loader, device):
    raw_model.eval()
    dices, latent_stds = [], []
    for batch in val_loader:
        sdt = batch["sdt"].to(device)
        mu, _ = raw_model.encode(sdt)
        rec = raw_model.decode(mu)
        dices.append(dice_score(sdt_to_mask(rec), sdt_to_mask(sdt)))
        latent_stds.append(mu.std().item())
    raw_model.train()
    return float(np.mean(dices)), float(np.mean(latent_stds))


def train(cfg, run_dir, debug):
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
            loss_log_path.write_text("step,loss,l1,dice_loss,kl,lr,latent_std,elapsed_s\n")
        if not val_log_path.exists():
            val_log_path.write_text("step,val_dice,latent_std\n")

    train_pids, val_pids = load_splits(cfg["splits_json"], debug=debug, debug_n=2)
    train_ds = LungSDTDataset(cfg["npy_dir"], train_pids, sdt_clip_mm=cfg["sdt_clip_mm"],
                               augment=True, base_seed=cfg["seed"])
    val_ds = LungSDTDataset(cfg["npy_dir"], val_pids, sdt_clip_mm=cfg["sdt_clip_mm"],
                             augment=False, base_seed=cfg["seed"] + 999_999)
    val_loader = DataLoader(val_ds, batch_size=max(1, cfg["batch_size"]), shuffle=False,
                             num_workers=cfg["num_workers"])

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
    start_time = time.time()

    while global_step < cfg["total_steps"]:
        train_ds.set_epoch(epoch)
        if ddp:
            sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                          shuffle=True, seed=cfg["seed"] + epoch)
            loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                                 num_workers=cfg["num_workers"], drop_last=True)
        else:
            loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                 num_workers=cfg["num_workers"], drop_last=(len(train_ds) > cfg["batch_size"]))

        for batch in loader:
            sdt = batch["sdt"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                rec, mu, logvar = model(sdt)
                l1 = F.l1_loss(rec, sdt)
                soft_rec = soft_mask_torch(rec, tau_mm=cfg["tau_mm"])
                soft_tgt = soft_mask_torch(sdt, tau_mm=cfg["tau_mm"])
                dloss = dice_loss(soft_rec, soft_tgt)
                kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
                loss = l1 + cfg["lambda_dice"] * dloss + cfg["beta_kl"] * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(raw_model)

            global_step += 1

            if is_main and (global_step % cfg["log_every"] == 0 or global_step == 1):
                elapsed = time.time() - start_time
                with open(loss_log_path, "a") as f:
                    f.write(f"{global_step},{loss.item():.6f},{l1.item():.6f},{dloss.item():.6f},"
                            f"{kl.item():.6f},{cfg['lr']:.6e},{mu.std().item():.6f},{elapsed:.1f}\n")
                print(f"step {global_step}/{cfg['total_steps']} loss {loss.item():.4f} "
                      f"l1 {l1.item():.4f} dice_loss {dloss.item():.4f} kl {kl.item():.4f}")

            do_val = global_step % cfg["val_every"] == 0 or global_step == cfg["total_steps"]
            if do_val:
                val_dice, latent_std = validate(raw_model, val_loader, device)
                if is_main:
                    print(f"[val] step {global_step} dice {val_dice:.4f} latent_std {latent_std:.4f}")
                    with open(val_log_path, "a") as f:
                        f.write(f"{global_step},{val_dice:.6f},{latent_std:.6f}\n")
                    ckpt = {"model": raw_model.state_dict(), "ema": ema.state_dict(),
                            "step": global_step, "cfg": cfg, "val_dice": val_dice}
                    torch.save(ckpt, ckpt_dir / "last.pt")
                    if val_dice > best_dice:
                        best_dice = val_dice
                        torch.save(ckpt, ckpt_dir / "best.pt")
            elif global_step % cfg["ckpt_every"] == 0 and is_main:
                torch.save({"model": raw_model.state_dict(), "ema": ema.state_dict(),
                            "step": global_step, "cfg": cfg}, ckpt_dir / "last.pt")

            if global_step >= cfg["total_steps"]:
                break
        epoch += 1

    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print(f"training complete: {global_step} steps, best val dice {best_dice:.4f}")


def encode_all(cfg, run_dir, ckpt_path):
    """Encode all clean (unaugmented) patients' lung SDTs with mu. Writes data/latents/{pid}_mu.npy and
    runs/<name>/latent_stats.json (latent_std, used by stage 2 scaling)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    splits = json.loads(Path(cfg["splits_json"]).read_text())
    all_pids = list(splits["train"]) + list(splits["val"])
    out_dir = Path("data/latents")
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_dir = Path(cfg["npy_dir"])
    all_std = []
    with torch.no_grad():
        for pid in all_pids:
            lung = np.load(npy_dir / f"{pid}_lung.npy")
            meta = json.loads((npy_dir / f"{pid}_meta.json").read_text())
            spacing_mm = np.array(meta["resampled_spacing_mm"], dtype=np.float64)
            sdt = compute_sdt(lung, spacing_mm, clip_mm=cfg["sdt_clip_mm"])
            sdt_t = torch.from_numpy(sdt).unsqueeze(0).unsqueeze(0).to(device)
            mu, _ = model.encode(sdt_t)
            mu_np = mu.squeeze(0).cpu().numpy().astype(np.float32)
            np.save(out_dir / f"{pid}_mu.npy", mu_np)
            all_std.append(float(mu_np.std()))

    latent_std = float(np.mean(all_std))
    stats = {"latent_std": latent_std, "n_patients": len(all_pids), "ckpt": str(ckpt_path)}
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(run_dir) / "latent_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"encoded {len(all_pids)} patients -> {out_dir}, latent_std={latent_std:.4f}")


def main():
    args = parse_args()
    cfg = resolve_config(args)

    if args.encode_all:
        if not args.ckpt:
            raise SystemExit("--encode_all requires --ckpt <path>")
        encode_all(cfg, args.run_dir, args.ckpt)
        return

    train(cfg, args.run_dir, args.debug)


if __name__ == "__main__":
    main()
