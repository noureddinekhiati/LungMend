#!/usr/bin/env python3
"""Build the 128^3 training arrays from the healthy cohort.

For each patient in data/raw/<pid>/ (lung.nii.gz = R231 mask, thorax.nii.gz = bone mask): keep the two lung components,
crop around the lung with a margin, resample lung and bone to 128^3 and write data/npy128/<pid>_{lung,thorax}.npy plus
<pid>_meta.json (crop box, spacings). Patients listed in data/raw/exclude.txt are skipped.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label, binary_fill_holes, zoom as ndi_zoom


def load_mask_pair(pid_dir):
    lung_img = nib.load(str(pid_dir / "lung.nii.gz"))
    thorax_img = nib.load(str(pid_dir / "thorax.nii.gz"))

    if lung_img.shape != thorax_img.shape:
        raise ValueError(f"shape mismatch lung{lung_img.shape} thorax{thorax_img.shape}")
    if not np.allclose(lung_img.affine, thorax_img.affine, atol=1e-3):
        raise ValueError(f"affine mismatch\nlung:\n{lung_img.affine}\nthorax:\n{thorax_img.affine}")

    lung = np.asanyarray(lung_img.dataobj).astype(np.uint8)
    thorax = np.asanyarray(thorax_img.dataobj).astype(np.uint8)
    spacing = np.abs(np.array(lung_img.header.get_zooms()[:3], dtype=np.float64))
    return lung, thorax, spacing, lung_img.shape


def clean_lung(lung):
    """Keep the two largest components; fill holes per axial (axis-0) slice.
    Returns (cleaned_mask, n_components_before, n_components_kept, merged_or_single_lung)."""
    labeled, n_before = cc_label(lung)
    if n_before == 0:
        return lung.copy(), 0, 0, True

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background is never a component to keep
    order = np.argsort(sizes)[::-1]
    largest = order[0]
    keep = {int(largest)}
    merged_or_single = True
    if len(order) > 1 and sizes[order[1]] > 0 and sizes[order[1]] >= 0.15 * sizes[largest]:
        keep.add(int(order[1]))
        merged_or_single = False

    cleaned = np.isin(labeled, list(keep)).astype(np.uint8)

    filled = np.empty_like(cleaned)
    for z in range(cleaned.shape[0]):
        filled[z] = binary_fill_holes(cleaned[z]).astype(np.uint8)

    return filled, int(n_before), len(keep), merged_or_single


def compute_bbox(mask):
    """Axis-agnostic bbox: lo inclusive, hi exclusive, one entry per axis."""
    ndim = mask.ndim
    lo = np.zeros(ndim, dtype=int)
    hi = np.zeros(ndim, dtype=int)
    for ax in range(ndim):
        other_axes = tuple(a for a in range(ndim) if a != ax)
        idx = np.where(mask.any(axis=other_axes))[0]
        if len(idx) == 0:
            raise ValueError("empty lung mask, cannot compute bbox")
        lo[ax] = idx[0]
        hi[ax] = idx[-1] + 1
    return lo, hi


def bbox_with_margin(lung, spacing_mm, margin_mm, vol_shape):
    lo, hi = compute_bbox(lung)
    margin_vox = margin_mm / spacing_mm
    lo = np.floor(lo - margin_vox).astype(int)
    hi = np.ceil(hi + margin_vox).astype(int)
    lo = np.clip(lo, 0, None)
    hi = np.minimum(hi, np.array(vol_shape, dtype=int))
    return lo, hi


def resample_to_target(volume, target_res):
    factors = [target_res / s for s in volume.shape]
    out = ndi_zoom(volume.astype(np.float32), factors, order=0, mode="nearest")
    out = out[:target_res, :target_res, :target_res]
    if out.shape != (target_res,) * 3:
        pad = [(0, target_res - s) for s in out.shape]
        out = np.pad(out, pad, mode="edge")
    return (out > 0.5).astype(np.uint8)


def process_patient(pid, pid_dir, out_dir, target_res, margin_mm):
    lung, thorax, spacing, orig_shape = load_mask_pair(pid_dir)
    cleaned_lung, n_before, n_kept, merged_or_single = clean_lung(lung)

    lo, hi = bbox_with_margin(cleaned_lung, spacing, margin_mm, orig_shape)
    sl = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))

    lung_crop = cleaned_lung[sl]
    thorax_crop = thorax[sl]

    lung_res = resample_to_target(lung_crop, target_res)
    thorax_res = resample_to_target(thorax_crop, target_res)

    crop_extent_vox = (hi - lo).astype(np.float64)
    crop_extent_mm = crop_extent_vox * spacing
    resampled_spacing_mm = (crop_extent_mm / target_res).tolist()

    occupancy = float(lung_res.mean())

    np.save(out_dir / f"{pid}_lung.npy", lung_res)
    np.save(out_dir / f"{pid}_thorax.npy", thorax_res)

    meta = dict(
        pid=pid,
        crop_bbox=[lo.tolist(), hi.tolist()],
        orig_spacing=spacing.tolist(),
        orig_shape=list(orig_shape),
        resampled_spacing_mm=resampled_spacing_mm,
        n_components_before=n_before,
        n_components_kept=n_kept,
        merged_or_single_lung=bool(merged_or_single),
        occupancy=occupancy,
    )
    with open(out_dir / f"{pid}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def write_splits(pids, seed, val_frac, out_path):
    pids = sorted(pids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pids))
    n_val = max(1, int(round(len(pids) * val_frac)))
    val_idx = set(perm[:n_val].tolist())
    val = sorted(pids[i] for i in val_idx)
    train = sorted(p for p in pids if p not in set(val))
    splits = dict(train=train, val=val, seed=seed, val_frac=val_frac)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"splits.json: {len(train)} train, {len(val)} val (seed={seed})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/raw")
    ap.add_argument("--out_dir", default="data/npy128")
    ap.add_argument("--target_res", type=int, default=128)
    ap.add_argument("--margin_mm", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.10)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude_path = data_root / "exclude.txt"
    exclude = set()
    if exclude_path.exists():
        exclude = {l.strip() for l in exclude_path.read_text().splitlines() if l.strip()}
    else:
        print(f"[WARN] {exclude_path} not found — treating exclude list as empty. "
              f"Human should populate it (doublon / exclure_infection / merged-lung case) "
              f"and re-run.")

    pids = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and p.name not in exclude
        and (p / "lung.nii.gz").exists() and (p / "thorax.nii.gz").exists()
    )
    if args.limit:
        pids = pids[: args.limit]

    print(f"Processing {len(pids)} patients (excluded {len(exclude)})...")

    rows = []
    errors = []
    flagged = []
    for i, pid in enumerate(pids):
        pid_dir = data_root / pid
        try:
            meta = process_patient(pid, pid_dir, out_dir, args.target_res, args.margin_mm)
        except Exception as e:
            print(f"[ERROR] {pid}: {e}", file=sys.stderr)
            errors.append((pid, str(e)))
            continue
        rows.append(meta)
        if meta["merged_or_single_lung"]:
            flagged.append(pid)
        print(f"[{i+1}/{len(pids)}] {pid} occ={meta['occupancy']:.3f} "
              f"comp {meta['n_components_before']}->{meta['n_components_kept']}"
              f"{' FLAGGED' if meta['merged_or_single_lung'] else ''}")

    stats_path = out_dir / "stats.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "occupancy", "n_components_before", "n_components_kept",
                    "merged_or_single_lung", "spacing_0", "spacing_1", "spacing_2"])
        for m in rows:
            w.writerow([m["pid"], m["occupancy"], m["n_components_before"], m["n_components_kept"],
                        m["merged_or_single_lung"], *m["resampled_spacing_mm"]])

    if flagged:
        Path("data/flagged.txt").write_text("\n".join(flagged) + "\n")

    if errors:
        (out_dir / "errors.txt").write_text("\n".join(f"{p}\t{e}" for p, e in errors) + "\n")

    occs = [m["occupancy"] for m in rows]
    print(f"\nProcessed {len(rows)} patients, {len(errors)} errors.")
    if occs:
        print(f"Occupancy: min={min(occs):.3f} mean={np.mean(occs):.3f} max={max(occs):.3f}")
    print(f"Flagged (merged/single lung): {len(flagged)}")

    if not args.limit:
        write_splits([m["pid"] for m in rows], args.split_seed, args.val_frac, "data/splits.json")


if __name__ == "__main__":
    main()
