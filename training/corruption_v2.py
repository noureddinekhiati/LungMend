"""Simulated R231 failures applied to healthy lung masks during training.

Operations: peripheral / subpleural erosion, large region drops, cavity holes, truncation, leaks outside the lung and
medial (hilar) drops. Corruptions are drawn on the fly from the rng passed in by the dataset, after augmentation.
"""
import copy

import numpy as np
from skimage.morphology import convex_hull_image
from scipy.ndimage import (
    label as cc_label,
    binary_dilation,
    binary_erosion,
    binary_closing,
    generate_binary_structure,
    distance_transform_edt,
    gaussian_filter,
)

DEFAULT_CONFIG = {
    "near_complete_p": 0.15,
    "near_complete_s_range": (0.0, 0.05),
    "severity_beta_a": 1.2,
    "severity_beta_b": 3.0,
    "severity_scale": 0.45,
    "severity_tol": 0.20,
    "max_ops": 4,
    "op_probs": {
        "region_drop": 0.45,
        "subpleural_erosion": 0.30,
        "cavity_holes": 0.15,
        "truncate": 0.10,
    },
    "region_drop": {"roughen_frac": 0.35},
    "subpleural_erosion": {
        "depth_mm_range": (2.0, 12.0),
        "bone_dist_mm": 25.0,
        "noise_sigma_mm": 8.0,
    },
    "cavity_holes": {"n_range": (1, 4), "diam_mm_range": (10.0, 40.0)},
    "truncate": {},
    "leak_p": 0.10,
    "leak_depth_mm_range": (1.0, 3.0),
    "leak_bone_dist_mm": 30.0,
    "leak_local_radius_mm": 20.0,
    "keep_components": 2,
    "keep_min_frac": 0.15,
    "force_target_s": None,  # debug/viz hook — bypass severity sampling
    # MEDIAL ops (off by default; with medial=False every code path and rng call
    # is exactly the pre-medial one).
    "medial": False,
    "medial_cfg": {
        "hilar_anchor_p": 0.3,          # region_drop uses the hilar anchor with this prob
        "hilar_z_range": (0.35, 0.65),  # fraction of the lung's craniocaudal extent
        "erosion_op_p": 0.15,           # mediastinal_erosion share of the op table
        "sigma_z_mm_range": (15.0, 40.0),
        "noise_sigma_mm": 10.0,
        "r_base_mm": 2.0,
        "r_amp_mm": 8.0,
        "wz_min": 0.15,
        "both_lungs_p": 0.5,
        # hilar region_drop growth (stochastic geodesic, see _stochastic_geodesic_region)
        "grow_noise_sigma_mm": (20.0, 6.0),
        "grow_noise_weights": (0.65, 0.35),
        "grow_p_min": 0.01,
        "grow_p_max": 0.95,
        "grow_max_iter": 2000,
    },
}

ANCHOR_SPECS = {
    # (axis, mode) — axis 0 assumed craniocaudal, axis 1 anterior-posterior,
    # axis 2 left-right (see module docstring). "min"/"max" pick the extreme
    # voxel along that axis within the chosen lung component.
    "apex": (0, "min"),
    "base": (0, "max"),
    "anterior": (1, "min"),
    "posterior": (1, "max"),
    "lateral": (2, "away_from_center"),
}
# weighted toward base + lateral + posterior
ANCHOR_WEIGHTS = {"apex": 0.10, "base": 0.30, "anterior": 0.10, "posterior": 0.25, "lateral": 0.25}


def _merge_config(config):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config:
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
    return cfg


def _struct():
    return generate_binary_structure(3, 1)


def _components(mask):
    return cc_label(mask, structure=_struct())


def _pick_component(mask, rng):
    """Pick a connected component with probability proportional to its size.
    Uniform-over-labels would give a stray 1-voxel speck (left behind by an
    earlier op, e.g. cavity_holes clipping off a sliver) the same odds as the
    real ~1e5-voxel lung lobe, silently turning region_drop/truncate into a
    near no-op."""
    labeled, n = _components(mask)
    if n == 0:
        return None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    comp_ids = np.nonzero(sizes)[0]
    if len(comp_ids) == 0:
        return None
    weights = sizes[comp_ids].astype(np.float64)
    probs = weights / weights.sum()
    chosen = comp_ids[rng.choice(len(comp_ids), p=probs)]
    return labeled == chosen


def _keep_largest(mask, max_components, min_frac):
    labeled, n = _components(mask)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    keep = [order[0]]
    largest_size = sizes[order[0]]
    for i in range(1, min(max_components, len(order))):
        if sizes[order[i]] >= min_frac * largest_size:
            keep.append(order[i])
    return np.isin(labeled, keep).astype(np.uint8)


def _pick_anchor(component_mask, rng, vol_center_axis2):
    names = list(ANCHOR_WEIGHTS.keys())
    probs = np.array([ANCHOR_WEIGHTS[n] for n in names])
    probs = probs / probs.sum()
    name = names[rng.choice(len(names), p=probs)]
    axis, mode = ANCHOR_SPECS[name]
    idx = np.argwhere(component_mask)
    if mode == "min":
        target_val = idx[:, axis].min()
    elif mode == "max":
        target_val = idx[:, axis].max()
    else:  # away_from_center
        comp_center = idx[:, axis].mean()
        target_val = idx[:, axis].max() if comp_center > vol_center_axis2 else idx[:, axis].min()
    candidates = idx[idx[:, axis] == target_val]
    anchor = tuple(candidates[rng.integers(len(candidates))])
    return anchor, name


def _geodesic_region(component_mask, anchor, target_voxels, rng):
    """Grow a topology-respecting (grassfire) ball from `anchor` inside
    component_mask until it holds ~target_voxels. Cropped to the component's
    bbox for speed."""
    idx = np.argwhere(component_mask)
    lo = idx.min(axis=0)
    hi = idx.max(axis=0) + 1
    pad = 2
    lo_p = np.maximum(lo - pad, 0)
    hi_p = np.minimum(hi + pad, np.array(component_mask.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo_p, hi_p))
    sub_mask = component_mask[sl]
    sub_anchor = tuple(int(a - l) for a, l in zip(anchor, lo_p))

    struct = _struct()
    visited = np.zeros(sub_mask.shape, dtype=bool)
    visited[sub_anchor] = True
    frontier = visited.copy()
    total = int(component_mask.sum())
    target = min(int(round(target_voxels)), total)
    count = 1
    while count < target:
        dil = binary_dilation(frontier, structure=struct) & sub_mask & ~visited
        n_new = int(dil.sum())
        if n_new == 0:
            break
        if count + n_new > target:
            new_idx = np.argwhere(dil)
            n_keep = target - count
            keep_sel = rng.choice(len(new_idx), size=n_keep, replace=False)
            visited[tuple(new_idx[keep_sel].T)] = True
            count += n_keep
            break
        visited |= dil
        count += n_new
        frontier = dil

    region_full = np.zeros(component_mask.shape, dtype=bool)
    region_full[sl] = visited
    return region_full


def _roughen(region, rng, noise_frac):
    struct = _struct()
    outer = binary_dilation(region, structure=struct) & ~region
    inner = region & ~binary_erosion(region, structure=struct)
    add_noise = rng.random(region.shape) < noise_frac
    remove_noise = rng.random(region.shape) < noise_frac
    region2 = region.copy()
    region2 |= (outer & add_noise)
    region2 &= ~(inner & remove_noise)
    region2 = binary_closing(region2, structure=struct, iterations=1)
    return region2


def op_region_drop(mask, thorax, rng, spacing_mm, target_voxels, cfg):
    if cfg["medial"] and target_voxels > 0 and rng.random() < cfg["medial_cfg"]["hilar_anchor_p"]:
        out = _hilar_region_drop(mask, rng, spacing_mm, target_voxels, cfg)
        if out is not None:  # None = no hilar contact voxel on either lung -> legacy anchors
            return out
    component_mask = _pick_component(mask, rng)
    if component_mask is None or target_voxels <= 0:
        return mask.copy(), 0, {"op": "region_drop", "skipped": True}
    vol_center_axis2 = mask.shape[2] / 2.0
    anchor, anchor_name = _pick_anchor(component_mask, rng, vol_center_axis2)
    region = _geodesic_region(component_mask, anchor, target_voxels, rng)
    region = _roughen(region, rng, cfg["region_drop"]["roughen_frac"])
    region = region & mask.astype(bool)
    new_mask = mask.astype(bool) & ~region
    removed = int(mask.sum()) - int(new_mask.sum())
    return new_mask.astype(np.uint8), removed, {
        "op": "region_drop", "anchor": anchor_name, "target_voxels": int(target_voxels),
    }


def op_subpleural_erosion(mask, thorax, rng, spacing_mm, target_voxels, cfg):
    if target_voxels <= 0 or mask.sum() == 0:
        return mask.copy(), 0, {"op": "subpleural_erosion", "skipped": True}
    p = cfg["subpleural_erosion"]
    bone_dist_mm = distance_transform_edt(~thorax, sampling=spacing_mm)
    surf_dist_mm = distance_transform_edt(mask.astype(bool), sampling=spacing_mm)
    depth_mm = rng.uniform(*p["depth_mm_range"])
    candidate = mask.astype(bool) & (bone_dist_mm < p["bone_dist_mm"]) & (surf_dist_mm <= depth_mm)
    n_candidate = int(candidate.sum())
    if n_candidate == 0:
        return mask.copy(), 0, {"op": "subpleural_erosion", "skipped": True}

    noise = rng.standard_normal(mask.shape).astype(np.float32)
    sigma_vox = p["noise_sigma_mm"] / max(float(np.mean(spacing_mm)), 1e-6)
    field = gaussian_filter(noise, sigma=sigma_vox)

    coverage = float(np.clip(target_voxels / n_candidate, 0.0, 1.0))
    if coverage >= 1.0:
        remove = candidate
    else:
        thresh = np.quantile(field[candidate], 1.0 - coverage)
        remove = candidate & (field >= thresh)

    new_mask = mask.astype(bool) & ~remove
    removed = int(mask.sum()) - int(new_mask.sum())
    return new_mask.astype(np.uint8), removed, {
        "op": "subpleural_erosion", "depth_mm": float(depth_mm), "coverage": coverage,
    }


def op_cavity_holes(mask, thorax, rng, spacing_mm, target_voxels, cfg):
    if mask.sum() == 0 or target_voxels <= 0:
        return mask.copy(), 0, {"op": "cavity_holes", "skipped": True}
    p = cfg["cavity_holes"]
    surf_dist_mm = distance_transform_edt(mask.astype(bool), sampling=spacing_mm)
    n_holes = int(rng.integers(p["n_range"][0], p["n_range"][1] + 1))
    new_mask = mask.astype(bool).copy()
    removed_total = 0
    cap = target_voxels * 1.3
    spacing_arr = np.asarray(spacing_mm, dtype=np.float64)
    for _ in range(n_holes):
        if removed_total >= cap:
            break
        diam_mm = rng.uniform(*p["diam_mm_range"])
        r_mm = diam_mm / 2.0
        interior = surf_dist_mm > (r_mm + 2.0)
        cand_idx = np.argwhere(interior & new_mask)
        if len(cand_idx) == 0:
            continue
        center = cand_idx[rng.integers(len(cand_idx))]
        rad_vox = np.ceil(r_mm / spacing_arr).astype(int) + 1
        lo = np.maximum(center - rad_vox, 0)
        hi = np.minimum(center + rad_vox + 1, np.array(mask.shape))
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        zz, yy, xx = np.meshgrid(
            (np.arange(lo[0], hi[0]) - center[0]) * spacing_arr[0],
            (np.arange(lo[1], hi[1]) - center[1]) * spacing_arr[1],
            (np.arange(lo[2], hi[2]) - center[2]) * spacing_arr[2],
            indexing="ij",
        )
        r0, r1, r2 = (r_mm * rng.uniform(0.6, 1.0) for _ in range(3))
        ellipsoid = (zz / max(r0, 1e-3)) ** 2 + (yy / max(r1, 1e-3)) ** 2 + (xx / max(r2, 1e-3)) ** 2 <= 1.0
        hole = np.zeros(mask.shape, dtype=bool)
        hole[sl] = ellipsoid
        hole &= new_mask
        removed_total += int(hole.sum())
        new_mask &= ~hole
    return new_mask.astype(np.uint8), removed_total, {"op": "cavity_holes", "n_holes": n_holes}


def op_truncate(mask, thorax, rng, spacing_mm, target_voxels, cfg):
    component_mask = _pick_component(mask, rng)
    if component_mask is None or target_voxels <= 0:
        return mask.copy(), 0, {"op": "truncate", "skipped": True}
    idx = np.argwhere(component_mask)
    normal = np.array([1.0, rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)])
    normal = normal / np.linalg.norm(normal)
    if rng.random() < 0.5:
        normal = -normal
    coords_mm = idx * np.asarray(spacing_mm, dtype=np.float64)
    proj = coords_mm @ normal
    order = np.argsort(proj)
    n = len(idx)
    target = min(int(round(target_voxels)), n)
    if target <= 0:
        return mask.copy(), 0, {"op": "truncate", "skipped": True}
    remove_idx = idx[order[n - target:]]
    new_mask = mask.astype(bool).copy()
    new_mask[tuple(remove_idx.T)] = False
    return new_mask.astype(np.uint8), target, {"op": "truncate", "target_voxels": int(target_voxels)}


def _two_lung_seeds(clean, min_frac=0.15, max_iter=10):
    """Two largest 3D components, each >= min_frac of the lung volume. If the lungs
    touch (e.g. anterior junction merges them at 128^3), erode until they separate.
    Returns (labeled, a, b) or None."""
    struct = _struct()
    m = clean
    for _ in range(max_iter + 1):
        labeled, n = _components(m)
        if n >= 2:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            a, b = np.argsort(sizes)[::-1][:2]
            if sizes[b] >= min_frac * m.sum():
                return labeled, a, b
        m = binary_erosion(m, structure=struct)
        if not m.any():
            return None
    return None


def _lung_labels(clean):
    """Split the CLEAN mask into left/right lungs (1 = lower x centroid, axis 2).
    Seeds = the two largest 3D components (eroded apart if the lungs touch); every
    lung voxel then goes to the nearest seed (EDT). A pure x-midpoint split mislabels
    the medial part of a lung that crosses the midpoint, so it is used only if no two
    seeds can be found. Returns int8 (z,y,x) with 0 = background."""
    lab = np.zeros(clean.shape, dtype=np.int8)
    seeds = _two_lung_seeds(clean)
    if seeds is None:
        idx_x = np.arange(clean.shape[2])[None, None, :]
        mid = np.argwhere(clean)[:, 2].mean()
        lab[clean & (idx_x < mid)] = 1
        lab[clean & (idx_x >= mid)] = 2
        return lab
    labeled, a, b = seeds
    if np.argwhere(labeled == a)[:, 2].mean() > np.argwhere(labeled == b)[:, 2].mean():
        a, b = b, a
    seed_lab = np.zeros(clean.shape, dtype=np.int8)
    seed_lab[labeled == a] = 1
    seed_lab[labeled == b] = 2
    _, nearest = distance_transform_edt(seed_lab == 0, return_indices=True)
    lab[clean] = seed_lab[tuple(nearest[:, clean])]
    return lab


def mediastinum_proxy(clean, lung_lab=None):
    """Per-AXIAL-slice 2D convex hull of (left U right) of the CLEAN mask minus
    the lungs, keeping the largest 2D component per slice (drops small lateral
    concavity notches), stacked along z. Slices containing only one lung -> empty.
    Used only to generate corruption, never as model input."""
    clean = np.asarray(clean, dtype=bool)
    if lung_lab is None:
        lung_lab = _lung_labels(clean)
    med = np.zeros(clean.shape, dtype=bool)
    s2 = generate_binary_structure(2, 1)
    for z in range(clean.shape[0]):
        sl = lung_lab[z]
        if not ((sl == 1).any() and (sl == 2).any()):
            continue
        lungs = sl > 0
        ys, xs = np.nonzero(lungs)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sub = lungs[y0:y1, x0:x1]
        gap = convex_hull_image(sub) & ~sub
        lab2, n2 = cc_label(gap, structure=s2)
        if n2 == 0:
            continue
        sizes = np.bincount(lab2.ravel())
        sizes[0] = 0
        med[z, y0:y1, x0:x1] = lab2 == np.argmax(sizes)
    return med


def _medial_context(clean, spacing_mm):
    clean = np.asarray(clean, dtype=bool)
    lung_lab = _lung_labels(clean)
    med = mediastinum_proxy(clean, lung_lab)
    ctx = {"lung_lab": lung_lab, "med": med, "med_edt_mm": None}
    if med.any():
        ctx["med_edt_mm"] = distance_transform_edt(~med, sampling=spacing_mm)
        ctx["med_contact"] = binary_dilation(med, structure=_struct())
    return ctx


def hilar_candidates(mask, ctx, mcfg):
    """Per lung side (1 = low x, 2 = high x): voxels of `mask` in that clean lung that
    are in 3D 1-voxel contact with the mediastinum proxy and lie within hilar_z_range
    of that lung's (clean) craniocaudal extent. Returns {side: (N,3) int array}."""
    out = {}
    for side in (1, 2):
        lung_s = ctx["lung_lab"] == side
        zs = np.nonzero(lung_s.any(axis=(1, 2)))[0]
        if len(zs) == 0 or ctx["med_edt_mm"] is None:
            out[side] = np.zeros((0, 3), dtype=int)
            continue
        lo = zs.min() + mcfg["hilar_z_range"][0] * (zs.max() - zs.min())
        hi = zs.min() + mcfg["hilar_z_range"][1] * (zs.max() - zs.min())
        cand = np.argwhere(mask.astype(bool) & lung_s & ctx["med_contact"])
        out[side] = cand[(cand[:, 0] >= lo) & (cand[:, 0] <= hi)]
    return out


def _rank_uniform(x):
    r = np.empty(x.size, dtype=np.float64)
    r[np.argsort(x, axis=None, kind="stable")] = np.arange(x.size)
    return (r / max(x.size - 1, 1)).reshape(x.shape)


def _stochastic_geodesic_region(region_mask, anchor, target_voxels, rng, spacing_mm, mcfg):
    """Stochastic geodesic growth inside region_mask: each step, every 6-connected
    frontier voxel is accepted independently with probability p(x) = p_min +
    (p_max - p_min) * smooth_noise(x) (noise sigma in mm), un-accepted frontier voxels
    stay candidates. Gives an irregular, lobulated boundary instead of the L1 ball
    (straight-faced octahedron) of full-frontier grassfire. No plane/height clipping."""
    idx = np.argwhere(region_mask)
    lo = np.maximum(idx.min(axis=0) - 2, 0)
    hi = np.minimum(idx.max(axis=0) + 3, np.array(region_mask.shape))
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    sub = region_mask[sl]
    a = tuple(int(x - l) for x, l in zip(anchor, lo))
    sp = np.asarray(spacing_mm, dtype=np.float64)
    # two-scale smooth noise (lobes + roughness), rank-normalised to a uniform [0,1]
    # marginal: min-max of smoothed noise piles up near 0.5 -> near-constant acceptance
    # -> smooth convex blob with straight faces. Squaring makes low-noise areas slow barriers.
    u = np.zeros(sub.shape, dtype=np.float64)
    for sigma_mm, w in zip(mcfg["grow_noise_sigma_mm"], mcfg["grow_noise_weights"]):
        u += w * _rank_uniform(gaussian_filter(rng.random(sub.shape, dtype=np.float32), sigma=sigma_mm / sp))
    p_acc = mcfg["grow_p_min"] + (mcfg["grow_p_max"] - mcfg["grow_p_min"]) * _rank_uniform(u) ** 2
    struct = _struct()
    visited = np.zeros(sub.shape, dtype=bool)
    visited[a] = True
    target = min(int(round(target_voxels)), int(sub.sum()))
    count = 1
    for _ in range(mcfg["grow_max_iter"]):
        if count >= target:
            break
        cand = binary_dilation(visited, structure=struct) & sub & ~visited
        if not cand.any():
            break
        acc = cand & (rng.random(sub.shape, dtype=np.float32) < p_acc)
        n_new = int(acc.sum())
        if count + n_new > target:
            new_idx = np.argwhere(acc)
            keep = rng.choice(len(new_idx), size=target - count, replace=False)
            visited[tuple(new_idx[keep].T)] = True
            count = target
            break
        visited |= acc
        count += n_new
    full = np.zeros(region_mask.shape, dtype=bool)
    full[sl] = visited
    return full


def _hilar_region_drop(mask, rng, spacing_mm, target_voxels, cfg):
    """Hilar blob: lung side uniform (p=0.5) among sides that have hilar candidates,
    anchor z uniform over the candidate slices of that side (so anchors spread over
    the whole 35-65% window instead of piling up where the contact surface is widest),
    then stochastic geodesic growth inside that lung only + roughening/closing r=1."""
    ctx, mcfg = cfg["_medial_ctx"], cfg["medial_cfg"]
    if ctx["med_edt_mm"] is None:
        return None
    cands = hilar_candidates(mask, ctx, mcfg)
    avail = [s for s in (1, 2) if len(cands[s])]
    if not avail:
        return None
    side = int(rng.integers(1, 3))
    if side not in avail:
        side = avail[0]
    c = cands[side]
    zs = np.unique(c[:, 0])
    z = zs[rng.integers(len(zs))]
    cz = c[c[:, 0] == z]
    anchor = tuple(cz[rng.integers(len(cz))])
    labeled, _ = _components(mask.astype(bool) & (ctx["lung_lab"] == side))
    lung_comp = labeled == labeled[anchor]
    region = _stochastic_geodesic_region(lung_comp, anchor, target_voxels, rng, spacing_mm, mcfg)
    region = _roughen(region, rng, cfg["region_drop"]["roughen_frac"])
    region &= mask.astype(bool) & (ctx["lung_lab"] == side)
    new_mask = mask.astype(bool) & ~region
    removed = int(mask.sum()) - int(new_mask.sum())
    zl = np.nonzero((ctx["lung_lab"] == side).any(axis=(1, 2)))[0]
    return new_mask.astype(np.uint8), removed, {
        "op": "region_drop", "anchor": "hilar", "side": side,
        "anchor_zfrac": float((anchor[0] - zl.min()) / max(zl.max() - zl.min(), 1)),
        "target_voxels": int(target_voxels),
    }


def op_mediastinal_erosion(mask, thorax, rng, spacing_mm, target_voxels, cfg):
    """Band-limited medial peeling with no hard band edges: a gaussian-in-z weight
    times smooth noise modulates a peel radius measured from the mediastinum proxy
    (3D EDT in mm). Capped at target*(1+tol) by keeping the deepest-in-peel voxels."""
    ctx = cfg["_medial_ctx"]
    mcfg = cfg["medial_cfg"]
    if target_voxels <= 0 or mask.sum() == 0 or ctx["med_edt_mm"] is None:
        return mask.copy(), 0, {"op": "mediastinal_erosion", "skipped": True}
    spacing_mm = np.asarray(spacing_mm, dtype=np.float64)
    med_z = np.nonzero(ctx["med"].any(axis=(1, 2)))[0]
    zc = rng.uniform(med_z.min(), med_z.max())
    sigma_z = rng.uniform(*mcfg["sigma_z_mm_range"])
    z_mm = (np.arange(mask.shape[0]) - zc) * spacing_mm[0]
    wz = np.exp(-0.5 * (z_mm / sigma_z) ** 2)[:, None, None]

    noise = gaussian_filter(rng.random(mask.shape, dtype=np.float32),
                            sigma=mcfg["noise_sigma_mm"] / spacing_mm)
    noise = (noise - noise.min()) / max(float(noise.max() - noise.min()), 1e-12)
    r_mm = mcfg["r_base_mm"] + mcfg["r_amp_mm"] * wz * noise

    if rng.random() < mcfg["both_lungs_p"]:
        side = "both"
        sel = ctx["lung_lab"] > 0
    else:
        side_id = int(rng.integers(1, 3))
        side = f"lung{side_id}"
        sel = ctx["lung_lab"] == side_id
    remove = mask.astype(bool) & sel & (ctx["med_edt_mm"] <= r_mm) & (wz > mcfg["wz_min"])

    cap = int(round(target_voxels * (1.0 + cfg["severity_tol"])))
    n_rm = int(remove.sum())
    if n_rm > cap:
        idx = np.argwhere(remove)
        depth = (ctx["med_edt_mm"] - r_mm)[remove]
        drop = idx[np.argsort(depth, kind="stable")[cap:]]
        remove[tuple(drop.T)] = False
    new_mask = mask.astype(bool) & ~remove
    removed = int(mask.sum()) - int(new_mask.sum())
    return new_mask.astype(np.uint8), removed, {
        "op": "mediastinal_erosion", "zc": float(zc), "sigma_z_mm": float(sigma_z),
        "side": side, "capped": n_rm > cap,
    }


OP_FUNCS = {
    "region_drop": op_region_drop,
    "subpleural_erosion": op_subpleural_erosion,
    "cavity_holes": op_cavity_holes,
    "truncate": op_truncate,
    "mediastinal_erosion": op_mediastinal_erosion,
}


def _apply_leak(mask, thorax, rng, spacing_mm, cfg):
    bone_dist_mm = distance_transform_edt(~thorax, sampling=spacing_mm)
    depth_mm = rng.uniform(*cfg["leak_depth_mm_range"])
    struct = _struct()
    iters = max(1, int(round(depth_mm / max(float(np.mean(spacing_mm)), 1e-6))))
    mask_bool = mask.astype(bool)
    dilated = binary_dilation(mask_bool, structure=struct, iterations=iters)
    candidate = dilated & ~mask_bool & (bone_dist_mm > cfg["leak_bone_dist_mm"])
    if not candidate.any():
        return mask
    idxs = np.argwhere(candidate)
    center = idxs[rng.integers(len(idxs))]
    zz, yy, xx = np.indices(mask.shape)
    spacing_arr = np.asarray(spacing_mm, dtype=np.float64)
    dist_mm = np.sqrt(
        ((zz - center[0]) * spacing_arr[0]) ** 2
        + ((yy - center[1]) * spacing_arr[1]) ** 2
        + ((xx - center[2]) * spacing_arr[2]) ** 2
    )
    local = dist_mm <= cfg["leak_local_radius_mm"]
    add = candidate & local
    return (mask_bool | add).astype(np.uint8)


def corrupt(mask, thorax, rng, spacing_mm, config=None):
    """mask, thorax: uint8 128^3 arrays (mask = lung, thorax = bone/thorax).
    Returns (corrupted_mask uint8, info dict)."""
    cfg = _merge_config(config)
    mask = np.asarray(mask, dtype=np.uint8)
    thorax_bool = np.asarray(thorax, dtype=bool)
    spacing_mm = np.asarray(spacing_mm, dtype=np.float64)

    total = int(mask.sum())
    info = {"total_voxels": total, "ops": [], "leak_applied": False, "leak_delta_voxels": 0}
    if total == 0:
        info.update(target_s=0.0, achieved_s=0.0, final_voxels=0)
        return mask.copy(), info

    if cfg["force_target_s"] is not None:
        target_s = float(cfg["force_target_s"])
    elif rng.random() < cfg["near_complete_p"]:
        target_s = float(rng.uniform(*cfg["near_complete_s_range"]))
    else:
        target_s = float(rng.beta(cfg["severity_beta_a"], cfg["severity_beta_b"]) * cfg["severity_scale"])
    info["target_s"] = target_s

    current = mask.copy()
    removed_total = 0
    target_voxels_total = target_s * total
    tol = cfg["severity_tol"]
    op_names = list(cfg["op_probs"].keys())
    op_probs = np.array([cfg["op_probs"][n] for n in op_names], dtype=np.float64)
    op_probs = op_probs / op_probs.sum()
    if cfg["medial"]:
        # proxy from the CLEAN (augmented) mask; no rng consumed here
        cfg["_medial_ctx"] = _medial_context(mask, spacing_mm)
        p_med = cfg["medial_cfg"]["erosion_op_p"]
        op_names = op_names + ["mediastinal_erosion"]
        op_probs = np.append(op_probs * (1.0 - p_med), p_med)

    for _ in range(cfg["max_ops"]):
        if target_voxels_total <= 0:
            break
        removed_frac = removed_total / total
        if removed_frac >= target_s * (1 - tol):
            break
        remaining = target_voxels_total - removed_total
        if remaining <= 0 or int(current.sum()) == 0:
            break
        op_name = op_names[rng.choice(len(op_names), p=op_probs)]
        current, removed_this, meta = OP_FUNCS[op_name](current, thorax_bool, rng, spacing_mm, remaining, cfg)
        removed_total += removed_this
        info["ops"].append(meta)

    if rng.random() < cfg["leak_p"]:
        before = int(current.sum())
        current = _apply_leak(current, thorax_bool, rng, spacing_mm, cfg)
        delta = int(current.sum()) - before
        info["leak_applied"] = True
        info["leak_delta_voxels"] = int(delta)

    current = _keep_largest(current, cfg["keep_components"], cfg["keep_min_frac"])

    final_total = int(current.sum())
    info["achieved_s"] = float(1.0 - final_total / total)
    info["final_voxels"] = final_total
    return current.astype(np.uint8), info
