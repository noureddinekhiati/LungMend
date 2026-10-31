# Ablation, simplified (200 annotated slices, Dice)

Input R231 mask: Dice 0.909.

## A. What each training component contributes

All models in this part are trained with the same ablation protocol (same data, steps and seed); read the change (Δ) with respect to the reference.

| Group | Change | Dice | Δ Dice vs reference | p |
|---|---|---|---|---|
| Reference | Full recipe (final VAE) | 0.940 | — | |
| Reference (VAE rows) | Full recipe, VAE retrained under the same protocol | 0.935 | — | |
| Representation | Latent diffusion → voxel-space diffusion (no VAE) | 0.917 | -0.023 | *** |
|  | SDT → binary mask in the VAE (vs VAE ref.) | 0.937 | +0.002 | n.s. |
|  | VAE trained without the Dice term (vs VAE ref.) | 0.940 | +0.005 | n.s. |
| Conditioning | Without the bone mask | 0.929 | -0.011 | ** |
|  | Without the distance-to-bone map | 0.926 | -0.014 | *** |
|  | Without lung-channel dropout (CFG training) | 0.939 | -0.001 | n.s. |
| Corruption | Our corruptions → random blobs | 0.923 | -0.017 | ** |
|  | Without medial (hilar) corruption | 0.938 | -0.002 | n.s. |
|  | Without leak corruption | 0.937 | -0.003 | * |

## B. What each inference step contributes (final trained model, steps added one at a time)

| Step | Dice | Δ Dice vs previous step | p | HD95 (mm) | Cons. recall |
|---|---|---|---|---|---|
| Plain sampling (1 sample, no guidance, nearest upsampling) | 0.964 | — | | 9.1 | 0.976 |
| + guidance w = 1.5 | 0.967 | +0.003 | ** | 8.2 | 0.963 |
| + average of 5 samples | 0.976 | +0.009 | *** | 6.2 | 0.969 |
| + smooth SDT upsampling (= LungMend, final) | 0.986 | +0.010 | *** | 5.5 | 0.981 |

p: Wilcoxon signed-rank on the 20 per-patient mean differences (patient = unit); * < 0.05, ** < 0.01, *** < 0.001, n.s. = not significant.
