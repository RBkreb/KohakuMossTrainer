# Static quality inspection at save

`QUALITY_CHECK` runs a dual-dimensional static inspection of the adapter's `Wdown`
factors at every checkpoint write: the Singular Value Intrusion Ratio (SVIR) and the
excess kurtosis (EK). Both come from the LoRA static-inspection method of Liu & Sun,
ICAITA 2026. No inference, no image generation, CPU only.

## What runs

The hook is `CheckpointManager.after_save`, so it sees the exact state dicts the
checkpoint just wrote (`trainable`, and `ema` when asked). Every rank calls the hook;
rank 0 computes and appends one row per source to `<output>/quality.jsonl`, and logs a
one-line summary. The run's final export (under `<output>/export/`) is inspected too
when `FINAL_CHECKPOINT` is off, because no checkpoint write covers that step then. With
`SAVE_FORMAT = "none"` and no checkpoint cadence nothing is written, so nothing is
inspected.

On a 252-layer rank-32 adapter the whole scan takes ~0.3 s on a desktop CPU; it runs
between training steps, never on the GPU.

## The two indices

For one factor `W` (`lora_down.weight`, float32 on CPU):

- `EK = mean(((x - mu) / sigma)^4) - 3` over the flattened entries. L2-decayed normal
  training keeps it negative (uniform-like); a heavy-tailed "spiky" distribution turns
  it positive.
- `SVIR = #{sigma_i > tau * median(sigma)} / r_eff`, with the effective (stable) rank
  `r_eff = ||W||_F^2 / sigma_1^2`. `tau = 3.0` globally and `3.5` for a factor of rank
  `min(shape) <= 4`, the paper's calibrated thresholds. Spectral energy concentrating
  on a few directions raises it.

Per-layer values aggregate the paper's way: SVIR as the mean over layers, EK as
quantiles over layers plus the share of positive layers (the paper's 5th-percentile
anomaly rule reads the worst layers, reported as `ek_p95` / `top`).

## Watch flags and row schema

`watch` lists the dimensions that fire:

- `svir` when the mean SVIR is above 0.05 (the paper's healthy bound: intrusions stay
  under 5% of the effective rank);
- `ek` when the median EK is positive (the paper's mild-overfitting signature: normal
  L2-decayed training stays negative).

One JSON line per source:

| field | meaning |
| --- | --- |
| `step`, `kind` | step; `checkpoint` or `final-export` |
| `source` | `trainable`, or `ema` with `"ema": true` |
| `layers` | inspected `Wdown` factors |
| `svir_mean`, `svir_max` | paper's spectral index |
| `layers_with_intrusions`, `intrusions_max` | layers with any `sigma > tau * median`, and the largest count |
| `ek_median`, `ek_mean`, `ek_p05`, `ek_p95`, `ek_max`, `ek_frac_pos` | EK distribution over layers |
| `reff_median` | median effective rank |
| `watch` | fired dimensions |
| `top` | the `top` layers by EK: name (suffix stripped), `ek`, `svir`, `intrusions`, `r_eff` |

Options: `QUALITY_CHECK = True` uses the defaults below; a dict overrides single keys.

| key | default | meaning |
| --- | --- | --- |
| `tau` | 3.0 | SVIR threshold factor |
| `tau_low_rank` | 3.5 | the factor for `rank <= 4` |
| `ema` | False | also inspect the EMA weights |
| `top` | 5 | worst layers by EK in each row (0 = none) |

## Scope

Inspection needs LoRA-style `lora_down` factors (`lora` / `locon` with LoRA factors,
and their NaRA / DoRA / SORA combinations). Algorithms without a 2-D down factor
(LoHa, LoKr, OFT, ...) are logged and skipped. The paper calibrated its thresholds on
SDXL / FLUX / Kolors community LoRAs; on MOSS the absolute bands are directional only.
The primary signal is the trend across a run's own checkpoints and its comparison with
reference adapters, not a single absolute grade.
