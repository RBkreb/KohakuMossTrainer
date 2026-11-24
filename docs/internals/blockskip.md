# DiT-BlockSkip: dynamic patch sampling and block skipping

A reproduction of [DiT-BlockSkip (arXiv 2603.20755)](https://arxiv.org/abs/2603.20755)
for adapter fine-tuning: train at a fixed low resolution with timestep-aware crops,
and update only the middle transformer blocks while the skipped front / tail blocks
are replayed from precomputed features. The goal is training memory and wall-clock,
not quality — **the method is a lossy approximation; see the quality warning below.**

Two independent components, enabled together or separately:

| component | knob | effect |
| --- | --- | --- |
| dynamic patch sampling | `RESOLUTION_MODE = "patch"` | every sample trains at `smin × smin`; the crop side grows with its sigma from `smin` to `smax` |
| block skipping | `BLOCKSKIP = {...}` | skip the first `n` + last `m` joint blocks; LoRA on the kept middle only |

## Dynamic patch sampling

High-noise draws should see global structure, low-noise draws fine detail. The patch
planner (`kohakumoss_data/resolution/patch.py`) draws each occurrence's sigma on the
host — seeded by `(seed, epoch, position, row_id)`, iid from the run's moss timestep
pmf, distributionally identical to the device-side sampler — then crops a box of side

```
side = round16(smin + sigma · (smax − smin))      # round to the latent factor
```

at a seeded random position and resizes it to `smin × smin`. The sigma rides along in
the `HostBatch` / `DeviceBatch`, and `TrainStep.noise` uses it instead of sampling its
own. Because every crop resizes to the same target, token counts and batching are
unchanged; the latent-cache key already covers the crop box, so cached latents stay
correct. `smin` / `smax` come from `PLANNER_KWARGS`; the sigma grid / cdf are injected
by the training entry (`data_knobs`) from `TIMESTEP_KWARGS` — a single-stage moss
schedule is required.

## Block skipping with residual features

Skipping blocks naively breaks the forward path (the paper's ablation: DINO 0.73 →
0.43). BlockSkip replays the skipped segments from features captured on the frozen
base model:

- **Head** (blocks `[0, n)`): everything before the first kept block is frozen
  (embedders, text refiner, modulation), so the joint stream entering block `n` is
  *identical* in every training step. The precompute stores that **absolute feature**
  and the training forward starts from it — exact, no approximation.
- **Tail** (blocks `[L−m, L)`): the stored residual `Δ = f_L − f_{L−m}` is added after
  the last kept block: `f'_{L} = f'_{L−m} + Δ` (paper Eq. 3). This is the lossy part:
  as LoRA updates drift the kept-block outputs, the frozen tail residual no longer
  matches exactly. The fold's per-block residual rescale is absorbed into Δ (the
  capture sees the real stream, rescales included).

The block selection follows the paper's finding that mid-level blocks carry the
subject semantics: skip a front-heavy split of the ends, `n ≈ 0.6k`, `m ≈ 0.4k` of
`k = round(depth · ratio)` skipped blocks. The paper selects `(n, m)` per dataset with
a cross-attention-masking sweep over trained LoRAs; this reproduction uses the
proportional split (override with an explicit `"skip": (n, m)`).

### The two stages and their alignment contract

```python
# stage 1 — precompute: full no-grad forwards, one feature file per optimiser step
RESOLUTION_MODE = "patch"
PLANNER_KWARGS = {"smin": 512, "smax": 1024}
BLOCKSKIP = {"mode": "precompute", "skip": 0.5, "cache": "D:/features/run1"}
SAVE_FORMAT = "none"

# stage 2 — train: identical everything else
BLOCKSKIP = {"mode": "train", "skip": 0.5, "cache": "D:/features/run1"}
```

Both runs must share `SEED`, `DATASETS`, caption / dropout settings, `STEPS`,
`GRAD_ACC`, `TIMESTEP` / `TIMESTEP_KWARGS` and the resolution mode: the precompute
replays the training loop's exact batch order, sigmas and noise (the sigma sampler and
the loader are seeded pure functions). Every training micro-batch checks its sigma
against the stored one (`torch.equal`) and **fails loudly** on drift — misaligned
features would otherwise train silently on garbage.

`PRESET` is restricted automatically in train mode: every `layers.\d+` pattern is
rewritten to the kept indices and non-block targets (modulation, embeds, text refiner)
are dropped, so the export is an ordinary LoRA over a subset of blocks — inference
runs the full model unchanged, with no residual features and no extra latency. A toml
`PRESET` is left untouched: restrict its block patterns yourself.

### Storage

One file per step holds, per micro-batch, the head feature and the tail delta:
`2 × N_joint × dim × 2 bytes` (bf16). At 512² with ~330 text tokens that is ~17 MB per
1-sample micro-batch; a 61-step batch-1 × grad-acc-4 run stored 3.7 GB. The reader
prefetches the next step's file on a background thread.

## What it composes with, and what it refuses

- Composes with every memory plan: skipped blocks are never entered, so their ring
  swaps, checkpoint recomputes, activation offload and backward all disappear. The
  gain is largest when the kept-block count approaches `STREAM_RESIDENT` (the ring
  stops thrashing).
- Refused at build: `PARALLEL != "none"`, `ALGO = "full"`, `STREAM_LOSS` /
  `CONCENTRATION_LOSS` (their hooks live on skipped blocks), `EVA`.
- Sampling / validation during a train run use the full model (the skip lives in the
  training forward only).
- The `Throughput` callback's TFLOP/s model assumes all blocks run; under BlockSkip it
  overestimates FLOPs. `step_time`, `tokens_per_s` and the VRAM rows are unaffected.

## Measured gain (this fork's box: RTX 3090 training + RTX 4060 encoders)

MOSS-Image sft-bf16, `balanced` profile (ckpt-ring16), batch 1 × grad_acc 4, 5-image
set, median of 60 steps, bf16:

| config | step time | vs base | peak alloc |
| --- | --- | --- | --- |
| LoRA @1024 (`resize`) | 9.77 s | 1.00× | 7.28 GiB |
| patch 512 only | 7.88 s | 1.24× | 5.96 GiB |
| patch + skip 30% (keep 29) | 5.23 s | 1.87× | 5.52 GiB |
| patch + skip 50% (keep 21) | 2.55 s | 3.82× | 5.24 GiB |

Precompute cost (61 steps): 324 s (skip 30%), 249 s (skip 50%). Including it, the net
wall-clock over 60 steps is 0.92× / 1.46×; the precompute is a one-time cost per
dataset / seed, so at 500 steps the net gain is ≈1.7× / ≈3.2×.

The step-time gain far exceeds the paper's ~25% (SANA, A100) because this box at small
batch is **weight-streaming-bound**, not compute-bound: skipping blocks cuts the ring
traffic directly, and 21 kept blocks nearly fit the 16 ring slots. The VRAM drop is
modest next to the paper's −71% because the ring baseline already keeps the frozen
weights on the host.

## Quality warning

**This fork measured speed / VRAM and mathematical equivalence only — no DINO / CLIP
quality evaluation was run.** The paper's own numbers (FLUX.1-dev, DreamBooth,
subject fidelity; higher is better):

| method | DINO | CLIP-I |
| --- | --- | --- |
| LoRA (upper bound) | 0.7324 | 0.8146 |
| BlockSkip 30% | 0.7194 (−1.8%) | 0.8036 (−1.3%) |
| BlockSkip 40% | 0.7171 (−2.1%) | 0.8034 |
| BlockSkip 50% | 0.6963 (**−4.9%**) | 0.7877 (**−3.3%**) |

Text fidelity (CLIP-T) is essentially unchanged. Without the residual features the
same skipping collapses to DINO ≈ 0.43; skipping the *last* 50% instead of the split
gives 0.48 — the skip positions matter. This reproduction's proportional split is not
the paper's per-dataset selection, so the loss here may be larger. Run your own
quality comparison before relying on BlockSkip for production training.

## Implementation map

| piece | file |
| --- | --- |
| spec / preset restriction / capture / reader / forwards | `kohakumoss/training/blockskip.py` |
| patch planner | `kohakumoss_data/resolution/patch.py` |
| carried sigma | `PlanItem.sigma` → `HostBatch.sigma` → `DeviceBatch.sigma` → `TrainStep.noise` |
| sigma table injection | `training/build/model.py:data_knobs` |
| stage wiring | `training/entry.py:worker` (preset restriction before build, `install_train` after `build_step`, precompute in place of `driver.run`) |

Checks: `tmp/blockskip_math_check.py` (fp64 residual identity, gradient flow, fold
rescale, planner determinism, preset regex) and the dual-GPU smokes
`tmp/blockskip_{pre,train}_smoke.py` in the workspace.