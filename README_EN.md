<p align="right">
  <a href="./README.md" title="简体中文"><img src="https://img.shields.io/badge/语言-简体中文-blue?logo=googletranslate" alt="简体中文"></a>
</p>

# KohakuMossTrainer · NaRA / SORA / EVA / BlockSkip fork

[KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)

| Added | Summary |
| --- | --- |
| **NaRA** | Noise-aware Rank Adaptation: the low-rank delta varies with the denoising noise level |
| **SORA** | Skewed Orthogonal Rotation Adaptation: diagonal calibration + near-orthogonal rotation |
| **EVA** | Explained Variance Adaptation: initialise the input-side factors from activation SVDs |
| **BlockSkip** | DiT-BlockSkip: dynamic patch sampling + block skipping with precomputed residuals — **near-catastrophic quality loss on MOSS-Image** |
| **SVIR+EK** | static quality inspection at every checkpoint write: SVIR spectral intrusion ratio + EK excess kurtosis, no inference |
| **Loraspectra(optional)** | rank audit at step N through the optional loraspectra CLI: stable rank / energy rank / MP rank and DIM utilisation |


## What's added

### NaRA

- **The `NARA` switch**: `NARA = True` works on lora / locon / loha / lokr, like `DORA`
- lora / locon are equivalent to `ALGO = "nara"`

  LoHa's two rank channels read the same core

  LoKr's core lives in w2's rank contraction
- **NaRA × DoRA**: the normalisation happens after `BA·C(sigma)`

  LoRA / LoKr use the Gram expansion `||W + dW||² = ||W||² + 2<W, dW> + ||dW||²`
- **Training / sampling**: each step installs `(sigma, seq)` automatically;
  the sampler turns on dynamic evaluation when it sees `needs_sigma`
- **Export**: the `lycoris` format keeps NaRA intact

  `comfy` / `merged` freeze the core at `export_sigma` (default 0.5)

  *`lycoris` export is recommended*

### SORA

Can be enabled separately or together:

- **`SORA_PRE_DIAG`** (Pre-Diag): a learned per-output-channel scale of the base
  output; the cheap replacement for DoRA. **Mutually exclusive with `DORA`**
- **`SORA_POST_ORTHO`** (Post-Ortho): a near-orthogonal rotation applied to the whole
  output, built from the first-order Taylor expansion of a low-rank skew-symmetric
  matrix, and therefore very cheap in parameters. Stacks with DoRA.
- **Export precision**: The deviation of `sora_diag` is usually only on the order of ~1e-3, and 83% of the values degrade with bf16. To ensure Pre-Diag quality, please export as **fp32**.
`Pre-Diag + Post-Ortho = SORA`

```
W' = R (diag(d) W_pre + s · BA)
```

### EVA
```
EVA = {
    "batches": 32,       # batch of one epoch
    "min_batches": 4,
    "sim_thresh": 0.99,
    "precision": "fp32",  # 8x time cost on fp64
}
```
Before training starts, a principal-component analysis of the **input activations** on
the real training data initialises the adapter's input-side factors to roughly the right
directions. It raises the starting point of convergence.

- Layers that share an input (q / k / v) are identified by an activation fingerprint and
  calibrated only once.
- `ALGO` must support EVA (LoRA / LoCon, LoKr and their NaRA variants);
  `DROPOUT['module']` must be 0.

### DiT-BlockSkip


- **Dynamic patch sampling** (`RESOLUTION_MODE = "patch"`): every sample trains at a
  fixed low resolution `smin × smin`; the crop side grows linearly with the sample's
  sigma from `smin` to `smax` . The sigma is drawn on the host planner, seeded per occurrence, and carried
  with the batch.
- **Block skipping + residual precomputing** (`BLOCKSKIP`): skip the first n + last m
  joint blocks and train LoRA on the kept middle only. The frozen front is replayed
  from an exact cached absolute feature; the skipped tail from a cached residual
  `Δ = f_L − f_{L−m}` added back. The precompute and train stages share
  one data / sigma / noise stream, with a per-micro-batch sigma equality check.

> ⚠️ **Quality warning: BlockSkip is near-catastrophic on MOSS-Image; keep it off for quality-first training.** 
> the tail residual `‖f_L − f_{L−m}‖` is **1.2–1.4x the head stream itself** 
> the drifting middle stream, corrupting both the training objective and inference
> behaviour. A 1200-step 50% symmetric-skip run generated images with many **missing
> limbs and eyes** . Head-only skipping is mathematically exact but still forfeits 31% of blocks' adapter capacity
>
> **Recommendation: disable `BLOCKSKIP` for quality runs** if you skip at all, **head only `(n, 0)`** 

### SVIR+EK (`QUALITY_CHECK`)

At every checkpoint write check the `Wdown`

- **SVIR (singular value intrusion ratio)**: `#{σ > τ·median(σ)} / effective rank`;
  energy concentrating on a few directions is the moderate/severe-overfitting signal.
- **EK (excess kurtosis)**: negative under normal weight-decayed training; a heavy
  spiky tail turns it positive, the mild-overfitting signal.

Only algorithms with `lora_down` factors are covered.

### Rank audit (`RANK_CHECK`)

Optional external-tool integration: at step N (default 10) the current LoRA factors are
written to a temporary safetensors and audited with
[`loraspectra`](https://github.com/SalyyS1/loraspectra):

- Per layer: von Neumann effective rank `R_EFF`, stable rank `R_STABLE`, 99% energy
  rank `K_99%`, Marchenko-Pastur signal rank `K_MP`, and a status (`OPTIMAL` /
  `OVER_PARAMETERIZED` / `RANK_COLLAPSED` / `DEAD_LAYER`).
- Global `guidance`: `energy_use` (mean k99 / `DIM`), `mp_share` (MP rank / `DIM`) and
  the per-status layer counts.


Need the environment variable `$LORASPECTRA` / `PATH`
Only `PARALLEL = "none"` and LoRA-style factors are supported.

## Usage

### Training

NaRA + DoRA:

```python
ALGO = "lora"            # "lora" | "locon" | "loha" | "lokr"
NARA = True
DORA = True

ADAPTER_MODE = "bypass"  # NaRA supports bypass only
SAVE_FORMAT = "both"
```

SORA (Pre-Diag + Post-Ortho):

```python
ALGO = "lora"
DORA = False
SORA_PRE_DIAG = True
SORA_POST_ORTHO = {"rank": 1, "initial_s_p": 1.0}   # or SORA_POST_ORTHO = True
ADAPTER_MODE = "bypass"
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
```

EVA:

```python
EVA = {"batches": 32, "min_batches": 4, "sim_thresh": 0.99, "precision": "fp32"}  # or {} for the defaults
```

`RESUME` / `INIT_FROM` skip the calibration.

BlockSkip:

```python
RESOLUTION_MODE = "patch"                 # dynamic patch sampling 
PLANNER_KWARGS = {"smin": 512, "smax": 1024}

# stage 1: precompute 
BLOCKSKIP = {"mode": "precompute", "skip": 0.5, "cache": "path/to/features"}
SAVE_FORMAT = "none"

# stage 2: train 
BLOCKSKIP = {"mode": "train", "skip": 0.5, "cache": "path/to/features"}
```

Both stages must share `SEED` / `DATASETS` / `STEPS` / `GRAD_ACC` / `TIMESTEP` and the
resolution settings.

Static quality check (`QUALITY_CHECK`):

```python
QUALITY_CHECK = True   # or {"tau": 3.0, "tau_low_rank": 3.5, "ema": False, "top": 5}
```

Rank audit (`RANK_CHECK`, optional external tool):

```python
RANK_CHECK = {"step": 10}   # or True; {"step","layers","keep","timeout"}
```

## Parameters

### NaRA hypernetwork (`ALGO_KWARGS`)

| key | default | meaning |
| --- | --- | --- |
| `nara_hidden1`, `nara_hidden2` | 256, 512 | mapper hidden widths |
| `nara_embed_dim` | 64 | embedding width |
| `nara_embed` | `fourier` | `fourier` / `mlp` / `raw` |
| `nara_fourier_scale` | 16.0 | frequency scale of the fixed Fourier embedding |
| `nara_c_scale` | 1.0 | core residual scale eta |
| `export_sigma` | 0.5 | the noise level a fixed-sigma export uses |

### SORA parameters

| key | default | meaning |
| --- | --- | --- |
| `SORA_POST_ORTHO` | `0` | `r_p`, the half-rank of the rotation (the skew term has rank 2·r_p) |
| `SORA_POST_ORTHO["initial_s_p"]` | `1.0` | upper bound of the skew term; `s_p = min(1, s_p0 / (‖C‖+‖D‖))` |
| `SORA_PRE_DIAG` | `False` | per-output-channel scale of the base |

Sensitivity of `r_p`: `r_p=1` trains fastest and is the least accurate, but the gap is
small; larger `r_p` has diminishing returns while inference cost rises monotonically.

### EVA parameters

| key | default | meaning |
| --- | --- | --- |
| `batches` | 32 | upper bound on calibration forward batches |
| `min_batches` | 4 | lower bound on batches |
| `sim_thresh` | 0.99 | per-component cosine between consecutive incremental SVDs that counts as converged |
| `precision` | `"fp64"` | `"fp64"` / `"fp32"` |

### BlockSkip parameters

| key | default | meaning |
| --- | --- | --- |
| `BLOCKSKIP["mode"]` | `"train"` | `"precompute"` (store features) / `"train"` (block-skip training) |
| `BLOCKSKIP["skip"]` | `0.5` | skip ratio, or an explicit `(n, m)`; a ratio splits n ≈ 0.6k front / m ≈ 0.4k back |
| `BLOCKSKIP["cache"]` | required | feature directory (one bf16 file per step) |
| `PLANNER_KWARGS["smin"]` | 512 | patch training resolution (`RESOLUTION_MODE="patch"`) |
| `PLANNER_KWARGS["smax"]` | 1024 | crop side at sigma=1 |

### Static quality check parameters (`QUALITY_CHECK`)

| key | default | meaning |
| --- | --- | --- |
| `tau` | 3.0 | SVIR threshold factor (`σ > τ·median(σ)` counts as an intrusion) |
| `tau_low_rank` | 3.5 | the factor for layers of rank `<= 4` |
| `ema` | `False` | also inspect the EMA weights |
| `top` | 5 | worst-EK layers recorded per row |

### Rank audit parameters (`RANK_CHECK`)

| key | default | meaning |
| --- | --- | --- |
| `step` | 10 | the one-shot audit step |
| `layers` | `True` | also parse the per-layer table |
| `keep` | `False` | keep the temporary adapter file |
| `timeout` | 300.0 | per-CLI-call timeout in seconds |

## Inference

### ComfyUI

[Inference nodes](https://github.com/RBkreb/comfyui-moss-nara)

**A `lycoris` weight that contains SORA layers must be loaded through the plugin**

## Limitations

### Storage: exporting SORA layers in `comfy` format

A SORA layer exported to the `comfy` format **writes the exact dense diff**
(`convert.comfy`): the rotation and the base scale act on `W` itself, so the size is
O(d²). At MOSS-Image's real shapes (42 layers, `dim=3072`, `ffn=8192`):

| preset | targets | `comfy` export size (bf16) |
| --- | --- | --- |
| `attn` | 168 | ≈ 2.95 GiB / checkpoint |
| `attn-mlp` | 252 | ≈ 8.86 GiB / checkpoint |

- With SORA, prefer `SAVE_FORMAT = "hf"`
- `SAVE_MERGED` is unavailable.

### Compute: EVA's `precision` defaults to fp64

`torch-incremental-pca` and `torch.svd_lowrank` run in fp64, so every QR and GEMM runs
in fp64.
- On consumer cards, `fp32` is recommended

| `precision` | behaviour | cost |
| --- | --- | --- |
| `"fp64"` (default) | the package as shipped | about 10x the time of fp32 |
| `"fp32"` | `svd_lowrank` on an fp32 copy | faster |

EVA's cost is `MAX_TOKENS × batches × layers`, **independent of dataset size and step
count**.

fp32 precision loss: with a fixed seed the largest principal-angle difference between
fp32 and fp64 is 0.10.

### Compute: Pre-Diag's cost depends on what the delta costs

Pre-Diag has to read that delta on every forward to apply the per-channel calibration,
**so its value depends on how expensive the delta itself is.**

- LoRA's delta is two `F.linear` calls: cheap.
- NaRA's delta is a function of the hypernetwork: every layer, every forward has to run
  the mapper once, plus two `F.linear` calls and a per-token `bmm`. At the same attn
  shape, the hyper alone is about 80% of one delta body.

So **NaRA + Pre-Diag is necessarily slower per step than LoRA + Pre-Diag**: the former
is a strict superset of the latter, and the extra work makes nothing else cheaper.

RTX 3090, attn 3072×3072 / 12288 tokens, fwd+bwd, bf16, single-layer synthetic benchmark. 

| combination | vs plain LoRA |
| --- | --- |
| LoRA + Pre-Diag | 1.75× |
| LoRA + Post-Ortho | 2.28× |
| LoRA + SORA (both on) | 3.05× |

The rotation (Post-Ortho) is the main cost: it reads and writes the whole
`[tokens, out]` activation twice.

### Quality and alignment: BlockSkip

- **Lossy, and measured near-catastrophic on MOSS-Image**: see the quality warning
  above (tail residual 1.2–1.4x the input stream, Eq. 3 invalid; a 1200-step 50% skip
  run produced missing limbs / eyes). Disable `BLOCKSKIP` for quality-first runs.
- The precompute and train stages must share `SEED` / `DATASETS` / caption / dropout /
  `STEPS` / `GRAD_ACC` / `TIMESTEP` / resolution settings exactly; every micro-batch
  checks sigma equality and fails loudly on drift.
- Refused: `PARALLEL != "none"`, `ALGO = "full"`, `STREAM_LOSS` / `CONCENTRATION_LOSS`.
- **EVA coexists with BlockSkip**: both stages restrict `PRESET` to the kept middle
  blocks, so EVA calibrates only those layers; keep the EVA knobs identical across the
  two runs (the calibration then consumes the same data / RNG prefix, and the
  per-micro-batch sigma check guards the rest). Without BlockSkip, EVA is unchanged.
- A toml `PRESET` is not auto-restricted: limit its block patterns to the kept range
  yourself.
- Feature disk cost: `2 × N_joint × 3072 × 2 B` per step (bf16); ~17 MB per 512²
  1-sample micro-batch.
### Quality check and rank audit

- `QUALITY_CHECK` inspects the `Wdown` factors only: NaRA's `C(sigma)`, the DoRA
- `τ` and EK bands were calibrated on SDXL / FLUX / Kolors community
  LoRAs; this implementation's `watch` rules (SVIR layer mean > 5%, EK median > 0)
  are conservative proxies, without linear fusion or three-grade
  classification. On MOSS treat them as directional.
- `RANK_CHECK` depends on the optional CLI (verified against loraspectra v1.8.0) and
  supports `PARALLEL = "none"` with LoRA-style factors only; it audits the early
  delta's utilisation at step N, not the gradient-spectrum rank rule, and the earlier
  the step the noisier the delta.

### Other

- `ADAPTER_MODE = "merge"` is refused (NaRA's sigma-dependent delta cannot fold into one weight)
- With NaRA on, `rank dropout` / `use_tucker` / `use_scalar` / `rs_lora` are refused
- The shared hypernetwork cannot cross FSDP shards or pipeline stages: fsdp / pipeline /
  threaded layouts are refused
- The LoKr variant needs w2 to have rank for the core: a full-rank w2 layout is refused
  at construction (lower `DIM` or raise `ALGO_KWARGS["factor"]`)
- The LoHa variant rebuilds a dense delta per sample in its bypass; pair it with a block
  checkpointing profile
- Pre-Diag requires `DROPOUT['input'] = 0` and `DROPOUT['rank'] = 0`
- SORA's `lycoris` export cannot carry a **per-row** fold scale (`R·S ≠ S·R`, and the
  factor set has no term for it): a uniform fold exports exactly (`R(cW) = cR(W)`), a
  per-row fold is refused per layer. Under the default `PRESET` the per-row layer count
  is 0.
- EVA's convergence log (`N/M converged`) reads low on degenerate subspaces because the
  per-component cosine flips between ±1; this is normal.

## How it works

## NaRA

NaRA makes the adapter's low-rank delta vary with the denoising noise level, placing a
dynamic core between two static LoRA factors:

```
h = W0 x + s · B C(sigma) A x
C(sigma) = I_r + c_scale · F(e_sigma)
```

- `F` is a small MLP and `e` is a Gaussian Fourier embedding of the noise level;
  **every adapted layer shares one hypernetwork** and stores only A (`lora_down`) / B
  (`lora_up`).
- The mapper's last layer starts at zero, so initially `C = I` and NaRA reduces to plain
  LoRA.
- Conditioning is **per sample** of the packed batch (sigma `[B]` -> cores `[B, r, r]`,
  fp32), expanded along the layer's token axis, consistent with T-LoRA's masking scheme.

## SORA

SORA adds an orthogonal, learnable conditioning to the whole layer's weight.

- **Pre-Diag**: `W = W_pre D + s·BA`. It turns D from "a measured quantity whose column
  norms must be recomputed every forward" into "a free parameter".
- **Post-Ortho**: `W' = (W_pre D + s·BA) P`, with `P ≈ I + s_p(D_p C_pᵀ − C_p D_pᵀ)`.
  The orthogonal transform preserves norms. The multiplication is reordered as
  `(W D_p) C_pᵀ − (W C_p) D_pᵀ`, dropping the complexity from `O(mn²)` to `O(mn r_p)`.

## EVA

EVA changes the starting values training begins from.

- One `SVDHook` per layer runs an incremental PCA on the layer input's token rows.
- LoRA / LoCon / NaRA: `lora_down = components[:DIM]`, `lora_up = 0`.
- LoKr / NaRA-LoKr: the top eigenvectors of the block Gram matrix (the dominant
  directions under the Kronecker constraint); `w2_a = 0` and `w1` keeps its random
  initialisation.

## Examples

[DoRA+NaRA example](https://huggingface.co/RBXLSP/TextureLux-DoRAN)

## Citation

### SORA — Skewed Orthogonal Rotation Adaptation

```bibtex
@inproceedings{chang2026sora,
  title     = {Calibrating and Rotating: A Unified Framework for Weight Conditioning in PEFT},
  author    = {Chang, Da and Xue, Peng and Li, Yu and Liu, Yongxiang and Xu, Pengxiang and Zhang, Shixun},
  booktitle = {The Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)},
  year      = {2026}
}
```

- Paper: <https://arxiv.org/abs/2511.00051>
- Repository: <https://github.com/MaeChd/SORA>

### EVA — Explained Variance Adaptation

```bibtex
@article{paischer2024eva,
  title   = {One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation},
  author  = {Paischer, Fabian and Hauzenberger, Lukas and Schmied, Thomas and Alkin, Benedikt and Deisenroth, Marc Peter and Hochreiter, Sepp},
  journal = {arXiv preprint arXiv:2410.07170},
  year    = {2024}
}
```

- Paper: <https://arxiv.org/abs/2410.07170>
- Repository: <https://github.com/ml-jku/EVA>

### NaRA — Noise-Aware Rank Adaptation

```bibtex
@inproceedings{nara2026,
  title={Na{RA}: Noise-Aware Lo{RA} for Parameter-Efficient Fine-Tuning of Diffusion {LLM}s},
  author={Shuaidi Wang and Zhan Zhuang and Ruping Huang and Yu Zhang},
  year={2026},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=s8UGIRF38a}
}
```

- Paper: https://arxiv.org/abs/2605.29716
- Repository: https://github.com/generaldi/NaRA

### DiT-BlockSkip — Memory-Efficient Fine-Tuning Diffusion Transformers

```bibtex
@article{park2026ditblockskip,
  title   = {Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping},
  author  = {Park, Sunghyun and Kim, Jeongho and Park, Hyoungwoo and Das, Debasmit and Yun, Sungrack and Hayat, Munawar and Choo, Jaegul and Porikli, Fatih and Choi, Seokeon},
  journal = {arXiv preprint arXiv:2603.20755},
  year    = {2026}
}
```

- Paper: <https://arxiv.org/abs/2603.20755>

### SVIR + EK

```bibtex
@inproceedings{liu2026static,
  title     = {Dual-Dimensional Static Quality Inspection Method for Diffusion Model {LoRA} Based on Singular Value Intrusion Ratio and Excess Kurtosis},
  author    = {Liu, Xiaohu and Sun, Ronghui},
  booktitle = {2026 8th International Conference on Artificial Intelligence Technologies and Applications (ICAITA)},
  pages     = {94--101},
  year      = {2026},
  doi       = {10.1109/ICAITA71179.2026.11736932}
}
```

- DOI: <https://doi.org/10.1109/ICAITA71179.2026.11736932>

### loraspectra — LoRA spectral audit CLI

- Install: `cargo install loraspectra`
- Repository: <https://github.com/SalyyS1/loraspectra>

## License

[KohakuMoss License 1.0](LICENSE)
