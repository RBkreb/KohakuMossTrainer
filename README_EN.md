<p align="right">
  <a href="./README.md" title="简体中文"><img src="https://img.shields.io/badge/语言-简体中文-blue?logo=googletranslate" alt="简体中文"></a>
</p>

# KohakuMossTrainer · NaRA / SORA / EVA fork

[KohakuBlueleaf/KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)

| Added | Summary |
| --- | --- |
| **NaRA** | Noise-aware Rank Adaptation: the low-rank delta varies with the denoising noise level |
| **SORA** | Skewed Orthogonal Rotation Adaptation: diagonal calibration + near-orthogonal rotation |
| **EVA** | Explained Variance Adaptation: initialise the input-side factors from activation SVDs |


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

## License

[KohakuMoss License 1.0](LICENSE)
