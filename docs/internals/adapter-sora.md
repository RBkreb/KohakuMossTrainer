# SORA: output conditioning on any additive adapter

SORA (Skewed Orthogonal Rotation Adaptation, arXiv:2511.00051) conditions the layer's
output around the adapter delta. Two orthogonal switches, both independent of ALGO
(LoRA / LoCon, LoHa, LoKr and their NaRA variants; `Kind.supports_sora`):

- `SORA_POST_ORTHO` — Post-Ortho: a near-orthogonal rotation of the whole output
  (base + delta, DoRA-normalised included). Stacks with DoRA and NaRA.
- `SORA_PRE_DIAG` — Pre-Diag: a learned per-output-channel scale of the base output,
  the paper's cheap DoRA replacement. **Mutually exclusive with `DORA`** (one
  conditioning, not two); `AdapterConfig.validate` refuses the pair.

## Knobs

```python
SORA_POST_ORTHO = {"rank": 1, "initial_s_p": 1.0}  # or a bare rank, or True (= rank 1)
SORA_PRE_DIAG = True
```

`rank` is r_p, the rotation's half-rank (1-2 in the paper; the skew term has rank
2 r_p). `initial_s_p` bounds the skew: s_p = min(1, s_p0 / (||C||_F + ||D||_F + eps)),
so the rotation stays close to identity however the factors grow.

## Parameters and forward

Per adapted layer, on the LyCORIS module (so checkpoints, parameter groups and the
raw-state reload carry them automatically):

- Post-Ortho: `sora_C`, `sora_D` (out x r_p; C Kaiming fan-out, D = C so the initial
  skew is zero and P starts at I), buffer `sora_s_p0`.
- Pre-Diag: `sora_diag` (out, ones).

The bound forward wraps the kind's own forward (`bind/select.py`, inside the
module-dropout wrap: a dropped module skips its conditioning too):

- Pre-Diag at strength m: `y = d_m fn(x) + (1 - d_m) delta_m(x)` with
  `d_m = 1 + m (d - 1)` and `delta_m = bypass_forward_diff(x, m)` — exactly
  `base + m ((d - 1) base + delta)` at every m, one extra low-rank delta per forward.
  The double delta read is why input / rank dropout must be 0 with Pre-Diag.
- Post-Ortho: `y = fn(x) + m (R - I) fn(x)`, the skew term computed as
  `s_p ((y C) D^T - (y D) C^T)` in fp32 — never the dense out x out rotation. Exact at
  m = 0 and 1; between, a strength lerp whose m^2 cross term is second order in the
  bounded skew.

Both work on stored (quantized / streamed) bases: they read the layer's output and
the module's bypass delta, never a dense weight. All SORA math runs fp32.

## Export and merge

`kinds/common.delta_builder` folds SORA into dW in weight space, so merged export,
`merge_into`, dense-diff paths and the inference stack's `earlier` deltas all see the
conditioned delta:

- Post-Ortho: `dW' = R (W + dW) - W` (and the bias through R).
- Pre-Diag: `dW' = diag(d - 1) W + dW`, `db' = (d - 1) b + db`.
- Both: Pre-Diag first, then the rotation — `W' = R (diag(d) W + dW)`, the forward's
  order.

Formats:

- `comfy`: a SORA layer exports its **exact dense diff** (`convert.comfy`): the
  rotation / base scale touches W itself, which no low-rank factor set expresses
  (DoRA's row norm included). NaRA's delta freezes at `export_sigma` as usual.
- `lycoris`: the native factor set plus `sora_C / sora_D / sora_diag / sora_s_p0`
  beside it (LyCORIS's custom state_dict only emits its algorithm's weight list;
  `convert.native` adds these). Folded models: a **uniform** fold scale (the fp16-fix
  uniform `via="gate"` fold) is a multiple of the identity on the output axis, so it
  commutes with the rotation — `R (c W) = c R W` — and the factor set carries it
  exactly (the dW rows are divided by it as usual; verified to 1e-7 relative against
  the folded forward). A **per-output-row** fold scale does not commute with the
  rotation and has no factor term, so `tensors()` refuses it on exactly the affected
  layers: export `comfy` (exact dense diff) or `merged` there.
- `merged` / `diffusers`: through the conditioned delta builder, exact.

Inference (`infer/inference.py`): the SORA keys are popped before LyCORIS module
matching, re-attached frozen, and the chained forward gets the same wrappers at the
load multiplier (`sora.wrap_infer`). A comfy dense diff loads as any diff (needs a
dense base in the trainer's own stack; ComfyUI itself adds it to any base).

## Cost

Post-Ortho: two N x out x r_p matmuls per forward (fp32), r_p <= 2 is far below the
per-sample fp32 DoRA norm it may stack with. Pre-Diag: one extra low-rank delta
forward. Parameters: 2 out r_p + out per layer (~0.1% at r_p = 1). No optimizer or
loss changes; the factors join the normal parameter groups (no weight decay, no
LoRA+ ratio, Muon-ineligible at r_p = 1).

Evidence: the paper's gains are LLM / NLU only (+0.3-0.8 over DoRA); on the DiT this
is a research switch, verified here for math and round-trip, not for quality.