# EVA: data-driven adapter initialisation

EVA (Explained Variance Adaptation, arXiv:2410.07170, NeurIPS 2025) initialises the
adapter's input-side factors from the principal directions of the target layers'
**input activations** on the training data, instead of random directions. The output
factors stay zero, so training starts at the base model exactly as with random init;
only the first gradient's alignment changes. No loss, optimizer or per-step change.

## Knob

```python
EVA = {"batches": 32, "min_batches": 4, "sim_thresh": 0.99}  # {} = these defaults
```

`EVA` present (and not False) runs the calibration on a fresh run; `RESUME` and
`INIT_FROM` skip it (the state already carries trained factors). `batches` caps the
calibration forwards, `min_batches` floors them, `sim_thresh` is the per-component
cosine similarity between consecutive incremental SVDs that counts as converged.
ALGO must be eva-capable (`Kind.supports_eva`: LoRA / LoCon, LoKr, and their NaRA
variants); module dropout must be 0 (every layer must produce activations).

## Calibration

`adapters/eva.calibrate` runs in `training/entry.worker` after the input pipeline is
built and **before the EMA copies its masters**, through the real training step:
`step.prepare` samples sigma from the run's own timestep sampler, so the activations
cover the noise levels the adapter will train on (the diffusion-specific part — the
paper's LLM calibration has no sigma axis). Forwards are no-grad, under the run's
autocast, on the rank's own data shard.

- One `SVDHook` (forward pre-hook) per target linear runs `torch-incremental-pca`'s
  IncrementalPCA on the token rows of the layer input, fp32, low-rank mode.
- Layers with equal inputs (q / k / v share one activation) are detected by a cheap
  subsample fingerprint on the first batch and calibrated once; the aliases copy the
  same components.
- A layer stops fitting once all its components converge; the loop ends when every
  layer converged (after `min_batches`) or `batches` is spent. Unconverged layers
  still take their current components; a layer that never fitted keeps random init.
- Under a process group, rank 0's initialised factors are broadcast to every rank
  (each calibrated on its own shard).

The calibration consumes the sigma / noise generator streams and the loader's first
batches; training continues after them.

## Factor initialisation

- LoRA / LoCon / NaRA: `lora_down = components[:DIM]`, `lora_up = 0`.
- LoKr / NaRA-LoKr: the input projection is kron(w1, w2_b), so the components
  (m1 r2 rows, reshaped over the n1 input blocks) cannot land in w2_b directly.
  `w2_b` takes the top r2 eigenvectors of the block Gram
  `sum over rows and blocks of v v^T` — the dominant n2-directions of the activation
  subspace under the Kronecker constraint (a documented heuristic, not the paper's
  plain-LoRA theorem); `w2_a = 0`, w1 keeps its random init. A full-matrix w2 layout
  has no rank for this and keeps random init.

Component count per layer: DIM for LoRA shapes, m1 r2 for LoKr (capped at 512 so one
layer's component matrix stays small next to the model). A LoKr layout whose m1 r2
exceeds the cap cannot be initialised — that layer keeps its random init, and the
calibration logs a warning naming it and the fix (`lower ALGO_KWARGS['factor'] or
DIM`), so a run never assumes an init it did not get.

## Scope

Uniform rank: the paper's cross-layer rank redistribution (explained-variance budget
reallocation) is **not** in this version — it needs per-layer dims at module build,
before calibration. Exports are unchanged: EVA only sets starting values, every
format and the inference stack see ordinary factors.

Dependency: `torch-incremental-pca` (the paper's own incremental SVD implementation).