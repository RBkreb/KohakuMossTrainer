# Rank check at step N

`RANK_CHECK` is a one-shot spectral audit of the adapter, taken at a chosen step
(default 10) through the optional `loraspectra` Rust CLI. It answers "how much of the
nominal rank is the adapter already using" without any Python SVD or new training
code; the tool is not a package dependency, and a run without it only logs a warning.

## What runs

At `step` (rank 0, `PARALLEL = "none"` only) the callback writes the trainable
`lora_down` / `lora_up` / `alpha` / `dora_scale` tensors to
`<output>/rank_check/step_XXXXXXXX.safetensors` and runs the CLI twice:

- `loraspectra audit <file> --json` for the stable summary;
- `loraspectra audit <file>` for the per-layer table and its `[WARNING]` lines.

The temporary adapter is deleted after the audit unless `keep` is on. The report goes
to `<output>/rank_check.json`; a one-line summary is logged.

Tool lookup is `$LORASPECTRA` (a path to the executable), else the `loraspectra` entry
on `PATH` — the same lookup on Windows and Linux. Missing tool, a non-plain layout
(rank 0 does not hold the whole adapter), `STEPS < step`, a resume past the step, or an
ALGO without LoRA factors each log one warning (or info) and skip.

## Metrics

Per layer the CLI reports the nominal rank, the von Neumann effective rank
`exp(entropy(sigma))`, the stable rank `||W||_F^2 / sigma_1^2`, the 99% cumulative
energy rank `k99`, the Gavish-Donoho Marchenko-Pastur signal rank `k_MP`, the
Frobenius norm and a status (`OPTIMAL`, `OVER_PARAMETERIZED` at `k99/rank <= 0.55`,
`RANK_COLLAPSED`, `DEAD_LAYER`). `guidance` in the report compares the summary means
with `DIM`:

- `energy_use = mean k99 / DIM`: near 1 means the early update needs the full rank;
  much below 1 means the rank is over-provisioned;
- `mp_share = mean k_MP / DIM`: the denoised signal rank relative to DIM;
- `status_counts`: how many layers the CLI calls optimal / over-parameterized / collapsed.

## Interpretation and limits

This audits the adapter delta at an early step, i.e. *utilization*, not the task's
rank requirement. The theoretically grounded rank-need estimator is the spectral gap
of the pre-finetuning gradient `G0 = dL/dW(W0)` (Karuturi et al., Spectral Dynamics of
Low-Rank Adaptation, in the gradience repository): per-layer `r_hat` = the largest
gap of `sigma(G0)`, or `max{r : sigma_r(G0)^2 >= C d_max/n}`. That needs gradient
taps and is not part of this check; the planned budget is 8 micro-batches x 512
sampled tokens per layer, one-shot, no dense `d x d` matrices (QR-core SVD on the
sampled `G X^T`). Until then, read the CLI numbers as a cheap early utilization signal
and prefer the trend across checkpoints over a single absolute value.

Early-delta spectra are also noisy: at step 10 the update is a few Adam steps old.
Raising `step` trades "weights are still near W0" against "the delta is less noise".

## Options

| key | default | meaning |
| --- | --- | --- |
| `step` | 10 | the one-shot step (>= 1) |
| `layers` | True | also parse the per-layer table |
| `keep` | False | keep the temporary adapter file |
| `timeout` | 300.0 | per-CLI-call timeout in seconds |
