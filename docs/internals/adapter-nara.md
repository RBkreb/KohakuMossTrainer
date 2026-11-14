# NaRA: noise-aware rank adaptation

NaRA (Noise-Aware Rank Adaptation, Wang et al., ICML 2026) conditions the adapter on
the diffusion noise level. Between the two static LoRA factors sits a dynamic core

```
h = W0 x + B C(sigma) A x
C(sigma) = I_r + c_scale * F(e_sigma)
```

where `F` is a small MLP (the mapper) and `e` a Gaussian Fourier embedding of the
noise level. One mapper is **shared by every adapted layer**: the per-layer state is
only A (`lora_down`) and B (`lora_up`), and the core is evaluated from the step's
sigma. The mapper's last layer starts at zero, so `C = I` at init and NaRA reduces
to LoRA. The reference implementation targets masked diffusion LLMs (one scalar
noise level per batch); here the conditioning is per sample of the packed batch,
expanded to the layer's token axis exactly like T-LoRA's masks.

## Pieces

| piece | file | role |
| --- | --- | --- |
| `NaraHyper` | `adapters/nara/hyper.py` | embedding (`fourier` fixed / `mlp` learned / `raw`) + mapper; `sigma [B] -> C [B, r, r]` in fp32 |
| `NaRAModule` | `adapters/nara/module.py` | `LoConModule`'s A/B factors; `bypass_forward_diff` reads the context, gathers C per token, runs the core product |
| `NaraLohaModule`, `NaraLokrModule` | `adapters/nara/module.py` | the NARA-switch variants of LoHa / LoKr (below) |
| context | `adapters/nara/context.py` | the installed `(sigma, seq)`; set / clear / save / restore, one global like T-LoRA's mask registry |
| kind records | `adapters/kinds/nara.py` | `NARA` (ALGO=`nara`) and the derived `NARA_LOHA` / `NARA_LOKR` variants of the base kinds: `needs_sigma=True`, `shared_module` factory, no kernel ops, fixed-sigma dW and ComfyUI conversion; registers the modules in LyCORIS's factory |
| file load | `adapters/nara/load.py` | splits the `nara_hyper` stem off a file's tensors, rebuilds the hypernetwork from them and the recorded configuration |

## Ownership of the shared hypernetwork

`apply()` builds one `NaraHyper` per adapter set (`Kind.shared_module(config)`, sized
by `DIM` and the `nara_*` knobs), places it in the master dtype and registers it under
the model root as `nara_hyper` — where DDP's reducer and the offload walkers see it
(the block streamers only stream block lists, so it stays resident; it is a few MB).
`LycoAdapterSet` owns its state:

- every `NaRAModule` holds it through `set_shared` **outside the module tree**
  (`object.__setattr__`), so a layer's state dict never duplicates it;
- `_entries()` / `_live()` carry its parameters and buffers under the `nara_hyper.`
  stem: optimiser groups, EMA, checkpoints and `load_state_dict` cover it once;
- `tensors("lycoris")` writes the same stem; `remove()` / `merge_into()` detach it
  from the model root.

## The per-step flow

Training: `StepPlan.begin` installs `(sigma, seq)` when any target's record has
`needs_sigma`; `end` clears it; `capture` / `install` carry it with the masks and
draws, so a pipeline micro-batch's recompute reads its own noise level.
`AdapterConfig.has_step_state` is true, so the async-threaded recompute guard applies
unchanged.

Sampling: `sample_latents(..., on_sigma=...)` calls the hook with the sigma and seq
of every backbone forward (both CFG branches of `two_pass`, the doubled batch of
`packed`). `MossPipeline` passes the context setter when a loaded stack reports
`needs_sigma`; the training loop passes it to `from_components` when the trainable
set does. Without a context a NaRA forward raises — there is no silent fallback.

Each forward evaluates the hypernetwork on the context's sigma (`[B, r, r]`, fp32
per-sample conditioning math, like the time embedding), gathers one core per token
through the layer's axis rows (`adapters.train.tlora.rows_of`: joint / text / image /
sample), casts to the activation dtype and runs `bmm(mid, C)` between the down and up
projections. The core is computed inside the layer's own forward, so an activation
checkpoint recomputes it with the block — no cached tensor outlives its graph.

Cost: one small MLP forward and one `[N, r, r]` gather per adapted layer per step
(the reference computes C once per step; here each layer re-evaluates it, which keeps
the context free of autograd state and every parallel / recompute path correct). The
gathered core is saved for the layer's backward; with block checkpointing it is
recomputed instead. At r = 16 and a few thousand tokens this is tens of MB per layer
without checkpointing — prefer a checkpointing profile.

## Knobs

`ALGO_KWARGS` (all optional, recorded in the file metadata):

| key | default | meaning |
| --- | --- | --- |
| `nara_hidden1`, `nara_hidden2` | 256, 512 | mapper hidden widths |
| `nara_embed_dim` | 64 | embedding width (even for `fourier`) |
| `nara_embed` | `fourier` | `fourier` (fixed frequencies), `mlp` (learned), `raw` (the scalar) |
| `nara_fourier_scale` | 16.0 | frequency scale of the fixed Fourier embedding |
| `nara_c_scale` | 1.0 | eta: the core's residual scale |
| `export_sigma` | 0.5 | the fixed noise level of dW exports (below) |

Validation rejects `ADAPTER_MODE="merge"` (a sigma-dependent dW is no single merged
weight), rank dropout, and `use_tucker` / `use_scalar` / `rs_lora`.

## Export and inference

- **`lycoris` / training state**: per-layer `lora_up.weight`, `lora_down.weight`,
  `alpha` plus the `nara_hyper.*` tensors. The per-layer tensors alone are
  LoRA-shaped; the hypernetwork stem is what names the kind on load.
- **`comfy` / `merged` / `merge_into`**: a sigma-dependent delta cannot stay dynamic
  in a static file, so these evaluate C at `export_sigma` and write the plain LoRA
  `up = B C^T s`, `down = A` (or bake the same dW). An approximation at one noise
  level, as T-LoRA's mask-free export is; the native format keeps the full adapter.
- **Inference**: `AdapterStack.add` splits the hypernetwork stem, rebuilds the
  hypernetwork from the recorded spec / network args (fp32, frozen), and builds each
  layer as a `NaRAModule` wired to it, on dense or stored bases alike. The bypass
  chain reads the live context, so sampling must pass `on_sigma`
  (`stack.needs_sigma` reports it). `mode="merge"` applies the fixed-sigma dW.
- **Parallel layouts**: one hypernetwork shared by every layer cannot be stepped
  across FSDP shards or pipeline stages; `ALGO="nara"` and the NARA switch are
  rejected there (`training/checks/parallel.py`). Single GPU and DDP train it.

## The NARA switch

`NARA = True` turns the noise conditioning into a switch over the nara-capable kinds
(LoRA, LoHa, LoKr), like `DORA`:

- `ALGO = "lora"` / `"locon"` with the switch trains exactly ALGO=`nara` — the record
  and module are the existing ones.
- `ALGO = "loha"` / `"lokr"` train `NaraLohaModule` / `NaraLokrModule`: the base
  kind's module class extended with the shared hypernetwork. The derived records
  (`NARA_LOHA`, `NARA_LOKR`) keep the base kind's applier surface — targeting,
  optimiser groups, exports — and drop the kernel ops (the core lives outside the
  factors the kernels read), so these layers always run the module bypass. A
  config whose `ALGO` has no variant is refused (`NARA does not apply to ...`).

Where the core sits, per variant (`C = C(sigma)` of the current sample):

- **LoRA**: `dW = B C^T A`, one `[r, r]` core between the two factors — the paper's
  formula.
- **LoHa**: `dW = (w1a C^T w1b) o (w2a C^T w2b)` — both rank channels read the same
  core. The Hadamard product has no rank-factorized application, so the bypass
  rebuilds one dense `[out, in]` delta per sample of the batch (LoHa's own bypass
  rebuilds one per forward); prefer block checkpointing with it.
- **LoKr**: `dW = w1 (x) (w2a C^T w2b)` — the core sits in the factorized `w2`'s rank
  contraction, and the bypass chains it per token between the `w2` factors, exactly
  like LoRA's. A layout whose `w2` came out full has no rank for the core and is
  rejected at construction (lower `DIM`, raise `ALGO_KWARGS["factor"]`;
  `algo_lokr_nara` composes factor 8 with DIM 16).

DoRA composes with the switch on every variant: the delta stays per sample while the
magnitude is the fixed export-sigma one, so the DoRA bypass route (`bypass_out` /
`bypass_in`) runs on dense bases too. Exports follow the fixed-sigma rules above with
the core folded into the up factors (`lora_up`, both `hada_w*_a`, `lokr_w2_a`), and
the `lycoris` format keeps the per-layer factors of the base algorithm plus the
`nara_hyper` stem — the hypernetwork stem names the kind on load, the tensor names
pick the variant.