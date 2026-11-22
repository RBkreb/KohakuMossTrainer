"""EVA data-driven initialisation (arXiv:2410.07170): incremental SVDs of the target
layers' input activations over calibration forwards through the training step (the
sigma sampler covers the noise levels), then the input-side factors start from the
converged principal directions with the output-side factors zero. LoRA / NaRA take
the components directly; LoKr / NaRA-LoKr take the top eigenvectors of the block
Gram of the reshaped components (the Kronecker-constrained input subspace). Uniform
rank: no cross-layer rank redistribution in this version."""

import torch
from torch import Tensor
from torch_incremental_pca import IncrementalPCA

from kohakumoss.logging.log import get_logger, hl, timed
from kohakumoss.training.build.loop import build_batcher, fixed_batches

log = get_logger(__name__)

DEFAULTS = {"batches": 32, "min_batches": 4, "sim_thresh": 0.99, "precision": "fp64"}
EVA_KEYS = frozenset(DEFAULTS)
PRECISIONS = ("fp64", "fp32")


class _Fp32Svd:
    """`torch.svd_lowrank` on an fp32 copy of its input, same q / niter as the package.

    `torch-incremental-pca` feeds `partial_fit`'s fp64 `vstack` straight into
    `svd_lowrank`, whose QR and GEMMs then run in fp64 (no tensor cores on consumer
    cards: 1/32 to 1/64 of fp32 throughput). This is the one hop that decides the
    dtype of the expensive part, and it is an instance attribute, so overriding it
    needs no copy of the package's `partial_fit`.

    `svd_lowrank` is randomized, so the subspace already varies run to run: with a
    fixed seed the fp32 and fp64 subspaces differ by a gap of 0.10 in the worst
    principal angle while two fp64 runs at different seeds differ by 0.92-0.98 (a
    4352x3072 layer, n=128). The precision difference is inside the method's own
    randomness; EVA="precision": "fp32" trades it for 9x on the calibration."""

    def __init__(self, q: int, niter: int):
        self.q = q
        self.niter = niter

    def __call__(self, x: Tensor):
        u, s, v = torch.svd_lowrank(x.float(), q=self.q, niter=self.niter)
        return u, s, v.mH


class SVDHook:
    """One layer's incremental input SVD; a component counts as converged when its
    cosine similarity across two consecutive fits reaches the threshold."""

    def __init__(self, n_components: int, sim_thresh: float, precision: str = "fp64"):
        if precision not in PRECISIONS:
            raise ValueError(f"EVA precision {precision!r} not in {PRECISIONS}")
        self.n_components = n_components
        q = 2 * n_components
        self.svd = IncrementalPCA(
            n_components=n_components,
            copy=True,
            lowrank=True,
            lowrank_q=q,
        )
        if precision == "fp32":
            # The package's own default `lowrank_niter`; a different value would change
            # the subspace for reasons unrelated to precision.
            self.svd.svd_fn = _Fp32Svd(q, niter=4)
        self.sim_thresh = sim_thresh
        self.converged = torch.zeros(n_components, dtype=torch.bool)

    def __call__(self, module, args):
        x = args[0] if isinstance(args, tuple) else args
        previous = getattr(self.svd, "components_", None)
        previous = None if previous is None else previous.clone()
        states = x.detach().reshape(-1, x.shape[-1])
        if states.shape[0] < self.n_components:
            return
        # The calibration forward runs under the run's autocast; the SVD's own
        # matmuls must stay fp32 (its QR has no bf16 kernel).
        with torch.autocast(device_type=x.device.type, enabled=False):
            self.svd.partial_fit(states.to(torch.float32))
            if previous is not None:
                sim = torch.nn.functional.cosine_similarity(
                    self.svd.components_, previous
                )
                self.converged = sim >= self.sim_thresh


def _fingerprint(x: Tensor) -> int:
    """A cheap hash of a subsample of the activation (q / k / v share one input)."""
    s = x.detach().reshape(-1, x.shape[-1])[::37, :64].float().cpu()
    return hash(s.numpy().tobytes())


COMPONENT_CAP = 512


def _lokr_rows(lyco) -> int:
    """The component rows a Kronecker w2_b init needs: one per (w1 row, w2_b rank)."""
    w1 = lyco.lokr_w1 if lyco.use_w1 else lyco.lokr_w1_a @ lyco.lokr_w1_b
    return int(w1.shape[0] * lyco.lokr_w2_b.shape[0])


def _components(t) -> int:
    """The input-side rank the layer's factors can express, capped at `COMPONENT_CAP`
    so one layer's component matrix stays small next to the model. A LoKr layer whose
    init needs more rows than the cap cannot be initialised; `_init_from` returns why,
    and `calibrate` reports it instead of leaving the layer at its random init
    silently."""
    lyco = t.lyco
    if hasattr(lyco, "lora_down"):
        return int(lyco.lora_dim)
    return min(_lokr_rows(lyco), COMPONENT_CAP)


def _init_lora(lyco, comp: Tensor) -> str | None:
    r = lyco.lora_dim
    lyco.lora_down.weight.copy_(comp[:r].to(lyco.lora_down.weight))
    lyco.lora_up.weight.zero_()
    return None


def _init_lokr(lyco, comp: Tensor) -> str | None:
    """w2_b (the zero-initialised input factor) from the dominant n2-directions of
    the components reshaped over the kron input blocks; w2_a stays zero. Returns None
    when initialised, else why it could not be."""
    rows = _lokr_rows(lyco)
    if comp.shape[0] < rows:
        return (
            f"needs {rows} component rows, {comp.shape[0]} available "
            f"(COMPONENT_CAP={COMPONENT_CAP}): lower ALGO_KWARGS['factor'] or DIM"
        )
    w1 = lyco.lokr_w1 if lyco.use_w1 else lyco.lokr_w1_a @ lyco.lokr_w1_b
    m1, n1 = w1.shape
    r2, n2 = lyco.lokr_w2_b.shape
    v = comp[:rows].reshape(rows, n1, n2).float()
    gram = torch.einsum("rbi,rbj->ij", v, v)
    _, vecs = torch.linalg.eigh(gram)
    top = vecs[:, -r2:].T.flip(0)
    lyco.lokr_w2_b.copy_(top.to(lyco.lokr_w2_b))
    lyco.lokr_w2_a.zero_()
    return None


def _init_from(t, comp: Tensor) -> str | None:
    """The layer's input-side factor from its activation subspace. Returns None when
    initialised, else why the layer kept its random init."""
    lyco = t.lyco
    if hasattr(lyco, "lora_down"):
        return _init_lora(lyco, comp)
    if hasattr(lyco, "lokr_w2_b") and not lyco.use_w2:
        return _init_lokr(lyco, comp)
    return "a full-matrix w2 has no rank for the core"


def _broadcast(aset) -> None:
    """Every rank calibrated on its own data shard; rank 0's init is the run's."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if torch.distributed.get_world_size() > 1:
        for p in aset.trainable_params():
            torch.distributed.broadcast(p.data, src=0)


def calibrate(k: dict, built, step, loader, device, autocast) -> None:
    """The EVA init of a fresh run: calibration forwards, then the factor copy."""
    cfg = {**DEFAULTS, **(k.get("EVA") or {})}
    aset = built.trainable
    targets = [t for t in aset.targets if t.record.supports_eva]
    if not targets:
        log.warning("EVA: no eva-capable target layer; keeping the random init")
        return
    with timed(
        log, f"EVA calibration ({len(targets)} layers, <= {cfg['batches']} batches)"
    ):
        batches = fixed_batches(
            loader, build_batcher(k, built, device), int(cfg["batches"])
        )
        if not batches:
            raise SystemExit("EVA: the data loader produced no calibration batch")
        marks, handles = {}, []
        for t in targets:

            def mark(module, args, output, path=t.path):
                x = args[0] if isinstance(args, tuple) else args
                marks[path] = _fingerprint(x)

            handles.append(t.linear.register_forward_hook(mark))
        with torch.no_grad():
            p = step.prepare(batches[0], 0)
            with autocast():
                built.backbone(p.z, p.text, p.sigma, p.seq)
            aset.end_step()
        for h in handles:
            h.remove()
        canonical, hooks, aliases = {}, {}, {}
        for t in targets:
            fp = marks[t.path]
            if fp in canonical:
                aliases[t.path] = canonical[fp]
                continue
            canonical[fp] = t.path
            hook = SVDHook(
                _components(t), float(cfg["sim_thresh"]), str(cfg["precision"])
            )
            handle = t.linear.register_forward_pre_hook(hook)
            hooks[t.path] = (t, hook, handle)
        converged, used = set(), 0
        for i, batch in enumerate(batches):
            with torch.no_grad():
                p = step.prepare(batch, i)
                with autocast():
                    built.backbone(p.z, p.text, p.sigma, p.seq)
                aset.end_step()
            used = i + 1
            for path, (t, hook, handle) in list(hooks.items()):
                if path in converged:
                    continue
                if hasattr(hook.svd, "components_") and bool(hook.converged.all()):
                    converged.add(path)
                    handle.remove()
            if len(converged) == len(hooks) and used >= int(cfg["min_batches"]):
                break
        fitted, skipped = 0, []
        with torch.no_grad():
            for t in targets:
                source = hooks.get(aliases.get(t.path, t.path))
                if source is None or not hasattr(source[1].svd, "components_"):
                    skipped.append(f"{t.path}: no activation SVD fitted")
                    continue
                why = _init_from(t, source[1].svd.components_.float())
                if why is None:
                    fitted += 1
                else:
                    skipped.append(f"{t.path}: {why}")
        for _, _, handle in hooks.values():
            handle.remove()
        _broadcast(aset)
    log.info(
        f"EVA: {hl(fitted)}/{len(targets)} layers initialised from activation SVDs "
        f"({used} batches, {len(converged)}/{len(hooks)} converged, "
        f"{len(aliases)} shared-input aliases)"
    )
    if skipped:
        shown = "; ".join(skipped[:4])
        more = f" (and {len(skipped) - 4} more)" if len(skipped) > 4 else ""
        log.warning(f"EVA: {len(skipped)} layers kept their random init: {shown}{more}")
