"""Choose each layer's bound forward once, from its kind record and the bind context.

Every decision here happens at bind time and yields a distinct callable; the bound
forward is that callable, called directly.
"""

from functools import partial

from torch import nn

from kohakumoss.adapters import sora
from kohakumoss.adapters.bind import compose, dora, eager, forwards
from kohakumoss.adapters.bind.context import BindCtx
from kohakumoss.adapters.bind.kernel_ops import kernel_ops
from kohakumoss.adapters.build.target import Target


def dropping(lyco) -> bool:
    """Input or rank dropout configured on this module (the kernel ops carry neither)."""
    drop = getattr(lyco, "dropout", 0.0)
    input_drop = (
        isinstance(drop, nn.Dropout) if isinstance(drop, nn.Module) else drop > 0
    )
    return input_drop or getattr(lyco, "rank_dropout", 0.0) > 0


def _merge(t: Target, ctx: BindCtx):
    rec = t.record
    if rec.kernel_merge is not None:
        return rec.kernel_merge(t, ctx)
    unit = compose.with_scalar(t.lyco, rec.kernel_diff_weight(t, ctx.ops))
    if unit is None:
        return None
    merge = None
    if ctx.m == 1.0 and not compose.has_scalar_param(t.lyco):
        merge = rec.kernel_merged_weight(t, ctx.ops)
    return compose.merged(t, unit, ctx.m, merge)


def _eager(t: Target, ctx: BindCtx, split: bool = False):
    if ctx.mode == "merge":
        lin = compose.lin_of(t)
        return eager.eager_merged(t.base_fn, t.lyco.shape, t.delta_at(ctx.m), lin)
    if t.record.module_forward is not None:
        return t.record.module_forward(t, ctx)
    if split:
        return eager.module_bypass_split(t.layer, t.lyco, ctx.m)
    return eager.module_bypass(t.layer, t.lyco, ctx.m)


def _dora_kernel(t: Target, ctx: BindCtx, mix, lin):
    """DoRA through the kernel ops (W + dW read inside the decompose op where it can),
    or None where the kind has no kernel dW."""
    unit = compose.with_scalar(t.lyco, t.record.kernel_diff_weight(t, ctx.ops))
    if unit is None:
        return None
    with_delta = ctx.ops.decompose_with_delta(t.lyco, t.wd_on_out)
    if with_delta is not None:
        return forwards.dora_with_delta(t.base_fn, t.lyco, unit, with_delta, mix, lin)
    decompose = ctx.ops.decompose(t.lyco, t.wd_on_out)
    if decompose is None:
        return None
    return forwards.dora(t.base_fn, t.lyco, unit, decompose, mix, lin)


def _stored_dora(t: Target):
    """The DoRA bypass of a stored base: input or output axis, with or without bias."""
    if not t.wd_on_out:
        return dora.bypass_in
    return dora.bypass_out if t.linear.bias is None else dora.bypass_out_biased


def _dora(t: Target, ctx: BindCtx):
    detach = ctx.detach and ctx.training
    if t.record.needs_sigma:
        # A dynamic per-sample delta: the magnitude is recomputed from
        # W + dW(sigma) every forward, never cached or fused (the core moves).
        if not t.wd_on_out:
            fn = dora.nara_in
        elif t.linear.bias is None:
            fn = dora.nara_out
        else:
            fn = dora.nara_out_biased
        return partial(fn, t.layer, t.base_fn, t.lyco, ctx.m, detach)
    if t.stored:
        # A stored base: the DoRA bypass route on the fixed delta.
        fn = _stored_dora(t)
        return partial(fn, t.layer, t.base_fn, t.unit_fn, t.lyco, ctx.m, detach)
    mix, lin = compose.mix_of(ctx.m), compose.lin_of(t)
    if ctx.use_ops and not detach:
        fn = _dora_kernel(t, ctx, mix, lin)
        if fn is not None:
            return fn
    if detach:
        fn = dora.detached_out if t.wd_on_out else dora.detached_in
        decompose = dora.detached(fn, t.lyco)
    else:
        decompose = dora.decomposer(t, ctx.backend)
    return eager.eager_dora(t.base_fn, t.unit_fn, decompose, mix, lin)


PRE_DIAG_MODES = ("split", "base", "legacy")
"""How Pre-Diag gets the base / delta split it needs, best first:

- `split` (default): the forward yields (base, delta) once; one delta per forward,
  the same pair of tensors the sum cost. Pre-Diag layers take the eager bypass even
  where a kernel op exists, because the kernel op's delta and the module's are the
  same quantity computed separately -- `split` is what makes it one.
- `base`: the sum stays intact and the base is recomputed (`y + m(d-1) base`); one
  delta plus one extra base GEMM -- strictly worse than `split` unless a base
  forward is cheaper than the delta, which no kind here is.
- `legacy`: `pre_diag` asks the module for the delta a second time (one extra kernel
  / eager delta, or a whole hypernetwork forward under NaRA).

Kept as `KOHAKUMOSS_PREDIAG` so the three can be measured against each other on one
build; the default is the fix."""


def pre_diag_mode() -> str:
    import os

    mode = os.environ.get("KOHAKUMOSS_PREDIAG", "split")
    return mode if mode in PRE_DIAG_MODES else "split"


def _pre_diag_split_target(t: Target, ctx: BindCtx) -> bool:
    """Whether this layer's Pre-Diag can run the (base, delta) pair form: the two
    halves must be the layer's own forward and the module's delta. Kinds with their
    own forward (oft / boft / glora / ia3), DoRA (its own single-delta route) and a
    stored base keep the summing form. Post-Ortho is not disqualifying: it wraps the
    Tensor `pre_diag_split` returns.

    A kernel op is not disqualifying either -- it is *avoided* for these layers: the
    kernel computes the same delta the pair needs, and asking for both would be the
    very duplication this removes."""
    return (
        not t.stored
        and not t.wd
        and t.record.own_forward is None
        and t.record.module_forward is None
    )


def bind_target(
    t: Target,
    mode: str,
    backend: str,
    m: float,
    detach: bool,
    module_p: float,
    autotune: bool = False,
    fuse_base_add: bool = True,
    dynamic: str = "tokens",
):
    """The forward to install on `t.linear` for this mode / backend / multiplier /
    training state. Its kernel ops are autotuned (`autotune`) or fixed to `backend`,
    the base add inside the op where one exists (`fuse_base_add`); the autotuned ops
    are kept on `t.autotuned` for the prewarm."""
    lyco = t.lyco
    lyco.multiplier = m
    t.backend = backend
    training = lyco.training
    ops = kernel_ops(backend, autotune, fuse_base_add, dynamic)
    ctx = BindCtx(
        mode="bypass" if t.stored else mode,
        backend=backend,
        ops=ops,
        m=m,
        detach=detach,
        module_p=module_p,
        training=training,
        use_ops=not (training and dropping(lyco)),
        draws=t.draws,
    )
    rec = t.record
    wants_pre_diag = bool(getattr(lyco, "sora_pre_diag", False))
    strategy = pre_diag_mode() if wants_pre_diag else "legacy"
    takes_split = (
        wants_pre_diag
        and strategy == "split"
        and ctx.mode != "merge"
        and _pre_diag_split_target(t, ctx)
    )
    eager_bypass = False
    if rec.own_forward is not None:
        fn = rec.own_forward(t, ctx)
    elif t.wd:
        fn = _dora(t, ctx)
    else:
        fn = None
        if ctx.use_ops and not takes_split:
            fn = rec.kernel_bypass(t, ctx) if ctx.mode == "bypass" else _merge(t, ctx)
        if fn is None:
            # `_eager` resolves to `module_bypass` here: no own / module forward, and
            # its other branch is the merge path, which `kernel_bypass`'s None cannot
            # have come from in bypass mode.
            eager_bypass = ctx.mode != "merge" and _pre_diag_split_target(t, ctx)
            fn = _eager(t, ctx, split=eager_bypass and takes_split)
    t.autotuned = ops.autotuned
    # SORA inside the module-dropout wrap: a dropped module skips its conditioning too.
    if wants_pre_diag:
        if eager_bypass and takes_split:
            fn = sora.pre_diag_split(fn, lyco, ctx.m)
        elif strategy == "base":
            fn = sora.pre_diag_base(
                fn, lyco, ctx.m, partial(_base_forward, t.layer)
            )
        else:
            fn = sora.pre_diag(fn, lyco, ctx.m)
    if getattr(lyco, "sora_C", None) is not None:
        fn = sora.post_ortho(fn, lyco, ctx.m)
    if rec.own_forward is None and module_p > 0 and training:
        fn = eager.module_dropout(t.layer, t.draws, fn)
    return fn


def _base_forward(layer, x):
    """The layer's own forward (bias included), for the `base` Pre-Diag form."""
    lin = layer()
    return lin._impl(lin, x)


def training_dependent(t: Target, module_p: float, detach: bool) -> bool:
    """Whether the bound forward differs between train() and eval()."""
    return (
        t.record.training_dependent
        or module_p > 0
        or dropping(t.lyco)
        or (t.wd and detach)
    )
