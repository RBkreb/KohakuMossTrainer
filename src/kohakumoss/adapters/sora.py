"""SORA conditioning on any additive adapter kind (arXiv:2511.00051), two orthogonal
switches: Post-Ortho, a near-orthogonal low-rank rotation of the layer's whole output
(stacks with DoRA and NaRA), and Pre-Diag, a learned per-channel scale of the base
output that replaces DoRA. Parameters live on the LyCORIS module, so checkpoints and
parameter groups carry them; the export delta builders fold them into dW exactly."""

import math

import torch
from torch import Tensor, nn

F32 = torch.float32
SORA_KEYS = ("sora_C", "sora_D", "sora_diag", "sora_s_p0")


def active(lyco) -> bool:
    return getattr(lyco, "sora_C", None) is not None or getattr(
        lyco, "sora_pre_diag", False
    )


def attach(lyco, linear: nn.Module, config) -> None:
    """The configured SORA parameters on a freshly built module (trainable)."""
    if config.sora_post_rank > 0:
        out, r = linear.out_features, config.sora_post_rank
        c = torch.empty(out, r)
        nn.init.kaiming_uniform_(c, a=math.sqrt(5), mode="fan_out")
        lyco.sora_C = nn.Parameter(c)
        # D = C makes the initial skew term zero: P starts at identity.
        lyco.sora_D = nn.Parameter(c.clone())
        lyco.register_buffer(
            "sora_s_p0", torch.tensor(float(config.sora_initial_s_p))
        )
    if config.sora_pre_diag:
        lyco.sora_pre_diag = True
        lyco.sora_diag = nn.Parameter(torch.ones(linear.out_features))


def pop_tensors(parts: dict) -> dict:
    """Take the SORA tensors out of a file's per-layer set (LyCORIS must not see them)."""
    return {k: parts.pop(k) for k in SORA_KEYS if k in parts}


def native_tensors(lyco) -> dict:
    """The SORA tensors of a module: LyCORIS's custom state_dict only emits its
    algorithm's weight list, so the native export adds these beside it."""
    out = {}
    for name in SORA_KEYS:
        value = getattr(lyco, name, None)
        if value is not None:
            out[name] = value.detach()
    return out


def attach_tensors(lyco, tensors: dict) -> None:
    """SORA tensors of a loaded file onto its rebuilt module (frozen)."""
    if "sora_C" in tensors:
        lyco.sora_C = nn.Parameter(tensors["sora_C"].float(), requires_grad=False)
        lyco.sora_D = nn.Parameter(tensors["sora_D"].float(), requires_grad=False)
        s0 = tensors.get("sora_s_p0")
        value = torch.tensor(1.0) if s0 is None else s0.float().reshape(())
        lyco.register_buffer("sora_s_p0", value)
    if "sora_diag" in tensors:
        lyco.sora_pre_diag = True
        lyco.sora_diag = nn.Parameter(
            tensors["sora_diag"].float(), requires_grad=False
        )


def _s_p(lyco, c: Tensor, d: Tensor) -> Tensor:
    """The skew term's bound: s_p * (||C|| + ||D||) <= s_p0, capped at 1."""
    return (float(lyco.sora_s_p0) / (c.norm() + d.norm() + 1e-6)).clamp(max=1.0)


def rotate_out(lyco, y: Tensor) -> Tensor:
    """(R - I) y for activation rows y (..., out), fp32: the low-rank skew term,
    never the dense out x out rotation."""
    c = lyco.sora_C.to(F32)
    d = lyco.sora_D.to(F32)
    s = _s_p(lyco, c, d)
    yf = y.to(F32)
    return s * ((yf @ c) @ d.T - (yf @ d) @ c.T)


def rotate_rows(lyco, m: Tensor) -> Tensor:
    """R m for a weight / bias whose rows are output channels, fp32."""
    c = lyco.sora_C.detach().to(device=m.device, dtype=F32)
    d = lyco.sora_D.detach().to(device=m.device, dtype=F32)
    s = _s_p(lyco, c, d)
    return m + s * (d @ (c.T @ m) - c @ (d.T @ m))


def post_ortho(fn, lyco, m: float):
    """x -> fn(x) + m (R - I) fn(x): exact at m = 0 and 1, a strength lerp between
    (the m^2 cross term is second order in the bounded skew)."""

    def forward(x: Tensor) -> Tensor:
        y = fn(x)
        return (y.to(F32) + m * rotate_out(lyco, y)).to(y.dtype)

    return forward


def pre_diag(fn, lyco, m: float):
    """x -> d_m fn(x) + (1 - d_m) delta_m(x), d_m = 1 + m (d - 1): exactly
    base + m ((d - 1) base + delta) at every strength, one extra low-rank delta."""

    def forward(x: Tensor) -> Tensor:
        y = fn(x)
        delta = lyco.bypass_forward_diff(x, m).to(F32)
        dm = 1.0 + m * (lyco.sora_diag.to(F32) - 1.0)
        return (dm * y.to(F32) + (1.0 - dm) * delta).to(y.dtype)

    return forward


def wrap_infer(make, lyco, m: float):
    """A chain builder `make(prev)` with this module's SORA effect at strength m."""
    if not active(lyco):
        return make

    def wrapped(prev):
        fn = make(prev)
        if getattr(lyco, "sora_pre_diag", False):
            fn = pre_diag(fn, lyco, m)
        if getattr(lyco, "sora_C", None) is not None:
            fn = post_ortho(fn, lyco, m)
        return fn

    return wrapped


def delta_at(t, inner, multiplier: float):
    """The (dW, db) builder with SORA folded in weight space: Post-Ortho rotates
    W + dW (and the bias), Pre-Diag scales W's rows; `inner` builds at strength 1.
    The fold runs on the host: checkpoint exports happen mid-run, and the rotation's
    full-size fp32 intermediates must not stack on the training-resident device."""
    lyco = t.lyco
    post = getattr(lyco, "sora_C", None) is not None
    pre = getattr(lyco, "sora_pre_diag", False)

    def build() -> tuple[Tensor, Tensor | None]:
        dw, db = inner()
        device = dw.device
        w = t.base_fn(F32).cpu()
        dw = dw.cpu()
        d = None
        if pre:
            d = lyco.sora_diag.detach().to(device="cpu", dtype=F32)
            dw = dw + (d - 1.0).reshape(-1, *[1] * (w.dim() - 1)) * w
        if post:
            # Pre-Diag first, then the rotation: W' = R (diag(d) W + dW), the
            # forward's order (the scale inside, the rotation outside).
            dw = rotate_rows(lyco, w + dw) - w
        bias = t.base_bias()
        if bias is not None:
            b = bias.detach().to(device="cpu", dtype=F32)
            total = b if db is None else b + db.cpu()
            if pre:
                total = total * d
            if post:
                total = rotate_rows(lyco, total)
            db = total - b
        if multiplier != 1.0:
            dw = dw * multiplier
            db = None if db is None else db * multiplier
        return dw.to(device), None if db is None else db.to(device)

    return build
