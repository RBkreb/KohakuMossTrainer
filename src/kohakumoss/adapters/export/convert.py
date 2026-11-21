"""Per-layer tensor sets for each export format: `native` is what LyCORIS saves,
`comfy` what ComfyUI's weight adapters evaluate to the same dW (see
docs/guides/export-to-comfy.md)."""

import torch
from torch import Tensor

from kohakumoss.adapters import sora
from kohakumoss.adapters.build.target import NormTarget, Target
from kohakumoss.adapters.kinds.common import names
from kohakumoss.adapters.kinds.record import Stacking


def native(t: Target) -> dict[str, Tensor]:
    """The LyCORIS module's own state dict (scalar folded, LyCORIS alpha rules),
    plus any SORA parameters its custom state_dict does not emit."""
    out = {k: v.detach() for k, v in t.lyco.state_dict().items()}
    out.update(sora.native_tensors(t.lyco))
    return out


def norm_native(n: NormTarget) -> dict[str, Tensor]:
    w, b = n.delta()
    out = {"w_norm": w}
    if b is not None:
        out["b_norm"] = b
    return out


def row_norm(w: Tensor) -> Tensor:
    """Per-output-row L2 norm, [O, 1]."""
    return w.reshape(w.shape[0], -1).norm(dim=1, keepdim=True)


def _comfy_dora_scale(t: Target, dora_scale: Tensor) -> Tensor:
    """ComfyUI normalises output-axis DoRA by ||W|| instead of ||W + dW||; fold that in."""
    if not t.wd_on_out:
        return dora_scale
    w = t.base_weight(torch.float32)
    dw = t.unit_fn().to(w.device)
    eps = torch.finfo(torch.float32).eps
    ratio = (row_norm(w) + eps) / (row_norm(w + dw) + eps)
    return dora_scale.float() * ratio.to(dora_scale.device)


def comfy(t: Target) -> dict[str, Tensor]:
    """Tensors ComfyUI evaluates to this layer's dW at strength 1. A SORA-conditioned
    layer exports its exact dense diff: the rotation / base scale touches W itself,
    which no low-rank factor set expresses (DoRA's row norm included)."""
    if sora.active(t.lyco):
        # Host tensors: the export set accumulates every layer's dense diff.
        return {k: v.cpu() for k, v in scaled_diff(t, 1.0).items()}
    out = t.record.comfy_tensors(t) if t.record.comfy_tensors is not None else native(t)
    if t.wd and "dora_scale" in out:
        out["dora_scale"] = _comfy_dora_scale(t, out["dora_scale"])
    return out


def reciprocal(scale: Tensor | float) -> Tensor | float:
    return 1.0 / scale.float() if isinstance(scale, Tensor) else 1.0 / float(scale)


def per_row(scale) -> bool:
    """`scale` differs between output rows. The only fold shape a factor set cannot
    carry through a SORA rotation: a uniform factor commutes with it, a per-row one
    does not (`R S != S R`)."""
    return (
        isinstance(scale, Tensor)
        and scale.numel() > 1
        and not bool((scale == scale.flatten()[0]).all())
    )


def scaled_diff(t: Target, scale) -> dict[str, Tensor]:
    """{"diff": dW (, "diff_b": db)} at multiplier 1, rows times `scale` (uniform or
    per output row)."""
    dw, db = t.delta(1.0)
    s = scale.float() if isinstance(scale, Tensor) else torch.tensor(float(scale))
    s = s.to(dw.device).reshape(-1, *[1] * (dw.dim() - 1))
    out = {"diff": dw * s}
    if db is not None:
        out["diff_b"] = db * s.flatten()
    return out


def _rows_of(scale, like: Tensor) -> Tensor:
    s = scale.float() if isinstance(scale, Tensor) else torch.tensor(float(scale))
    s = s.to(like.device).flatten()
    return s.reshape(-1, *[1] * (like.dim() - 1)) if s.numel() > 1 else s.reshape(())


def _row_scaled(tensors: dict, scaled: tuple, scale) -> dict:
    out = dict(tensors)
    for name in scaled:
        out[name] = out[name] * _rows_of(scale, out[name]).to(out[name].dtype)
    return out


def apply_row_scale(t: Target, tensors: dict, scale: Tensor | float) -> dict:
    """Multiply this layer's dW rows by `scale` (uniform or per output row) in the
    same tensor set: the output-row factor of each term, a uniform factor of each term,
    or an exact `diff` when the set has neither for this scale."""
    uniform = not per_row(scale)
    if isinstance(scale, Tensor) and uniform:
        scale = float(scale.flatten()[0]) if scale.numel() else 1.0
    if "diff" in tensors:
        return _row_scaled(tensors, names(("diff", "diff_b"), tensors), scale)
    if t.record.stacking == Stacking.COMMUTING and uniform:
        return dict(tensors)
    out_axis_dora = not t.wd or t.wd_on_out
    scaled = t.record.row_scale_tensors(tensors) if out_axis_dora else ()
    if not scaled and uniform:
        scaled = t.record.uniform_scale_tensors(tensors)
    if not scaled:
        return scaled_diff(t, scale)
    if "dora_scale" in tensors:
        scaled = (*scaled, "dora_scale")
    return _row_scaled(tensors, scaled, scale)
