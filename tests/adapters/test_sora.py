"""SORA, small nets (no model): the bound forwards and export deltas match independent
fp64 formulas (Post-Ortho alone, with DoRA, Pre-Diag, both combined), the lycoris /
comfy exports round-trip through the inference stack, and the configuration refuses
Pre-Diag with DoRA or dropout."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kohakumoss.adapters import AdapterConfig, apply
from kohakumoss.adapters.infer.inference import load
from kohakumoss.adapters.train.scope import evaluated
from kohakumoss.model.linear import MossLinear

pytestmark = pytest.mark.cpu
D = 24


def linear() -> MossLinear:
    lin = MossLinear(D, D)
    lin.init_weights()
    return lin


def _holder(lin: nn.Module) -> nn.Module:
    """One linear at a MOSS module path (a container, no forward)."""
    block = nn.Module()
    block.proj = nn.ModuleDict({"q": lin})
    root = nn.Module()
    root.layers = nn.ModuleList([block])
    return root


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([self._block(), self._block()])

    @staticmethod
    def _block():
        block = nn.Module()
        block.proj = nn.Module()
        for name in ("q", "k", "v", "o"):
            setattr(block.proj, name, linear())
        return block

    def forward(self, x):
        for b in self.layers:
            p = b.proj
            x = x + p.o(p.v(x) * p.k(x).mean(-1, keepdim=True))
        return x


def config(**kw) -> AdapterConfig:
    base = dict(algo="lora", preset="attn", dim=4, alpha=4.0, backend="torch")
    return AdapterConfig(**{**base, **kw})


def prepared(cfg):
    torch.manual_seed(7)
    model = Net()
    x = torch.randn(3, 16, D)
    aset = apply(model, cfg, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for t in aset.targets:
            lyco = t.lyco
            if getattr(lyco, "sora_C", None) is not None:
                lyco.sora_C.normal_(0, 0.05)
                lyco.sora_D.normal_(0, 0.05)
            if getattr(lyco, "sora_pre_diag", False):
                lyco.sora_diag.uniform_(0.7, 1.3)
            if hasattr(lyco, "lora_up"):
                lyco.lora_up.weight.normal_(0, 0.05)
    aset.bind()
    return model, x, aset


def rotation(lyco) -> torch.Tensor:
    c, d = lyco.sora_C.double(), lyco.sora_D.double()
    s = min(1.0, float(lyco.sora_s_p0) / (c.norm() + d.norm() + 1e-6).item())
    return torch.eye(D, dtype=torch.float64) + s * (d @ c.T - c @ d.T)


def unit_delta(lyco) -> torch.Tensor:
    return lyco.get_diff_weight(1.0)[0].double().reshape(lyco.shape) * lyco.scale


def assert_layers(aset, x, wref_of, bref_of=None):
    with torch.no_grad():
        for t in aset.targets:
            w = t.base_weight(torch.float64)
            dw = unit_delta(t.lyco)
            wref = wref_of(t.lyco, w, dw)
            b = t.base_bias()
            bias = None if b is None else b.double()
            if bref_of is not None and bias is not None:
                bias = bref_of(t.lyco, bias)
            ref = F.linear(x.double(), wref, bias)
            got = t.linear(x).double()
            assert (got - ref).abs().max().item() < 2e-4


def test_post_ortho_matches_fp64():
    _, x, aset = prepared(config(sora_post_rank=2))
    assert_layers(
        aset,
        x,
        lambda lyco, w, dw: rotation(lyco) @ (w + dw),
        lambda lyco, b: rotation(lyco) @ b,
    )
    aset.remove()


def test_post_ortho_with_dora_matches_fp64():
    _, x, aset = prepared(config(sora_post_rank=2, dora=True))

    def wref(lyco, w, dw):
        merged = w + dw
        mag = lyco.dora_scale.double().reshape(-1, 1)
        return rotation(lyco) @ (mag * merged / merged.norm(dim=1, keepdim=True))

    assert_layers(aset, x, wref, lambda lyco, b: rotation(lyco) @ b)
    aset.remove()


def test_pre_diag_matches_fp64():
    _, x, aset = prepared(config(sora_pre_diag=True))

    def wref(lyco, w, dw):
        return lyco.sora_diag.double().reshape(-1, 1) * w + dw

    assert_layers(aset, x, wref, lambda lyco, b: lyco.sora_diag.double() * b)
    aset.remove()


def test_combined_order_matches_fp64():
    _, x, aset = prepared(config(sora_post_rank=2, sora_pre_diag=True))

    def wref(lyco, w, dw):
        d = lyco.sora_diag.double().reshape(-1, 1)
        return rotation(lyco) @ (d * w + dw)

    assert_layers(
        aset, x, wref, lambda lyco, b: rotation(lyco) @ (lyco.sora_diag.double() * b)
    )
    aset.remove()


def test_export_delta_matches_forward():
    _, x, aset = prepared(config(sora_post_rank=2, sora_pre_diag=True))
    with torch.no_grad():
        for t in aset.targets:
            with evaluated([t]):
                dw, db = t.delta(1.0)
            w = t.base_weight(torch.float64)
            b = t.base_bias()
            bias = None if b is None else b.double()
            if db is not None:
                bias = db.double() if bias is None else bias + db.double()
            ref = F.linear(x.double(), w + dw.double(), bias)
            assert (t.linear(x).double() - ref).abs().max().item() < 2e-4
    aset.remove()


@pytest.mark.parametrize("fmt", ["lycoris", "comfy"])
def test_roundtrip(fmt):
    model, x, aset = prepared(config(sora_post_rank=2, sora_pre_diag=True))
    with torch.no_grad():
        trained = model(x).clone()
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / f"a_{fmt}.safetensors")
        aset.export(path, fmt, dtype=torch.float32)
        aset.remove()
        with torch.no_grad():
            base = model(x).clone()
        stack = load(model, path, mode="bypass")
        with torch.no_grad():
            loaded = model(x)
        assert (loaded - trained).abs().max().item() < 2e-4
        stack.remove()
    # The adapter is gone: the model must be back at its own weights, bit-exactly.
    with torch.no_grad():
        assert torch.equal(model(x), base)


def test_pre_diag_multiplier_exact():
    model, x, aset = prepared(config(sora_pre_diag=True))
    aset.set_multiplier(0.5)
    with torch.no_grad():
        for t in aset.targets:
            with evaluated([t]):
                dw, db = t.delta(0.5)
            w = t.base_weight(torch.float64)
            b = t.base_bias()
            bias = None if b is None else b.double()
            if db is not None:
                bias = db.double() if bias is None else bias + db.double()
            ref = F.linear(x.double(), w + dw.double(), bias)
            assert (t.linear(x).double() - ref).abs().max().item() < 2e-4
    aset.remove()


def test_pre_diag_refuses_dora_and_dropout():
    with pytest.raises(ValueError, match="replaces DoRA"):
        apply(Net(), config(sora_pre_diag=True, dora=True))
    with pytest.raises(ValueError, match="reads the delta twice"):
        apply(Net(), config(sora_pre_diag=True, dropout=0.1))
    with pytest.raises(ValueError, match="SORA does not apply"):
        apply(Net(), config(algo="ia3", sora_post_rank=1))


def folded_export(rows, **kw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(folded output, s, unfolded reload's output) of one layer trained on the base
    folded by `rows`: the lycoris export carries the fold, so the reload on the unfolded
    base times the fold must equal the folded run (`folded == S . unfolded`)."""
    torch.manual_seed(7)
    s = torch.as_tensor(rows, dtype=torch.float32)
    folded_lin = linear()
    plain_lin = linear()
    with torch.no_grad():
        w = folded_lin.weight.detach().clone()
        plain_lin.weight.copy_(w)
        folded_lin.weight.copy_(w * s.reshape(-1, 1))
    x = torch.randn(3, 16, D)
    aset = apply(_holder(folded_lin), config(**kw), dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for t in aset.targets:
            lyco = t.lyco
            if getattr(lyco, "sora_C", None) is not None:
                lyco.sora_C.normal_(0, 0.05)
                lyco.sora_D.normal_(0, 0.05)
            if getattr(lyco, "sora_pre_diag", False):
                lyco.sora_diag.uniform_(0.7, 1.3)
            if hasattr(lyco, "lora_up"):
                lyco.lora_up.weight.normal_(0, 0.05)
    aset.bind()
    with torch.no_grad():
        trained = folded_lin(x).clone()
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "folded.safetensors")
        aset.export(
            path,
            "lycoris",
            dtype=torch.float32,
            row_scales={t.path: s for t in aset.targets},
        )
        aset.remove()
        stack = load(_holder(plain_lin), path, mode="bypass")
        with torch.no_grad():
            reloaded = plain_lin(x)
        stack.remove()
    return trained, s, reloaded


@pytest.mark.parametrize(
    "kw",
    [
        {"sora_post_rank": 2},
        {"sora_post_rank": 2, "sora_pre_diag": True},
        {"sora_post_rank": 2, "dora": True},
    ],
)
def test_uniform_fold_exports_as_lycoris(kw):
    """A uniform fold scale is a multiple of the identity on the output axis, so it
    commutes with the rotation: the factor set carries it exactly."""
    trained, s, reloaded = folded_export([3.0] * D, **kw)
    want = reloaded * s.reshape(1, 1, -1)
    assert (trained - want).abs().max().item() < 2e-4


def test_per_row_fold_refuses_lycoris():
    """A per-row fold does not commute with the rotation, and no factor term carries
    it: the export refuses instead of writing a silently wrong file."""
    with pytest.raises(ValueError, match="per-output-row fold scale"):
        folded_export(torch.linspace(0.5, 2.0, D).tolist(), sora_post_rank=2)
