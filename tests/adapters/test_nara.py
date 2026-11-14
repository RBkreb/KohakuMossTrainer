"""NaRA, one linear (no model): the shared hypernetwork and the bound forward match an
independent fp64 formula (forward and backward, per-sample sigma over a packed layout),
the step state captures the noise level, the native / raw / comfy exports round-trip
through the inference stack, and the configuration refuses what the kind cannot do."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kohakumoss.adapters import AdapterConfig, AdapterStack, apply
from kohakumoss.adapters.bind.context import Draws
from kohakumoss.adapters.nara import clear_context, set_context
from kohakumoss.adapters.nara.context import get_context
from kohakumoss.adapters.nara.hyper import NaraHyper
from kohakumoss.adapters.train.steps import StepPlan
from kohakumoss.model.linear import MossLinear
from kohakumoss.types.seqinfo import build_seqinfo
from tests._util.error_stats import assert_no_regression
from tests._util.negative import must_fail

pytestmark = pytest.mark.cpu
OUT, IN = 48, 64
PATH = "layers.0.proj.q"
DIM = 8


@pytest.fixture(autouse=True)
def _no_context():
    yield
    clear_context()


def holder(lin: nn.Module) -> nn.Module:
    """The linear at a MOSS module path (a container, no forward)."""
    block = nn.Module()
    block.proj = nn.ModuleDict({"q": lin})
    root = nn.Module()
    root.layers = nn.ModuleList([block])
    return root


def dense_linear(w, b) -> MossLinear:
    lin = MossLinear(IN, OUT, bias=True)
    with torch.no_grad():
        lin.weight.copy_(w)
        lin.bias.copy_(b)
    return lin


def problem(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(OUT, IN, generator=g) * 0.1
    b = torch.randn(OUT, generator=g) * 0.1
    x = torch.randn(40, IN, generator=g)
    return w, b, x


def layout():
    """Two samples: 16 image tokens each, 3 and 5 text tokens (40 joint rows)."""
    seq = build_seqinfo(torch.tensor([[4, 4], [4, 4]]), torch.tensor([3, 5]))
    sigma = torch.tensor([0.2, 0.9])
    return sigma, seq


def adapted(lin: nn.Module, **kw):
    root = holder(lin)
    root.requires_grad_(False)
    config = AdapterConfig(
        algo="nara", preset="attn", dim=DIM, alpha=float(DIM), backend="torch", **kw
    )
    return apply(root, config)


def perturb(aset, seed: int = 3) -> None:
    """Every trainable tensor random (the zero-initialised factors included)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in aset.named_trainable():
            p.add_(torch.randn(p.shape, generator=g).to(p) * 0.3)


def ref_hyper(hyper: NaraHyper, sigma: torch.Tensor, dtype) -> torch.Tensor:
    """Independent C(sigma) of the hypernetwork's tensors, in `dtype`."""
    s = sigma.to(dtype).reshape(-1, 1)
    if hyper.embed == "fourier":
        proj = s @ hyper.fourier_w.to(dtype)
        e = torch.cat((proj.sin(), proj.cos()), dim=-1)
    elif hyper.embed == "mlp":
        e = F.silu(
            s @ hyper.embed_proj.weight.to(dtype).T + hyper.embed_proj.bias.to(dtype)
        )
    else:
        e = s
    h = e
    for m in hyper.mapper:
        if isinstance(m, nn.Linear):
            h = h @ m.weight.to(dtype).T + m.bias.to(dtype)
        else:
            h = F.silu(h)
    eye = torch.eye(hyper.dim, dtype=dtype)
    return eye + hyper.c_scale * h.view(-1, hyper.dim, hyper.dim)


def token_core(hyper, sigma, seq, dtype):
    core = ref_hyper(hyper, sigma, dtype)
    return core.index_select(0, seq.seg_joint.long())


def ref_forward(w, b, x, lyco, core, dtype):
    """base + scale * (x A^T C) B^T with one core per token, in `dtype`."""
    w, b, x = w.to(dtype), b.to(dtype), x.to(dtype)
    a = lyco.lora_down.weight.to(dtype)
    up = lyco.lora_up.weight.to(dtype)
    mid = x @ a.T
    mid = torch.bmm(mid.unsqueeze(1), core).squeeze(1)
    return x @ w.T + b + (mid @ up.T) * float(lyco.scale)


@pytest.mark.parametrize("embed", ["fourier", "mlp", "raw"])
def test_hyper_starts_as_identity(embed):
    hyper = NaraHyper(DIM, embed=embed)
    sigma = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(hyper(sigma), torch.eye(DIM).expand(3, DIM, DIM))


@pytest.mark.parametrize("embed", ["fourier", "mlp", "raw"])
def test_hyper_matches_fp64(embed):
    hyper = NaraHyper(DIM, embed=embed, embed_dim=32)
    with torch.no_grad():
        for p in hyper.parameters():
            p.add_(torch.randn(p.shape) * 0.3)
    sigma = torch.tensor([0.1, 0.6, 0.95])
    out = hyper(sigma)
    eager = ref_hyper(hyper, sigma, torch.float32)
    ref = ref_hyper(hyper, sigma, torch.float64)
    assert_no_regression(out, eager, ref, name=f"hyper-{embed}", test="adapters")
    must_fail(
        lambda: assert_no_regression(
            torch.eye(DIM).expand(3, DIM, DIM), eager, ref, test="adapters"
        ),
        "the mapper's delta dropped",
    )


def test_forward_and_backward_match_fp64():
    w, b, x = problem()
    sigma, seq = layout()
    aset = adapted(dense_linear(w, b))
    perturb(aset)
    lyco = aset.targets[0].lyco
    hyper = aset.shared
    set_context(sigma, seq)

    x_act = x.clone().requires_grad_(True)
    out = aset.targets[0].linear(x_act)
    core32 = token_core(hyper, sigma, seq, torch.float32)
    eager = ref_forward(w, b, x, lyco, core32, torch.float32)
    ref = ref_forward(
        w, b, x, lyco, token_core(hyper, sigma, seq, torch.float64), torch.float64
    )
    assert_no_regression(out, eager, ref, name="nara-forward", test="adapters")

    g = torch.randn_like(out)
    out.backward(g)
    grads = {
        "x": x_act.grad,
        "A": lyco.lora_down.weight.grad,
        "B": lyco.lora_up.weight.grad,
        "mapper": hyper.mapper[0].weight.grad,
    }

    def ref_grads(dtype):
        """The same graph over detached leaves, the hypernetwork rebuilt in `dtype`."""
        xr = x.detach().to(dtype).requires_grad_(True)
        a = lyco.lora_down.weight.detach().to(dtype).requires_grad_(True)
        up = lyco.lora_up.weight.detach().to(dtype).requires_grad_(True)
        h2 = NaraHyper(hyper.dim).to(dtype)
        h2.load_state_dict(hyper.state_dict())
        core = ref_hyper(h2, sigma, dtype).index_select(0, seq.seg_joint.long())
        mid = torch.bmm((xr @ a.T).unsqueeze(1), core).squeeze(1)
        y = xr @ w.to(dtype).T + b.to(dtype) + (mid @ up.T) * float(lyco.scale)
        y.backward(g.to(dtype))
        return {
            "x": xr.grad,
            "A": a.grad,
            "B": up.grad,
            "mapper": h2.mapper[0].weight.grad,
        }

    eager_g, ref_g = ref_grads(torch.float32), ref_grads(torch.float64)
    for name in grads:
        assert_no_regression(
            grads[name],
            eager_g[name],
            ref_g[name],
            name=f"nara-grad-{name}",
            test="adapters",
        )
    must_fail(
        lambda: assert_no_regression(
            torch.zeros_like(grads["mapper"]),
            eager_g["mapper"],
            ref_g["mapper"],
            test="adapters",
        ),
        "the hypernetwork's gradient dropped",
    )


def test_missing_context_raises():
    w, b, x = problem()
    aset = adapted(dense_linear(w, b))
    with pytest.raises(RuntimeError, match="noise level"):
        aset.targets[0].linear(x)


def test_step_state_captures_the_noise_level():
    target = SimpleNamespace(
        record=SimpleNamespace(per_token_mask=False, needs_sigma=True, draw_step=None),
        axis="joint",
        draws=Draws(DIM),
    )
    config = SimpleNamespace(
        seed=0, mask_kwargs=lambda: {}, dim=DIM, module_dropout=0.0
    )
    plan = StepPlan([target], config)
    sigma, seq = layout()
    plan.begin(sigma, seq, 1)
    first = plan.capture()
    other = torch.tensor([0.4, 0.7])
    plan.begin(other, seq, 2)
    assert torch.equal(get_context()[0], other)
    plan.install(first)
    assert torch.equal(get_context()[0], sigma)
    plan.end()
    with pytest.raises(RuntimeError):
        get_context()


@pytest.mark.parametrize("fmt", ["lycoris", "raw"])
def test_export_roundtrip(tmp_path, fmt):
    w, b, x = problem()
    sigma, seq = layout()
    aset = adapted(dense_linear(w, b))
    perturb(aset)
    set_context(sigma, seq)
    trained = aset.targets[0].linear(x).detach()

    path = str(tmp_path / "nara.safetensors")
    if fmt == "lycoris":
        aset.export(path, "lycoris", dtype=None)
    else:
        aset.save_state(path)

    twin = dense_linear(w, b)
    stack = AdapterStack(holder(twin), mode="bypass", device="cpu")
    stack.add(path)
    assert stack.needs_sigma
    set_context(sigma, seq)
    loaded = twin(x)
    assert torch.allclose(loaded, trained, atol=2e-5, rtol=2e-5), (
        (loaded - trained).abs().max()
    )
    stack.remove()
    assert not stack.needs_sigma
    plain = F.linear(x, w, b)
    assert torch.equal(twin(x), plain)


def test_comfy_export_is_fixed_sigma_lora():
    w, b, x = problem()
    aset = adapted(dense_linear(w, b))
    perturb(aset)
    lyco = aset.targets[0].lyco
    stem = f"lora_unet_{PATH.replace('.', '_')}"
    per = {
        k[len(stem) + 1 :]: v
        for k, v in aset.tensors("comfy").items()
        if k.startswith(stem + ".")
    }
    core = ref_hyper(aset.shared, torch.tensor([lyco.export_sigma]), torch.float64)[0]
    a = lyco.lora_down.weight.double()
    up = lyco.lora_up.weight.double()
    ref_dw = (up @ core.T @ a) * float(lyco.scale)
    dw = per["lora_up.weight"].double() @ per["lora_down.weight"].double()
    dw = dw * float(per["alpha"]) / DIM
    assert torch.allclose(dw, ref_dw, atol=1e-4), (dw - ref_dw).abs().max()
    plain_lora = (up @ a) * float(lyco.scale)

    def matches_plain():
        assert torch.allclose(dw, plain_lora, atol=1e-4)

    must_fail(matches_plain, "the core vanished from the export")


@pytest.mark.parametrize("wd_on_out", [True, False])
def test_dora_bypass_matches_dynamic_norm(wd_on_out):
    """DoRA normalizes after the live BA*C(sigma): the magnitude ratio is the
    per-sample ||W + dW(sigma)|| in fp32, not the fixed export-sigma one."""
    w, b, x = problem()
    sigma, seq = layout()
    m = 0.7
    aset = adapted(
        dense_linear(w, b),
        dora=True,
        multiplier=m,
        dora_on_output=wd_on_out,
    )
    perturb(aset)
    lyco = aset.targets[0].lyco
    rows = seq.seg_joint.long()
    set_context(sigma, seq)
    xin = x.clone().requires_grad_(True)
    y = aset.targets[0].linear(xin)

    with torch.no_grad():
        cores = ref_hyper(aset.shared, sigma, torch.float64)
        ref = torch.zeros_like(x.double())
        for sample in range(cores.shape[0]):
            sel = rows == sample
            core = cores[sample]
            dw = (
                lyco.lora_up.weight.double() @ core.T @ lyco.lora_down.weight.double()
            ) * float(lyco.scale)
            merged = w.double() + dw
            if wd_on_out:
                s = lyco.dora_scale.double().reshape(-1) / (
                    merged.norm(dim=1) + torch.finfo(torch.float64).eps
                )
                wx = F.linear(x[sel].double(), w.double())
                ref[sel] = (
                    (1 - m) * wx
                    + m * s.reshape(1, -1) * (wx + F.linear(x[sel].double(), dw))
                    + b.double()
                )
            else:
                c = lyco.dora_scale.double().reshape(-1) / (
                    merged.norm(dim=0) + torch.finfo(torch.float64).eps
                )
                scaled = merged * c.reshape(1, -1)
                ref[sel] = F.linear(
                    x[sel].double(), w.double() + m * (scaled - w.double()), b.double()
                )
    assert torch.allclose(y.double(), ref, atol=2e-5, rtol=2e-5), (y - ref).abs().max()

    y.pow(2).sum().backward()
    assert lyco.dora_scale.grad is not None
    assert lyco.dora_scale.grad.abs().sum() > 0
    assert aset.shared.mapper[0].weight.grad.abs().sum() > 0

    def static_magnitude():
        """The fixed export-sigma norm no longer matches the forward."""
        with torch.no_grad():
            core = ref_hyper(
                aset.shared, torch.tensor([lyco.export_sigma]), torch.float64
            )[0]
            dw = (
                lyco.lora_up.weight.double() @ core.T @ lyco.lora_down.weight.double()
            ) * float(lyco.scale)
            merged = w.double() + dw
            dim = 1 if wd_on_out else 0
            s = lyco.dora_scale.double().reshape(-1) / (
                merged.norm(dim=dim) + torch.finfo(torch.float64).eps
            )
            static = torch.zeros_like(x.double())
            for sample in range(cores.shape[0]):
                sel = rows == sample
                cs = cores[sample]
                dws = (
                    lyco.lora_up.weight.double() @ cs.T @ lyco.lora_down.weight.double()
                ) * float(lyco.scale)
                if wd_on_out:
                    wx = F.linear(x[sel].double(), w.double())
                    static[sel] = (
                        (1 - m) * wx
                        + m * s.reshape(1, -1) * (wx + F.linear(x[sel].double(), dws))
                        + b.double()
                    )
                else:
                    scaled = (w.double() + dws) * s.reshape(1, -1)
                    static[sel] = F.linear(
                        x[sel].double(),
                        w.double() + m * (scaled - w.double()),
                        b.double(),
                    )
        assert torch.allclose(y.double(), static, atol=2e-5, rtol=2e-5)

    must_fail(static_magnitude, "the magnitude ignored the live core")


@pytest.mark.parametrize("fmt", ["lycoris", "raw"])
def test_dora_export_roundtrip(tmp_path, fmt):
    w, b, x = problem()
    sigma, seq = layout()
    aset = adapted(dense_linear(w, b), dora=True)
    perturb(aset)
    set_context(sigma, seq)
    trained = aset.targets[0].linear(x).detach()

    path = str(tmp_path / "nara-dora.safetensors")
    if fmt == "lycoris":
        aset.export(path, "lycoris", dtype=None)
    else:
        aset.save_state(path)

    twin = dense_linear(w, b)
    stack = AdapterStack(holder(twin), mode="bypass", device="cpu")
    stack.add(path)
    assert stack.needs_sigma
    set_context(sigma, seq)
    loaded = twin(x)
    assert torch.allclose(loaded, trained, atol=2e-5, rtol=2e-5), (
        (loaded - trained).abs().max()
    )
    stack.remove()


@pytest.mark.parametrize(
    "kw",
    [
        {"mode": "merge"},
        {"rank_dropout": 0.1},
        {"algo_kwargs": {"use_tucker": True}},
        {"algo_kwargs": {"nara_embed": "bogus"}},
    ],
)
def test_config_refuses(kw):
    config = AdapterConfig(algo="nara", dim=DIM, alpha=float(DIM), **kw)
    assert config.validate()

    def valid_flagged():
        assert AdapterConfig(algo="nara", dim=DIM, alpha=float(DIM)).validate()

    must_fail(valid_flagged, "a valid config was flagged")


def test_nara_and_dora_validate_together():
    config = AdapterConfig(algo="nara", dim=DIM, alpha=float(DIM), dora=True)
    assert config.validate() == []
    assert config.record.needs_sigma


def test_state_dict_covers_the_hypernetwork():
    w, b, _ = problem()
    aset = adapted(dense_linear(w, b))
    state = aset.state_dict()
    assert any(k.startswith("nara_hyper.") for k in state)
    perturb(aset)
    changed = aset.state_dict()
    assert not torch.equal(
        state["nara_hyper.mapper.4.weight"], changed["nara_hyper.mapper.4.weight"]
    )
    aset.load_state_dict(state)
    again = aset.state_dict()
    assert torch.equal(
        again["nara_hyper.mapper.4.weight"], state["nara_hyper.mapper.4.weight"]
    )
    names = {n for n, _ in aset.named_trainable()}
    assert "nara_hyper.mapper.0.weight" in names
