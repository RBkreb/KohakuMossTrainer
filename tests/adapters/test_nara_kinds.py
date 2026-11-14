"""The NARA switch on LoHa / LoKr, one linear (no model): each variant's bound forward
and backward match an independent fp64 formula (per-sample cores over a packed layout),
an identity core reduces to the plain algorithm, the native / raw / comfy exports
round-trip or evaluate to the fixed-sigma delta, DoRA composes through the bypass
route, and the configuration refuses what the variants cannot do."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from kohakumoss.adapters import AdapterConfig, AdapterStack, apply
from kohakumoss.adapters.bind.context import Draws
from kohakumoss.adapters.nara import clear_context, get_context, set_context
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
ALGO_KWARGS = {"lora": {}, "loha": {}, "lokr": {"factor": 1}}


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
    return torch.tensor([0.2, 0.9]), seq


def adapted(algo: str, lin: nn.Module, **kw):
    root = holder(lin)
    root.requires_grad_(False)
    config = AdapterConfig(
        algo=algo,
        nara=True,
        preset="attn",
        dim=DIM,
        alpha=float(DIM),
        backend="torch",
        algo_kwargs=dict(ALGO_KWARGS[algo]),
        **kw,
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
    proj = s @ hyper.fourier_w.to(dtype)
    e = torch.cat((proj.sin(), proj.cos()), dim=-1)
    h = e
    for m in hyper.mapper:
        if isinstance(m, nn.Linear):
            h = h @ m.weight.to(dtype).T + m.bias.to(dtype)
        else:
            h = F.silu(h)
    eye = torch.eye(hyper.dim, dtype=dtype)
    return eye + hyper.c_scale * h.view(-1, hyper.dim, hyper.dim)


def loha_delta(w1a, w1b, w2a, w2b, core, scale):
    """dW = (w1a C^T w1b) o (w2a C^T w2b) s."""
    return ((w1a @ core.T @ w1b) * (w2a @ core.T @ w2b)) * scale


def lokr_delta(w1, w1a, w1b, w2a, w2b, core, scale, use_w1):
    """dW = w1 (x) (w2a C^T w2b) s."""
    w2 = w2a @ core.T @ w2b
    return torch.kron(w1 if use_w1 else w1a @ w1b, w2) * scale


def ref_delta(algo: str, lyco, core: torch.Tensor) -> torch.Tensor:
    """dW of one variant at one core, from the module's own factors (upcast to the
    core's dtype)."""
    dt = core.dtype

    def up(t):
        return None if t is None else t.to(dt)

    if algo == "loha":
        return loha_delta(
            up(lyco.hada_w1_a),
            up(lyco.hada_w1_b),
            up(lyco.hada_w2_a),
            up(lyco.hada_w2_b),
            core,
            lyco.scale,
        )
    return lokr_delta(
        up(getattr(lyco, "lokr_w1", None)),
        up(getattr(lyco, "lokr_w1_a", None)),
        up(getattr(lyco, "lokr_w1_b", None)),
        up(lyco.lokr_w2_a),
        up(lyco.lokr_w2_b),
        core,
        lyco.scale,
        lyco.use_w1,
    )


def ref_forward(algo, w, b, x, fac, cores, rows, dtype):
    """base + one delta row-block per sample, in `dtype` (`fac` the factor leaves)."""
    w, b, x = w.to(dtype), b.to(dtype), x.to(dtype)
    out = F.linear(x, w, b)
    for s in range(cores.shape[0]):
        m = rows == s
        core = cores[s].to(dtype)
        if algo == "loha":
            dw = loha_delta(
                fac["hada_w1_a"],
                fac["hada_w1_b"],
                fac["hada_w2_a"],
                fac["hada_w2_b"],
                core,
                fac["scale"],
            )
        else:
            dw = lokr_delta(
                fac.get("lokr_w1"),
                fac.get("lokr_w1_a"),
                fac.get("lokr_w1_b"),
                fac["lokr_w2_a"],
                fac["lokr_w2_b"],
                core,
                fac["scale"],
                fac["use_w1"],
            )
        out[m] = out[m] + F.linear(x[m], dw.to(dtype))
    return out


@pytest.mark.parametrize("algo", ["loha", "lokr"])
def test_forward_and_backward_match_fp64(algo):
    w, b, x = problem()
    sigma, seq = layout()
    aset = adapted(algo, dense_linear(w, b))
    perturb(aset)
    lyco = aset.targets[0].lyco
    hyper = aset.shared
    rows = seq.seg_joint.long()
    set_context(sigma, seq)

    x_act = x.clone().requires_grad_(True)
    out = aset.targets[0].linear(x_act)
    g = torch.randn_like(out)
    out.backward(g)
    factor_grads = {n: p.grad for n, p in lyco.named_parameters()}

    def ref_grads(dtype):
        """The same graph over detached leaves, the hypernetwork rebuilt in `dtype`."""
        xr = x.detach().to(dtype).requires_grad_(True)
        leaves = {
            n: p.detach().to(dtype).requires_grad_(True)
            for n, p in lyco.named_parameters()
        }
        fac = dict(leaves, scale=lyco.scale, use_w1=getattr(lyco, "use_w1", None))
        h2 = NaraHyper(hyper.dim).to(dtype)
        h2.load_state_dict(hyper.state_dict())
        cores = ref_hyper(h2, sigma, dtype)
        y = ref_forward(algo, w, b, xr, fac, cores, rows, dtype)
        y.backward(g.to(dtype))
        return y.detach(), xr.grad, leaves

    out32, gx32, leaves32 = ref_grads(torch.float32)
    out64, gx64, leaves64 = ref_grads(torch.float64)
    assert_no_regression(
        out, out32, out64, name=f"nara-{algo}-forward", test="adapters"
    )
    assert_no_regression(
        x_act.grad, gx32, gx64, name=f"nara-{algo}-grad-x", test="adapters"
    )
    for name, grad in factor_grads.items():
        assert_no_regression(
            grad,
            leaves32[name].grad,
            leaves64[name].grad,
            name=f"nara-{algo}-grad-{name}",
            test="adapters",
        )
    must_fail(
        lambda: assert_no_regression(
            torch.zeros_like(factor_grads[next(iter(factor_grads))]),
            leaves32[next(iter(leaves32))].grad,
            leaves64[next(iter(leaves32))].grad,
            test="adapters",
        ),
        "a factor gradient dropped",
    )


@pytest.mark.parametrize("algo", ["loha", "lokr"])
def test_hypernetwork_grad_flows(algo):
    """The mapper receives gradient through the per-sample cores."""
    w, b, x = problem()
    sigma, seq = layout()
    aset = adapted(algo, dense_linear(w, b))
    perturb(aset)
    set_context(sigma, seq)
    xin = x.clone().requires_grad_(True)
    out = aset.targets[0].linear(xin)
    out.pow(2).sum().backward()
    assert aset.shared.mapper[0].weight.grad is not None
    assert aset.shared.mapper[0].weight.grad.abs().sum() > 0
    for _, p in aset.targets[0].lyco.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert torch.isfinite(xin.grad).all()


@pytest.mark.parametrize("algo", ["loha", "lokr"])
def test_identity_core_matches_plain_algo(algo):
    """The hypernetwork starts as the identity: the forward is the plain algorithm's."""
    w, b, x = problem()
    aset = adapted(algo, dense_linear(w, b))
    g = torch.Generator().manual_seed(5)
    lyco = aset.targets[0].lyco
    with torch.no_grad():
        for _, p in lyco.named_parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.3)
    sigma, seq = layout()
    set_context(sigma, seq)
    y = aset.targets[0].linear(x)
    with torch.no_grad():
        dw = ref_delta(algo, lyco, torch.eye(DIM))
    ref = F.linear(x, w, b) + x @ dw.T
    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), (y - ref).abs().max()


@pytest.mark.parametrize("algo", ["loha", "lokr"])
def test_comfy_export_is_fixed_sigma(algo):
    """The comfy tensors evaluate (ComfyUI rules) to the fixed-sigma dW, not the
    unfolded plain delta."""
    w, b, x = problem()
    aset = adapted(algo, dense_linear(w, b))
    perturb(aset)
    lyco = aset.targets[0].lyco
    stem = f"lora_unet_{PATH.replace('.', '_')}"
    per = {
        k[len(stem) + 1 :]: v
        for k, v in aset.tensors("comfy").items()
        if k.startswith(stem + ".")
    }
    core = ref_hyper(aset.shared, torch.tensor([lyco.export_sigma]), torch.float64)[0]
    scale = float(per["alpha"]) / DIM
    if algo == "loha":
        dw = (
            per["hada_w1_a"].double()
            @ per["hada_w1_b"].double()
            * (per["hada_w2_a"].double() @ per["hada_w2_b"].double())
        )
    else:
        w1 = (
            per["lokr_w1"].double()
            if "lokr_w1" in per
            else per["lokr_w1_a"].double() @ per["lokr_w1_b"].double()
        )
        w2 = per["lokr_w2_a"].double() @ per["lokr_w2_b"].double()
        dw = torch.kron(w1, w2)
    dw = dw * scale
    ref = ref_delta(algo, lyco, core).double() * float(lyco.scalar)
    # fp32 core / factor noise relative to the delta's own scale (cancelling entries
    # of the Hadamard product carry a larger absolute error than their magnitude).
    assert (dw - ref).abs().max() / ref.abs().max() < 1e-5, (dw - ref).abs().max()

    plain = ref_delta(algo, lyco, torch.eye(DIM, dtype=torch.float64)).double()
    plain = plain * float(lyco.scalar)

    def matches_plain():
        assert torch.allclose(dw, plain, atol=1e-4)

    must_fail(matches_plain, "the core vanished from the export")


@pytest.mark.parametrize("algo", ["loha", "lokr"])
@pytest.mark.parametrize("fmt", ["lycoris", "raw"])
def test_export_roundtrip(algo, fmt, tmp_path):
    w, b, x = problem()
    sigma, seq = layout()
    aset = adapted(algo, dense_linear(w, b))
    perturb(aset)
    set_context(sigma, seq)
    trained = aset.targets[0].linear(x).detach()

    path = str(tmp_path / f"nara-{algo}.safetensors")
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
    assert torch.allclose(loaded, trained, atol=2e-4, rtol=2e-4), (
        (loaded - trained).abs().max()
    )
    stack.remove()
    assert not stack.needs_sigma
    plain = F.linear(x, w, b)
    assert torch.equal(twin(x), plain)


@pytest.mark.parametrize("algo", ["loha", "lokr"])
def test_dora_nara_bypass_matches_formula(algo):
    """DoRA on a noise-conditioned delta: y = b + (1-m) Wx + m s (Wx + dW(sigma) x),
    s the magnitude ratio of the fixed export-sigma delta."""
    w, b, x = problem()
    sigma, seq = layout()
    m = 0.7
    aset = adapted(algo, dense_linear(w, b), dora=True, multiplier=m)
    perturb(aset)
    lyco = aset.targets[0].lyco
    rows = seq.seg_joint.long()
    set_context(sigma, seq)
    y = aset.targets[0].linear(x)

    with torch.no_grad():
        dw_fixed = ref_delta(
            algo,
            lyco,
            ref_hyper(aset.shared, torch.tensor([lyco.export_sigma]), torch.float64)[0],
        ).double()
        s = lyco.dora_scale.double().reshape(-1) / (
            (w.double() + dw_fixed).norm(dim=1) + torch.finfo(torch.float64).eps
        )
        wx = F.linear(x.double(), w.double())
        bias = b.double().reshape(1, -1)
        ref = wx + bias
        cores = ref_hyper(aset.shared, sigma, torch.float64)
        for sample in range(cores.shape[0]):
            sel = rows == sample
            dw = ref_delta(algo, lyco, cores[sample]).double()
            ref[sel] = (
                (1 - m) * wx[sel]
                + m * (s.reshape(1, -1) * (wx[sel] + F.linear(x[sel].double(), dw)))
                + bias
            )
    assert torch.allclose(y.double(), ref, atol=1e-3, rtol=1e-3), (y - ref).abs().max()


def test_lokr_rejects_a_full_w2():
    """A layout whose LoKr w2 came out full has no rank for the core."""
    w, b, _ = problem()
    lin = dense_linear(w, b)
    root = holder(lin)
    root.requires_grad_(False)
    config = AdapterConfig(
        algo="lokr", nara=True, preset="attn", dim=10000, alpha=1.0, backend="torch"
    )
    with pytest.raises(ValueError, match="factorized LoKr w2"):
        apply(root, config)


@pytest.mark.parametrize(
    "algo, kw",
    [
        ("loha", {"mode": "merge"}),
        ("loha", {"rank_dropout": 0.1}),
        ("loha", {"algo_kwargs": {"use_tucker": True}}),
        ("loha", {"algo_kwargs": {"nara_embed": "bogus"}}),
        ("lokr", {"mode": "merge"}),
        ("lora", {"mode": "merge"}),
        ("tlora", {}),
        ("oft", {}),
        ("full", {}),
    ],
)
def test_config_refuses(algo, kw):
    merged = dict(ALGO_KWARGS.get(algo, {}))
    merged.update(kw.pop("algo_kwargs", {}))
    config = AdapterConfig(
        algo=algo, nara=True, dim=DIM, alpha=float(DIM), algo_kwargs=merged, **kw
    )
    assert config.validate()

    if algo in ALGO_KWARGS:

        def valid_flagged():
            assert AdapterConfig(algo=algo, dim=DIM, alpha=float(DIM)).validate()

        must_fail(valid_flagged, "a valid config was flagged")


@pytest.mark.parametrize("algo", ["lora", "loha", "lokr"])
def test_nara_and_dora_validate_together(algo):
    """The switch and DoRA are orthogonal on the kinds that carry both."""
    config = AdapterConfig(
        algo=algo,
        nara=True,
        dora=True,
        dim=DIM,
        alpha=float(DIM),
        algo_kwargs=dict(ALGO_KWARGS[algo]),
    )
    assert config.validate() == [], (algo, config.validate())
    assert config.record.needs_sigma


def test_state_dict_covers_the_hypernetwork():
    w, b, _ = problem()
    aset = adapted("loha", dense_linear(w, b))
    state = aset.state_dict()
    assert any(k.startswith("nara_hyper.") for k in state)
    perturb(aset)
    changed = aset.state_dict()
    assert not torch.equal(
        state["nara_hyper.mapper.4.weight"], changed["nara_hyper.mapper.4.weight"]
    )
    aset.load_state_dict(state)
    assert torch.equal(
        aset.state_dict()["nara_hyper.mapper.4.weight"],
        state["nara_hyper.mapper.4.weight"],
    )


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
