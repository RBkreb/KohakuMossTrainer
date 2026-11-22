"""EVA core (no model): the incremental SVD hook recovers a planted input subspace,
the LoRA / LoKr factor inits land in it with zero output factors, and the
configuration refuses what EVA cannot initialise."""

import pytest
import torch
from torch import nn

from kohakumoss.adapters import AdapterConfig
from kohakumoss.adapters.eva import SVDHook, _init_lokr, _init_lora

pytestmark = pytest.mark.cpu


def test_svd_hook_recovers_planted_subspace():
    torch.manual_seed(11)
    d, r = 64, 4
    basis = torch.linalg.qr(torch.randn(d, r))[0]
    hook = SVDHook(r, 0.99)
    lin = nn.Linear(d, d)
    for _ in range(40):
        coeffs = torch.randn(512, r) * torch.tensor([8.0, 4.0, 2.0, 1.0])
        x = coeffs @ basis.T + 0.01 * torch.randn(512, d)
        hook(lin, (x,))
    assert bool(hook.converged.all())
    s = torch.linalg.svdvals(hook.svd.components_.float() @ basis)
    assert (s - 1).abs().max().item() < 1e-3


def test_lora_init_components_and_zero_up():
    torch.manual_seed(11)
    d, r = 64, 4
    basis = torch.linalg.qr(torch.randn(d, r))[0]

    class Lyco:
        lora_dim = r
        lora_down = nn.Linear(d, r, bias=False)
        lora_up = nn.Linear(r, d, bias=False)

    lyco = Lyco()
    with torch.no_grad():
        _init_lora(lyco, basis.T.contiguous())
    assert torch.equal(lyco.lora_down.weight, basis.T)
    assert torch.count_nonzero(lyco.lora_up.weight) == 0


def test_lokr_init_subspace_and_zero_up():
    torch.manual_seed(11)
    m1, n1, n2, r2 = 2, 8, 8, 3

    class Lyco:
        use_w1 = True
        use_w2 = False
        lora_dim = r2
        lokr_w1 = nn.Parameter(torch.randn(m1, n1))
        lokr_w2_a = nn.Parameter(torch.randn(n2, r2))
        lokr_w2_b = nn.Parameter(torch.zeros(r2, n2))

    lyco = Lyco()
    sub = torch.linalg.qr(torch.randn(n2, r2))[0].T
    rows = []
    for i in range(m1):
        for j in range(n1):
            v = torch.zeros(n1 * n2)
            v[j * n2 : (j + 1) * n2] = sub[j % r2]
            rows.append(v)
    with torch.no_grad():
        _init_lokr(lyco, torch.stack(rows)[: m1 * r2])
    w = lyco.lokr_w2_b
    assert torch.count_nonzero(lyco.lokr_w2_a) == 0
    assert (w @ w.T - torch.eye(r2)).abs().max().item() < 1e-4
    assert (w @ sub.T @ sub - w).abs().max().item() < 1e-4


def test_lokr_init_reports_a_comp_too_small():
    """A w2_b init that needs more component rows than available returns the reason
    instead of silently leaving the random init."""
    m1, n1, n2, r2 = 2, 8, 8, 3

    class Lyco:
        use_w1 = True
        use_w2 = False
        lora_dim = r2
        lokr_w1 = nn.Parameter(torch.randn(m1, n1))
        lokr_w2_a = nn.Parameter(torch.randn(n2, r2))
        lokr_w2_b = nn.Parameter(torch.zeros(r2, n2))

    lyco = Lyco()
    before = lyco.lokr_w2_b.clone()
    why = _init_lokr(lyco, torch.randn(m1 * r2 - 1, n1 * n2))
    assert why is not None
    assert "component rows" in why
    assert torch.equal(lyco.lokr_w2_b, before), "a refused init still wrote w2_b"


def test_eva_refuses_incapable_kind_and_dropout():
    cfg = AdapterConfig(algo="loha", preset="attn", dim=4, eva={"batches": 4})
    assert any("EVA does not apply" in p for p in cfg.validate())
    cfg = AdapterConfig(algo="lora", preset="attn", dim=4, eva={"rho": 2})
    assert any("unknown keys" in p for p in cfg.validate())
    cfg = AdapterConfig(algo="lora", preset="attn", dim=4, eva={}, module_dropout=0.1)
    assert any("activations" in p for p in cfg.validate())
    assert AdapterConfig(algo="lora", preset="attn", dim=4).validate() == []


def test_eva_precision_switch():
    """`precision` picks the dtype of the incremental SVD: fp64 as the package ships,
    fp32 the fast path. The fp32 run must reach the same planted subspace."""
    for value in ("fp64", "fp32"):
        cfg = AdapterConfig(algo="lora", preset="attn", dim=4, eva={"precision": value})
        assert cfg.validate() == []
    bad = AdapterConfig(algo="lora", preset="attn", dim=4, eva={"precision": "fp16"})
    assert any("not in ('fp64', 'fp32')" in p for p in bad.validate())

    torch.manual_seed(11)
    d, r = 64, 4
    basis = torch.linalg.qr(torch.randn(d, r))[0]
    batches = []
    for _ in range(12):
        coeffs = torch.randn(256, r) * torch.tensor([8.0, 4.0, 2.0, 1.0])
        batches.append(coeffs @ basis.T + 0.01 * torch.randn(256, d))

    subspaces = {}
    for prec in ("fp64", "fp32"):
        hook = SVDHook(r, 0.99, prec)
        for x in batches:
            hook(torch.nn.Linear(d, d), (x,))
        subspaces[prec] = hook.svd.components_.float()
        assert bool(hook.converged.all())
    # both find the planted subspace; compare column-wise after matching the sign
    for prec, comp in subspaces.items():
        s = torch.linalg.svdvals(comp @ basis)
        assert (s - 1).abs().max().item() < 1e-2, prec

    with pytest.raises(ValueError, match="not in"):
        SVDHook(r, 0.99, "fp16")
