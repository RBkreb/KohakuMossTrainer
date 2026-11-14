"""Noise-conditioned LyCORIS modules: NaRAModule (LoRA's A/B) and the NARA-switch
variants of LoHa / LoKr, each holding the shared hypernetwork's core C(sigma) between
its rank factors (see docs/internals/adapter-nara.md)."""

import torch
import torch.nn.functional as F
from lycoris.functional.loha import diff_weight as loha_diff_weight
from lycoris.functional.lokr import make_kron
from lycoris.modules import LoConModule, LohaModule, LokrModule
from torch import Tensor

from kohakumoss.adapters.nara.context import get_context
from kohakumoss.adapters.nara.hyper import NaraHyper
from kohakumoss.adapters.train.tlora import rows_of


class NaraMixin:
    """The shared-hypernetwork wiring of a noise-conditioned module: the hyper is held
    outside the module tree (the set owns it) and the core is evaluated per forward on
    the step's noise level."""

    nara_axis = "joint"

    def set_shared(self, hyper: NaraHyper) -> None:
        """Wire the set's shared hypernetwork (not a submodule: the set owns it)."""
        object.__setattr__(self, "_nara_shared", hyper)

    @property
    def shared(self) -> NaraHyper:
        hyper = getattr(self, "_nara_shared", None)
        if hyper is None:
            raise RuntimeError(f"{self.lora_name}: no shared NaRA hypernetwork wired")
        return hyper

    def core_at(self, sigma: float) -> Tensor:
        """C at one fixed noise level (export / merge), fp32 on the hyper's device."""
        hyper = self.shared
        device = next(hyper.parameters()).device
        s = torch.tensor([float(sigma)], device=device, dtype=torch.float32)
        return hyper(s)[0]

    def sample_cores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(one core per sample in `x`'s dtype, the token -> sample rows) of the
        step's noise level, through the layer's axis."""
        sigma, seq = get_context()
        rows = rows_of(seq, self.nara_axis, sigma.shape[0], sigma.device)
        return self.shared(sigma).to(x.dtype), rows

    def dora_ratio(self, w: Tensor, detach: bool) -> tuple[Tensor, Tensor]:
        """(mag / ||W + dW(sigma_s)|| per sample over the DoRA axis, fp32
        [B, rows]; the token -> sample rows). Recomputed every forward from the
        live cores: no static-BA norm cache or fused kernel applies."""
        sigma, seq = get_context()
        rows = rows_of(seq, self.nara_axis, sigma.shape[0], sigma.device)
        cores = self.shared(sigma)
        norms = self.dora_norms(w.to(cores.device), cores)
        norms = norms + torch.finfo(torch.float32).eps
        if detach:
            norms = norms.detach()
        return self.dora_scale.float().reshape(-1).to(norms.device) / norms, rows

    def dora_norms(self, w: Tensor, cores: Tensor) -> Tensor:
        """||W + dW(sigma_s)|| over the DoRA axis, one row per sample, fp32."""
        raise NotImplementedError

    def token_cores(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(one core per token row in `x`'s dtype, the token -> sample rows): the
        step's per-sample cores gathered through the layer's axis."""
        cores, rows = self.sample_cores(x)
        return cores.index_select(0, rows.long()), rows


class NaRAModule(NaraMixin, LoConModule):
    """One adapted linear's A (lora_down) / B (lora_up); the core C(sigma) comes from
    the shared NaraHyper wired by `set_shared`. `export_sigma` is the fixed noise level
    of dW exports."""

    name = "nara"
    support_module = {"linear"}
    weight_list = ["lora_up.weight", "lora_down.weight", "alpha"]
    weight_list_det = ["lora_up.weight"]

    def __init__(
        self,
        lora_name,
        org_module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        bypass_mode=None,
        nara_hidden1: int = 256,
        nara_hidden2: int = 512,
        nara_embed_dim: int = 64,
        nara_embed: str = "fourier",
        nara_fourier_scale: float = 16.0,
        nara_c_scale: float = 1.0,
        export_sigma: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            use_tucker=False,
            bypass_mode=bypass_mode,
            **kwargs,
        )
        self.export_sigma = float(export_sigma)

    def make_weight(self, device=None):
        """dW at the fixed export_sigma: scale belongs to the caller (LyCORIS rule)."""
        core = self.core_at(self.export_sigma)
        weight = self.lora_up.weight.float() @ core.T @ self.lora_down.weight.float()
        return weight.to(device) * self.scalar.to(device)

    def dora_norms(self, w: Tensor, cores: Tensor) -> Tensor:
        """Gram expansion of ||W + B C^T A||^2 = ||W||^2 + 2<W, dW> + ||dW||^2:
        only [B, out, r] intermediates, no per-sample dense delta."""
        up = self.lora_up.weight.float() * (self.scale * self.scalar.float())
        down = self.lora_down.weight.float()
        if self.wd_on_out:
            base = (w * w).sum(dim=1)
            cross = (torch.einsum("or,srk->sok", w @ down.T, cores) * up).sum(dim=-1)
            gram = torch.einsum("sji,jk,skl->sil", cores, down @ down.T, cores)
            delta = torch.einsum("ok,skl,ol->so", up, gram, up)
        else:
            base = (w * w).sum(dim=0)
            cross = torch.einsum("ia,sba,bi->si", w.T @ up, cores, down)
            gram = torch.einsum("sia,ab,slb->sil", cores, up.T @ up, cores)
            delta = torch.einsum("ki,skl,li->si", down, gram, down)
        return (base + 2 * cross + delta).clamp_min(0).sqrt()

    def bypass_forward_diff(self, x, scale=1):
        cores, _ = self.token_cores(x)
        mid = F.linear(x, self.lora_down.weight.to(x.dtype))
        mid = torch.bmm(mid.unsqueeze(1), cores).squeeze(1)
        y = F.linear(mid, self.lora_up.weight.to(x.dtype))
        scalar = self.scalar.to(device=y.device, dtype=y.dtype)
        return self.dropout(y * (self.scale * scale) * scalar)


class NaraLohaModule(NaraMixin, LohaModule):
    """LoHa with the shared core in both rank channels: at each sample's noise level
    dW = (w1a C^T w1b) o (w2a C^T w2b); the Hadamard product has no rank-factorized
    application, so the bypass rebuilds one dense delta per sample."""

    name = "nara-loha"
    support_module = {"linear"}

    def __init__(
        self,
        lora_name,
        org_module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_out=True,
        bypass_mode=None,
        rs_lora=False,
        nara_hidden1: int = 256,
        nara_hidden2: int = 512,
        nara_embed_dim: int = 64,
        nara_embed: str = "fourier",
        nara_fourier_scale: float = 16.0,
        nara_c_scale: float = 1.0,
        export_sigma: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            use_tucker=False,
            use_scalar=use_scalar,
            rank_dropout_scale=rank_dropout_scale,
            weight_decompose=weight_decompose,
            wd_on_out=wd_on_out,
            bypass_mode=bypass_mode,
            rs_lora=rs_lora,
        )
        self.export_sigma = float(export_sigma)

    def get_weight(self, shape):
        """dW at the fixed export_sigma: both up factors read the core's transpose."""
        core = self.core_at(self.export_sigma)
        gamma = torch.tensor(self.scale, dtype=torch.float32, device=core.device)
        w1u = self.hada_w1_a.float() @ core.T
        w2u = self.hada_w2_a.float() @ core.T
        weight = loha_diff_weight(
            self.hada_w1_b.float(),
            w1u,
            self.hada_w2_b.float(),
            w2u,
            None,
            None,
            gamma=gamma,
        )
        return weight.reshape(shape) if shape is not None else weight

    def dora_norms(self, w: Tensor, cores: Tensor) -> Tensor:
        """One dense merged weight per sample (the Hadamard delta has no rank
        factorization), the same cost profile as the bypass."""
        gamma = self.scale * self.scalar.float()
        w1a, w1b = self.hada_w1_a.float(), self.hada_w1_b.float()
        w2a, w2b = self.hada_w2_a.float(), self.hada_w2_b.float()
        dim = 1 if self.wd_on_out else 0
        norms = []
        for core in cores:
            p = w1a @ core.T @ w1b
            q = w2a @ core.T @ w2b
            norms.append((w + (p * q) * gamma).norm(dim=dim))
        return torch.stack(norms)

    def bypass_forward_diff(self, x, scale=1):
        cores, rows = self.sample_cores(x)
        scalar = self.scalar.to(device=x.device, dtype=x.dtype)
        gamma = scalar * (self.scale * scale)
        w1a = self.hada_w1_a.to(x.dtype)
        w1b = self.hada_w1_b.to(x.dtype)
        w2a = self.hada_w2_a.to(x.dtype)
        w2b = self.hada_w2_b.to(x.dtype)
        y = x.new_zeros(x.shape[:-1] + (self.shape[0],))
        for s in range(cores.shape[0]):
            idx = (rows == s).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            core = cores[s]
            p = w1a @ core.T @ w1b
            q = w2a @ core.T @ w2b
            y.index_put_((idx,), F.linear(x[idx], (p * q) * gamma))
        return y


class NaraLokrModule(NaraMixin, LokrModule):
    """LoKr with the shared core in the factorized w2's rank contraction:
    dW = w1 (x) (w2a C^T w2b); the bypass chains the per-token core between the w2
    factors, then w1 across the groups. A layout whose w2 is full has no rank for the
    core and is rejected."""

    name = "nara-lokr"
    support_module = {"linear"}

    def __init__(
        self,
        lora_name,
        org_module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,
        use_scalar=False,
        decompose_both=False,
        factor: int = -1,
        rank_dropout_scale=False,
        weight_decompose=False,
        wd_on_out=True,
        full_matrix=False,
        bypass_mode=None,
        rs_lora=False,
        unbalanced_factorization=False,
        nara_hidden1: int = 256,
        nara_hidden2: int = 512,
        nara_embed_dim: int = 64,
        nara_embed: str = "fourier",
        nara_fourier_scale: float = 16.0,
        nara_c_scale: float = 1.0,
        export_sigma: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            use_tucker=False,
            use_scalar=use_scalar,
            decompose_both=decompose_both,
            factor=factor,
            rank_dropout_scale=rank_dropout_scale,
            weight_decompose=weight_decompose,
            wd_on_out=wd_on_out,
            full_matrix=full_matrix,
            bypass_mode=bypass_mode,
            rs_lora=rs_lora,
            unbalanced_factorization=unbalanced_factorization,
        )
        if self.use_w2:
            raise ValueError(
                f"{lora_name}: NaRA needs a factorized LoKr w2 for the core; this "
                "layout made w2 full (lower DIM, raise ALGO_KWARGS['factor'], see "
                "configs/moss/algo/algo_lokr_nara.py)"
            )
        self.export_sigma = float(export_sigma)

    def get_weight(self, shape):
        """dW at the fixed export_sigma: the core sits in the w2 rebuild."""
        core = self.core_at(self.export_sigma)
        w2 = self.lokr_w2_a.float() @ core.T @ self.lokr_w2_b.float()
        w1 = (
            self.lokr_w1.float()
            if self.use_w1
            else self.lokr_w1_a.float() @ self.lokr_w1_b.float()
        )
        gamma = torch.tensor(self.scale, dtype=torch.float32, device=core.device)
        weight = make_kron(w1, w2, gamma)
        return weight.reshape(shape) if shape is not None else weight

    def dora_norms(self, w: Tensor, cores: Tensor) -> Tensor:
        """Kronecker structure: the cross term reads W as [m1, m2, n1, n2] blocks,
        and ||dW||^2 factorizes over the kron axes (no dense delta)."""
        gamma = self.scale * self.scalar.float()
        w1 = (
            self.lokr_w1.float()
            if self.use_w1
            else self.lokr_w1_a.float() @ self.lokr_w1_b.float()
        )
        w2 = (
            torch.einsum(
                "or,skr,ki->soi", self.lokr_w2_a.float(), cores, self.lokr_w2_b.float()
            )
            * gamma
        )
        m1, n1 = w1.shape
        m2, n2 = w2.shape[1:]
        w4 = w.reshape(m1, m2, n1, n2)
        batch = cores.shape[0]
        if self.wd_on_out:
            base = (w * w).sum(dim=1)
            cross = torch.einsum("ikjl,ij,skl->sik", w4, w1, w2).reshape(batch, -1)
            delta = torch.einsum(
                "i,sk->sik", (w1 * w1).sum(dim=1), (w2 * w2).sum(dim=2)
            ).reshape(batch, -1)
        else:
            base = (w * w).sum(dim=0)
            cross = torch.einsum("ikjl,ij,skl->sjl", w4, w1, w2).reshape(batch, -1)
            delta = torch.einsum(
                "j,sl->sjl", (w1 * w1).sum(dim=0), (w2 * w2).sum(dim=1)
            ).reshape(batch, -1)
        return (base + 2 * cross + delta).clamp_min(0).sqrt()

    def bypass_forward_diff(self, x, scale=1):
        cores, _ = self.token_cores(x)
        scalar = self.scalar.to(device=x.device, dtype=x.dtype)
        c = (
            self.lokr_w1.to(x.dtype)
            if self.use_w1
            else self.lokr_w1_a.to(x.dtype) @ self.lokr_w1_b.to(x.dtype)
        )
        uq = c.size(1)
        h_in_group = x.reshape(*x.shape[:-1], uq, -1)
        mid = F.linear(h_in_group, self.lokr_w2_b.to(x.dtype))
        mid = torch.bmm(mid, cores)
        hb = F.linear(mid, self.lokr_w2_a.to(x.dtype))
        hc = F.linear(hb.transpose(-1, -2), c)
        h = hc.transpose(-1, -2).reshape(*hc.shape[:-2], -1)
        return h * (self.scale * scale) * scalar
