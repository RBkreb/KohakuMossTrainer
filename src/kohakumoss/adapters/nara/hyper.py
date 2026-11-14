"""NaRA's shared hypernetwork: a noise-level embedding and an MLP mapper producing
the core C(sigma) = I + c_scale * F(e_sigma), one [B, r, r] batch for every layer."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

EMBEDS = ("fourier", "mlp", "raw")


class NaraHyper(nn.Module):
    """sigma [B] -> C [B, dim, dim] in fp32. `embed`: fourier (fixed random frequencies),
    mlp (learned) or raw (the scalar itself). The mapper's last layer starts at zero,
    so C is the identity at init and NaRA reduces to LoRA."""

    def __init__(
        self,
        dim: int,
        hidden1: int = 256,
        hidden2: int = 512,
        embed_dim: int = 64,
        embed: str = "fourier",
        fourier_scale: float = 16.0,
        c_scale: float = 1.0,
    ):
        super().__init__()
        if embed not in EMBEDS:
            raise ValueError(f"nara_embed={embed!r} not in {EMBEDS}")
        if embed == "fourier" and embed_dim % 2:
            raise ValueError(
                f"nara_embed_dim must be even for fourier, got {embed_dim}"
            )
        if dim <= 0 or hidden1 <= 0 or hidden2 <= 0:
            raise ValueError("nara dim / hidden sizes must be positive")
        self.dim = dim
        self.c_scale = float(c_scale)
        self.embed = embed
        if embed == "fourier":
            self.register_buffer(
                "fourier_w", torch.randn(1, embed_dim // 2) * fourier_scale
            )
            input_dim = embed_dim
        elif embed == "mlp":
            self.embed_proj = nn.Linear(1, embed_dim)
            input_dim = embed_dim
        else:
            input_dim = 1
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, dim * dim),
        )
        self.register_buffer("eye", torch.eye(dim), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming hidden layers, zero output layer (C starts as the identity)."""
        layers = [m for m in self.mapper if isinstance(m, nn.Linear)]
        for m in layers[:-1]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.zeros_(m.bias)
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if self.embed == "mlp":
            nn.init.kaiming_uniform_(self.embed_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.embed_proj.bias)

    def forward(self, sigma: Tensor) -> Tensor:
        x = sigma.float().reshape(-1, 1)
        match self.embed:
            case "fourier":
                proj = x @ self.fourier_w.to(x.dtype)
                x = torch.cat((proj.sin(), proj.cos()), dim=-1)
            case "mlp":
                x = F.silu(self.embed_proj(x))
        flat = self.mapper(x)
        return self.eye + self.c_scale * flat.view(-1, self.dim, self.dim)
