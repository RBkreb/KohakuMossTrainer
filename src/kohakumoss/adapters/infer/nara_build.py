"""A noise-conditioned LyCORIS module from a file's per-layer tensors and its shared
hypernetwork: the hypernetwork stem names the kind, the tensor names the variant."""

import torch
from torch import nn

from kohakumoss.adapters import sora
from kohakumoss.adapters.build.network import lora_name_of
from kohakumoss.adapters.build.target import weak_org
from kohakumoss.adapters.nara.module import NaraLohaModule, NaraLokrModule, NaRAModule
from kohakumoss.adapters.train.tlora import axis_of

LOKR_KEYS = frozenset(
    {"lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_w2", "lokr_w2_a", "lokr_w2_b"}
)


def build_nara(
    linear: nn.Module, path: str, parts: dict, hyper, device, sora_tensors: dict | None = None
) -> nn.Module:
    """The variant module of `parts` wired to `hyper`, frozen on `device`."""
    name = lora_name_of("lora_unet", path)
    parts = dict(parts)
    dora_scale = parts.pop("dora_scale", None)
    for t in ("hada_t1", "hada_t2", "lokr_t1", "lokr_t2"):
        if parts.get(t) is not None:
            raise ValueError(f"{path}: NaRA is linear-only, {t} is unsupported")
    if "lora_down.weight" in parts:
        down = parts["lora_down.weight"]
        lyco = NaRAModule(
            name,
            linear,
            1.0,
            down.shape[0],
            float(parts["alpha"]),
            bypass_mode=True,
        )
        with torch.no_grad():
            lyco.lora_down.weight.copy_(down)
            lyco.lora_up.weight.copy_(parts["lora_up.weight"])
    elif "hada_w1_a" in parts:
        lyco = NaraLohaModule.make_module_from_state_dict(
            name,
            linear,
            parts["hada_w1_a"],
            parts["hada_w1_b"],
            parts["hada_w2_a"],
            parts["hada_w2_b"],
            None,
            None,
            float(parts["alpha"]),
            None,
        )
        lyco.bypass_mode = True
    elif parts.keys() & LOKR_KEYS:
        lyco = NaraLokrModule.make_module_from_state_dict(
            name,
            linear,
            parts.get("lokr_w1"),
            parts.get("lokr_w1_a"),
            parts.get("lokr_w1_b"),
            parts.get("lokr_w2"),
            parts.get("lokr_w2_a"),
            parts.get("lokr_w2_b"),
            None,
            None,
            float(parts["alpha"]),
            None,
        )
        lyco.bypass_mode = True
    else:
        raise KeyError(f"{path}: no NaRA variant matches {sorted(parts)}")
    if dora_scale is not None:
        lyco.wd = True
        lyco.wd_on_out = dora_scale.shape[0] == lyco.shape[0]
        lyco.dora_scale = nn.Parameter(dora_scale.float())
    lyco.set_shared(hyper)
    lyco.nara_axis = axis_of(path)
    weak_org(lyco, linear)
    sora.attach_tensors(lyco, sora_tensors or {})
    lyco.to(device=device, dtype=torch.float32)
    lyco.requires_grad_(False)
    lyco.eval()
    return lyco
