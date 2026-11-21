"""Records of NaRA: noise-aware adaptation whose core C(sigma) comes from one shared
hypernetwork (see docs/internals/adapter-nara.md). `NARA` is the LoRA-shaped kind
(ALGO='nara'); the NARA switch derives one variant record per nara-capable kind
(`NARA_LOHA`, `NARA_LOKR`) that keeps the base kind's applier surface and swaps the
module class and the shared hypernetwork in."""

from dataclasses import replace
from functools import partial

import torch
from lycoris.wrapper import network_module_dict

from kohakumoss.adapters.kinds.common import names, unit_diff
from kohakumoss.adapters.kinds.lowrank import LOCON, LOHA, LOKR
from kohakumoss.adapters.kinds.record import Kind, Stacking, register
from kohakumoss.adapters.nara.hyper import NaraHyper
from kohakumoss.adapters.nara.module import NaraLohaModule, NaraLokrModule, NaRAModule
from kohakumoss_common.calls import nothing


def _shared(config) -> NaraHyper:
    """The set's one hypernetwork, sized by DIM and the nara_* knobs."""
    kwargs = {k[5:]: v for k, v in config.algo_kwargs.items() if k.startswith("nara_")}
    return NaraHyper(config.dim, **kwargs)


def _nara_as_lora(t) -> dict:
    """NaRA at its export_sigma as one plain LoRA: up = B C^T s, down = A."""
    lyco = t.lyco
    core = lyco.core_at(lyco.export_sigma)
    up = (lyco.lora_up.weight.float() @ core.T) * lyco.scale
    out = {
        "lora_up.weight": up.detach().float(),
        "lora_down.weight": lyco.lora_down.weight.detach().float(),
        "alpha": torch.tensor(float(lyco.lora_dim)),
    }
    if getattr(lyco, "wd", False) and getattr(lyco, "dora_scale", None) is not None:
        out["dora_scale"] = lyco.dora_scale.detach().float()
    return out


def _comfy_loha(t) -> dict:
    """The hada factors with the core folded into both up factors at export_sigma."""
    out = {k: v.detach() for k, v in t.lyco.state_dict().items()}
    core = t.lyco.core_at(t.lyco.export_sigma)
    out["hada_w1_a"] = out["hada_w1_a"].float() @ core.T
    out["hada_w2_a"] = out["hada_w2_a"].float() @ core.T
    return out


def _comfy_lokr(t) -> dict:
    """The lokr factors with the core folded into w2_a at export_sigma."""
    out = {k: v.detach() for k, v in t.lyco.state_dict().items()}
    core = t.lyco.core_at(t.lyco.export_sigma)
    out["lokr_w2_a"] = out["lokr_w2_a"].float() @ core.T
    return out


def _nara_kind(base: Kind, module: type, name: str, comfy) -> Kind:
    """The noise-conditioned variant of one nara-capable kind: the base's applier
    surface, no kernel ops (the core lives outside the factors the kernels read), the
    shared hypernetwork and the fixed-sigma comfy conversion."""
    return replace(
        base,
        name=name,
        module=module,
        algos={},
        kernel_diff_weight=nothing,
        kernel_merged_weight=nothing,
        kernel_bypass=nothing,
        kernel_merge=None,
        comfy_tensors=comfy,
        needs_sigma=True,
        shared_module=_shared,
        supports_nara=True,
    )


NARA = register(
    Kind(
        name="nara",
        module=NaRAModule,
        algos={"nara": "nara"},
        diff_weight=unit_diff,
        comfy_tensors=_nara_as_lora,
        uniform_scale_tensors=partial(names, ("lora_up.weight",)),
        row_scale_tensors=partial(names, ("lora_up.weight",)),
        stacking=Stacking.ADDITIVE,
        loraplus_tensors=("lora_up.weight",),
        needs_sigma=True,
        shared_module=_shared,
        supports_dora=True,
        supports_nara=True,
        supports_sora=True,
        supports_eva=True,
    )
)
NARA_LOHA = register(_nara_kind(LOHA, NaraLohaModule, "nara-loha", _comfy_loha))
NARA_LOKR = register(_nara_kind(LOKR, NaraLokrModule, "nara-lokr", _comfy_lokr))

network_module_dict["nara"] = NaRAModule
network_module_dict["nara-loha"] = NaraLohaModule
network_module_dict["nara-lokr"] = NaraLokrModule

# Record name -> the record the NARA switch trains (the lora / locon shape is the
# existing ALGO='nara').
NARA_VARIANTS: dict[str, Kind] = {
    LOCON.name: NARA,
    LOHA.name: NARA_LOHA,
    LOKR.name: NARA_LOKR,
    NARA.name: NARA,
}

NARA_NETWORK = {
    "lora": "nara",
    "locon": "nara",
    "nara": "nara",
    "loha": "nara-loha",
    "lokr": "nara-lokr",
}


def nara_record_of(rec: Kind) -> Kind | None:
    """The record the NARA switch trains for `rec` (None when it has no variant)."""
    return NARA_VARIANTS.get(rec.name)


def nara_network_algo(algo: str) -> str:
    """The LyCORIS network_module name of `algo` under the NARA switch."""
    try:
        return NARA_NETWORK[algo]
    except KeyError:
        raise KeyError(f"NARA does not apply to ALGO={algo!r}") from None
