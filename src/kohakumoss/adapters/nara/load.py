"""Read a NaRA file's shared-hypernetwork tensors (the `nara_hyper` stem) and rebuild
the hypernetwork from them and the configuration the file records."""

import json

import torch

from kohakumoss.adapters.build.target import SHARED
from kohakumoss.adapters.nara.hyper import NaraHyper


def split_tensors(tensors: dict) -> tuple[dict, dict]:
    """(hypernetwork tensors under their own names, the remaining per-layer tensors)."""
    prefix = f"{SHARED}."
    hyper = {k[len(prefix) :]: v for k, v in tensors.items() if k.startswith(prefix)}
    rest = {k: v for k, v in tensors.items() if not k.startswith(prefix)}
    return hyper, rest


def hyper_from(tensors: dict, meta: dict, device) -> NaraHyper:
    """The hypernetwork of a file's tensors: its structure from the recorded spec
    (a training state) or the network args (an export), fp32 and frozen on `device`."""
    spec = meta.get("kohakumoss_spec")
    if spec:
        fields = json.loads(spec)
        dim, kwargs = int(fields["dim"]), fields.get("algo_kwargs", {})
    elif "ss_network_args" in meta:
        dim = int(meta["ss_network_dim"])
        kwargs = json.loads(meta["ss_network_args"])
    else:
        raise ValueError(
            f"{SHARED} tensors without a recorded configuration "
            "(kohakumoss_spec or ss_network_args)"
        )
    kwargs = {k[5:]: v for k, v in kwargs.items() if k.startswith("nara_")}
    hyper = NaraHyper(dim, **kwargs)
    hyper.load_state_dict(tensors)
    hyper.to(device=device, dtype=torch.float32)
    hyper.requires_grad_(False)
    hyper.eval()
    return hyper
