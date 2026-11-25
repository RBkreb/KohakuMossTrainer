"""Parameter grouping: weight-decay rules, decay-to-one targets, decay-to-start anchors,
Muon eligibility, LR_GROUPS scales."""

import re
from collections.abc import Iterable

from torch import Tensor, nn

from kohakumoss.logging.log import get_logger

EMBEDDING_PREFIXES = ("video_embedder", "text_embedder.proj", "time_embedding")
HEAD_PREFIXES = ("final_linear_video",)
FUSED_SPLITS = {r"(^|\.)qkv\.weight$": 3, r"(^|\.)ffn_up_gate\.weight$": 2}
log = get_logger(__name__)


def _compile_lr_groups(
    lr_groups: dict[str, float] | None,
) -> list[tuple[str, re.Pattern, float]]:
    """`LR_GROUPS` as (source text, compiled pattern, multiplier), dict order kept."""
    return [
        (text, re.compile(text), float(scale))
        for text, scale in (lr_groups or {}).items()
    ]


def _lr_scale(
    rules: list[tuple[str, re.Pattern, float]], name: str, used: set[str]
) -> float:
    """The first rule whose pattern searches `name`: its multiplier, else 1.0."""
    for text, pattern, scale in rules:
        if pattern.search(name):
            used.add(text)
            return scale
    return 1.0


def _warn_lr_groups(rules: list[tuple[str, re.Pattern, float]], used: set[str]) -> None:
    """A pattern that matched no parameter is almost always a typo."""
    for text, _, _ in rules:
        if text not in used:
            log.warning(f"LR_GROUPS pattern {text!r} matched no trainable parameter")


def _is_no_decay(name: str, p: nn.Parameter) -> bool:
    return (
        p.ndim < 2
        or "norm" in name
        or name.endswith(".bias")
        or name.startswith(EMBEDDING_PREFIXES)
    )


def last_block_qk_norm(depth: int) -> str:
    """Name pattern of the last joint block's q/k norm weights."""
    return rf"^layers\.{depth - 1}\.(q_norm|k_norm)\.weight$"


def muon_eligible(name: str, p: nn.Parameter) -> bool:
    """2-D hidden matrices; embeddings and the output head stay on AdamW."""
    return p.ndim == 2 and not name.startswith(EMBEDDING_PREFIXES + HEAD_PREFIXES)


def muon_splits(name: str) -> int:
    for pattern, n in FUSED_SPLITS.items():
        if re.search(pattern, name):
            return n
    return 1


def build_param_groups(
    named: Iterable[tuple[str, nn.Parameter]],
    lr: float,
    weight_decay: float,
    rules: str = "moss",
    decay_to_one: str | None = None,
    use_muon: bool = False,
    anchors: dict[str, Tensor] | None = None,
    lr_groups: dict[str, float] | None = None,
) -> list[dict]:
    """Groups for the optimiser. `rules`: "moss" | "none" | "all".

    Each group carries `names` (for diagnostics), `use_muon`, `splits` (per-param chunk
    counts for Muon) and `decay_to_one` (bool). With `anchors` (name -> start value),
    every decayed parameter decays towards its anchor instead of zero: its group
    carries `anchor`, the anchors in `params` order (`anchor_pull`). `lr_groups`
    (name regex -> multiplier, first matching pattern in dict order) sets each group's
    lr to `lr * multiplier`.
    """
    scales = _compile_lr_groups(lr_groups)
    used: set[str] = set()
    buckets: dict[tuple, dict] = {}
    for name, p in named:
        if not p.requires_grad:
            continue
        match rules:
            case "moss":
                wd = 0.0 if _is_no_decay(name, p) else weight_decay
            case "none":
                wd = 0.0
            case "all":
                wd = weight_decay
            case _:
                raise ValueError(f"unknown WD_RULES {rules!r}")
        to_one = bool(decay_to_one and re.match(decay_to_one, name))
        muon = use_muon and muon_eligible(name, p)
        anchored = anchors is not None and wd > 0 and not to_one
        scale = _lr_scale(scales, name, used)
        key = (wd, to_one, muon, anchored, scale)
        group = buckets.setdefault(
            key,
            {
                "params": [],
                "names": [],
                "splits": [],
                "lr": lr * scale,
                "weight_decay": 0.0 if to_one else wd,
                "decay_to_one": to_one,
                "decay_to_one_coef": weight_decay if to_one else 0.0,
                "use_muon": muon,
            },
        )
        group["params"].append(p)
        group["names"].append(name)
        group["splits"].append(muon_splits(name))
        if anchored:
            group.setdefault("anchor", []).append(anchors[name])
    _warn_lr_groups(scales, used)
    return list(buckets.values())
