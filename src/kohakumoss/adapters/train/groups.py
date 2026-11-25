"""Optimiser parameter groups of an adapter set: weight-decay rules, Muon eligibility,
the LoRA+ ratio on each kind's zero-initialised factors and LR_GROUPS name scales."""

import re

from kohakumoss.logging.log import get_logger

NO_DECAY = ("dora_scale", "rescale", "scalar", "lambda_layer", "alpha", "sora_")
WD_RULES = ("moss", "none", "all")
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


def is_up(name: str, up_side: tuple[str, ...]) -> bool:
    """`name` (relative to its LyCORIS module) is one of the kind's LoRA+ factors."""
    return any(name == tag or name.startswith(tag + ".") for tag in up_side)


def _no_decay_name(name: str, p) -> bool:
    return p.dim() < 2 or any(tag in name for tag in NO_DECAY)


def _decay(rules: str, name: str, p, weight_decay: float) -> float:
    match rules:
        case "moss":
            return 0.0 if _no_decay_name(name, p) else weight_decay
        case "none":
            return 0.0
        case "all":
            return weight_decay
    raise ValueError(f"unknown WD_RULES {rules!r}; known {WD_RULES}")


def _muon(use_muon: bool, name: str, p) -> bool:
    """2-D factors with both sides > 1 (DoRA magnitudes and scalars stay on AdamW)."""
    return (
        use_muon and p.dim() == 2 and min(p.shape) > 1 and not _no_decay_name(name, p)
    )


def build(
    entries,
    lr: float,
    weight_decay: float,
    rules: str,
    use_muon: bool,
    ratio: float,
    lr_groups: dict[str, float] | None = None,
) -> list[dict]:
    """`entries`: (full name, name in its module, param, up_side) per trainable param.

    `lr_groups` (name regex -> multiplier, matched against the full name, first
    matching pattern in dict order) scales the group lr on top of the LoRA+ ratio:
    `lr * multiplier * (ratio if up else 1)`.
    """
    scales = _compile_lr_groups(lr_groups)
    used: set[str] = set()
    buckets: dict[tuple, dict] = {}
    for full, name, p, up_side in entries:
        wd = _decay(rules, name, p, weight_decay)
        up = ratio != 1.0 and is_up(name, up_side)
        muon = _muon(use_muon, name, p)
        scale = _lr_scale(scales, full, used)
        group = buckets.setdefault(
            (wd, up, muon, scale),
            {
                "params": [],
                "names": [],
                "splits": [],
                "lr": lr * scale * (ratio if up else 1.0),
                "weight_decay": wd,
                "decay_to_one": False,
                "decay_to_one_coef": 0.0,
                "use_muon": muon,
            },
        )
        group["params"].append(p)
        group["names"].append(full)
        group["splits"].append(1)
    _warn_lr_groups(scales, used)
    return [buckets[k] for k in sorted(buckets)]
