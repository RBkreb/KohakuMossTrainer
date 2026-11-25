"""Static quality inspection of adapters at checkpoint writes: SVIR (singular value
intrusion ratio) and EK (excess kurtosis) over the LoRA Wdown factors, no inference.
With QUALITY_CHECK['baseline'] (default 'auto': on under EVA), the init factors are
measured once before training and the watch flags read the drift from that baseline
instead of the paper's random-init premise. See docs/internals/quality.md."""

import json
import re
from pathlib import Path

import torch

from kohakumoss.logging.log import get_logger, hl
from kohakumoss_common.calls import nothing

log = get_logger(__name__)

DEFAULTS = {"tau": 3.0, "tau_low_rank": 3.5, "ema": False, "top": 5, "baseline": "auto"}
SVIR_WATCH = 0.05
REPORT = "quality.jsonl"
BASELINE = "quality_baseline.json"
SUFFIX = ".lora_down.weight"


def options(k: dict) -> dict | None:
    """QUALITY_CHECK normalized: None when off, else the defaults merged with the
    given dict."""
    v = k.get("QUALITY_CHECK", False)
    if v is None or v is False:
        return None
    if v is True:
        return dict(DEFAULTS)
    return {**DEFAULTS, **v}


def option_problems(k: dict) -> list[str]:
    """QUALITY_CHECK as a bool or a dict over DEFAULTS: tau values > 0, bool ema, int
    top >= 0."""
    v = k.get("QUALITY_CHECK", False)
    if v is None or isinstance(v, bool):
        return []
    if not isinstance(v, dict):
        return [f"QUALITY_CHECK must be a bool or a dict, got {v!r}"]
    p = []
    unknown = sorted(set(v) - set(DEFAULTS))
    if unknown:
        p.append(f"QUALITY_CHECK: unknown keys {unknown}, allowed {sorted(DEFAULTS)}")
    for name in ("tau", "tau_low_rank"):
        val = v.get(name, DEFAULTS[name])
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val <= 0:
            p.append(f"QUALITY_CHECK[{name!r}] must be a number > 0, got {val!r}")
    ema = v.get("ema", DEFAULTS["ema"])
    if not isinstance(ema, bool):
        p.append(f"QUALITY_CHECK['ema'] must be a bool, got {ema!r}")
    baseline = v.get("baseline", DEFAULTS["baseline"])
    if not isinstance(baseline, bool) and baseline != "auto":
        p.append(
            f"QUALITY_CHECK['baseline'] must be a bool or 'auto', got {baseline!r}"
        )
    top = v.get("top", DEFAULTS["top"])
    if isinstance(top, bool) or not isinstance(top, int) or top < 0:
        p.append(f"QUALITY_CHECK['top'] must be an int >= 0, got {top!r}")
    return p


def kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis of the flattened `x` (0 for a normal draw, -1.2 for uniform)."""
    x = x.flatten().to(torch.float64)
    var = x.var(unbiased=False)
    if var <= 0:
        return 0.0
    z = (x - x.mean()) / var.sqrt()
    return z.square().square().mean().item() - 3.0


def layer_metrics(w: torch.Tensor, tau: float, tau_low_rank: float) -> dict | None:
    """EK, SVIR, intrusion count and effective rank of one 2-D Wdown factor; None for
    a non-2-D, empty or constant tensor."""
    if not torch.is_tensor(w) or w.dim() != 2 or w.numel() == 0:
        return None
    x = w.detach().to("cpu", torch.float32)
    var = x.var(unbiased=False)
    if var <= 0:
        return None
    sv = torch.linalg.svdvals(x)
    med = sv.median()
    r_eff = (x.square().sum() / sv[0].square()).item()
    rank = min(x.shape)
    thr = tau_low_rank if rank <= 4 else tau
    intrusions = int((sv > thr * med).sum())
    return {
        "ek": kurtosis(x),
        "svir": intrusions / r_eff if r_eff > 0 else 0.0,
        "intrusions": intrusions,
        "r_eff": r_eff,
    }


def wdown_layers(state: dict) -> dict[str, torch.Tensor]:
    """The Wdown factors of a trainable state: name without the LyCORIS suffix."""
    return {
        name[: -len(SUFFIX)]: t
        for name, t in state.items()
        if name.endswith(SUFFIX) and torch.is_tensor(t)
    }


def baseline_enabled(k: dict) -> bool:
    """Whether the run reports baseline-relative SVIR / EK: 'auto' turns it on
    exactly when EVA initialises the factors (their init is not the paper's
    Gaussian premise: EK starts non-zero, and spectral growth along the data's
    principal directions is expected signal, not necessarily memorisation)."""
    opts = options(k)
    if opts is None:
        return False
    mode = opts["baseline"]
    if mode == "auto":
        return k.get("EVA") not in (None, False)
    return bool(mode)


def capture_baseline(k: dict, trainable, output_dir) -> None:
    """The run's starting-point SVIR / EK per Wdown layer, written once before
    training (after EVA / INIT_FROM, rank 0, PARALLEL 'none'); a report never
    fails the run, so neither does this."""
    if not baseline_enabled(k):
        return
    try:
        if k["PARALLEL"] != "none":
            log.warning(
                "quality baseline needs PARALLEL='none' (rank 0 holds the full "
                f"factors); skipped under PARALLEL={k['PARALLEL']!r}"
            )
            return
        opts = options(k)
        layers = {}
        for name, t in wdown_layers(trainable.state_dict()).items():
            row = layer_metrics(t, opts["tau"], opts["tau_low_rank"])
            if row is not None:
                layers[name] = row
        if not layers:
            log.warning("quality baseline: no lora_down factors; skipped")
            return
        blob = {
            "tau": opts["tau"],
            "tau_low_rank": opts["tau_low_rank"],
            "layers": layers,
        }
        (Path(output_dir) / BASELINE).write_text(json.dumps(blob, indent=1))
        log.info(
            f"quality baseline: {hl(len(layers))} Wdown layers at init -> {BASELINE}"
        )
    except Exception as err:  # noqa: BLE001 - monitoring never fails the run
        log.warning(f"quality baseline capture failed: {err}")


def load_baseline(output_dir, opts: dict) -> dict | None:
    """The captured per-layer baseline; None (absolute thresholds) when the file
    is missing (a resumed run whose original had none) or its tau differs."""
    path = Path(output_dir) / BASELINE
    if not path.is_file():
        log.warning(
            f"quality baseline {BASELINE} missing (resumed run?); "
            "reporting absolute SVIR / EK thresholds"
        )
        return None
    blob = json.loads(path.read_text(encoding="utf-8"))
    if (
        blob.get("tau") != opts["tau"]
        or blob.get("tau_low_rank") != opts["tau_low_rank"]
    ):
        log.warning(
            "quality baseline was captured with a different tau; "
            "reporting absolute SVIR / EK thresholds"
        )
        return None
    return blob["layers"]


def _baseline_name(name: str) -> str:
    """The baseline's key of a factor name: EMA states carry an `ema<decay>/` prefix,
    the baseline is captured from the trainable set."""
    return re.sub(r"^ema[^/]*/", "", name)


def inspect_state(state: dict, opts: dict, baseline: dict | None = None) -> dict | None:
    """Layer-wise SVIR / EK of every Wdown factor in `state`, aggregated the paper's
    way (SVIR mean over layers, EK quantiles and the positive share); None without
    usable factors. With a `baseline` (per-layer init metrics), matched layers also
    carry svir_delta / ek_delta and the watch flags read the deltas: drift from the
    run's own starting point, not from the paper's random-init premise."""
    layers = {}
    for name, t in wdown_layers(state).items():
        row = layer_metrics(t, opts["tau"], opts["tau_low_rank"])
        if row is not None:
            key = _baseline_name(name)
            if baseline is not None and key in baseline:
                row["svir_delta"] = row["svir"] - baseline[key]["svir"]
                row["ek_delta"] = row["ek"] - baseline[key]["ek"]
            layers[name] = row
    if not layers:
        return None
    eks = torch.tensor([v["ek"] for v in layers.values()], dtype=torch.float64)
    svirs = torch.tensor([v["svir"] for v in layers.values()], dtype=torch.float64)
    reffs = torch.tensor([v["r_eff"] for v in layers.values()], dtype=torch.float64)
    intrusions = [v["intrusions"] for v in layers.values()]
    matched = [v for v in layers.values() if "svir_delta" in v]
    watch = []
    if matched:
        d_svirs = torch.tensor([v["svir_delta"] for v in matched], dtype=torch.float64)
        d_eks = torch.tensor([v["ek_delta"] for v in matched], dtype=torch.float64)
        if d_svirs.mean().item() > SVIR_WATCH:
            watch.append("svir")
        if d_eks.median().item() > 0.0:
            watch.append("ek")
    else:
        if svirs.mean().item() > SVIR_WATCH:
            watch.append("svir")
        if eks.median().item() > 0.0:
            watch.append("ek")
    top = sorted(layers.items(), key=lambda kv: -kv[1].get("ek_delta", kv[1]["ek"]))[
        : opts["top"]
    ]
    out = {
        "layers": len(layers),
        "svir_mean": svirs.mean().item(),
        "svir_max": svirs.max().item(),
        "layers_with_intrusions": sum(1 for n in intrusions if n),
        "intrusions_max": max(intrusions),
        "ek_median": eks.median().item(),
        "ek_mean": eks.mean().item(),
        "ek_p05": torch.quantile(eks, 0.05).item(),
        "ek_p95": torch.quantile(eks, 0.95).item(),
        "ek_max": eks.max().item(),
        "ek_frac_pos": (eks > 0).double().mean().item(),
        "reff_median": reffs.median().item(),
        "watch": watch,
        "top": [
            {
                "layer": name,
                "ek": row["ek"],
                "svir": row["svir"],
                "intrusions": row["intrusions"],
                "r_eff": row["r_eff"],
                **(
                    {"ek_delta": row["ek_delta"], "svir_delta": row["svir_delta"]}
                    if "svir_delta" in row
                    else {}
                ),
            }
            for name, row in top
        ],
    }
    if matched:
        out["baseline_layers"] = len(matched)
        out["svir_mean_delta"] = d_svirs.mean().item()
        out["ek_median_delta"] = d_eks.median().item()
    return out


def rows_of(
    k: dict, state: dict, step: int, kind: str, baseline: dict | None = None
) -> list[dict]:
    """One summary row per inspected source (trainable, and ema when asked)."""
    opts = options(k)
    sources = ("trainable", "ema") if opts["ema"] else ("trainable",)
    rows = []
    for source in sources:
        summary = inspect_state(state.get(source) or {}, opts, baseline)
        if summary is not None:
            rows.append({"step": step, "kind": kind, "source": source, **summary})
    return rows


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append the rows to `<output>/quality.jsonl` (ASCII JSON, one row a line)."""
    with open(path, "a", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def log_row(row: dict) -> None:
    """The console line of one summary row."""
    flags = ",".join(row["watch"]) or "none"
    delta = ""
    if "svir_mean_delta" in row:
        delta = (
            f", delta SVIR mean {row['svir_mean_delta']:+.4f} "
            f"delta EK median {row['ek_median_delta']:+.2f} "
            f"(baseline {row['baseline_layers']}/{row['layers']})"
        )
    log.info(
        f"quality step {row['step']} {row['source']}: {row['layers']} layers, "
        f"SVIR mean {row['svir_mean']:.4f} max {row['svir_max']:.4f} "
        f"({row['layers_with_intrusions']} with intrusions), "
        f"EK median {row['ek_median']:+.2f} p95 {row['ek_p95']:+.2f} "
        f"({row['ek_frac_pos']:.0%} > 0){delta}, watch: {flags}"
    )


def report(k: dict, output_dir, state: dict, step: int, kind: str) -> None:
    """Inspect, append to the report and log; a state without LoRA Wdown factors is
    logged and skipped."""
    baseline = load_baseline(output_dir, options(k)) if baseline_enabled(k) else None
    rows = rows_of(k, state, step, kind, baseline)
    if not rows:
        log.warning(
            f"quality step {step}: no lora_down factors (QUALITY_CHECK needs a "
            f"LoRA-style ALGO); skipped"
        )
        return
    append_rows(Path(output_dir) / REPORT, rows)
    for row in rows:
        log_row(row)


def safe_report(k: dict, output_dir, state: dict, step: int, kind: str) -> None:
    """`report` with failures logged, never raised into the run."""
    try:
        report(k, output_dir, state, step, kind)
    except Exception as err:  # noqa: BLE001 - a report never fails a finished save
        log.warning(f"quality check of step {step} failed: {err}")


def after_save_quality(k: dict, output_dir, is_main: bool):
    """`after_save(step, path, parts)` of the CheckpointManager: the report rows of
    the saved trainable (and EMA) weights, rank 0 only."""
    if options(k) is None or not is_main:
        return nothing

    def after_save(step: int, path, parts: dict | None = None) -> None:
        if parts:
            safe_report(k, output_dir, parts, step, "checkpoint")

    return after_save


def final_quality(k: dict, parts, state, output_dir, is_main: bool) -> None:
    """The run's final export write: one report of `parts(state)`; every rank calls it
    (the snapshot is collective), rank 0 writes."""
    if options(k) is None:
        return
    try:
        snapshot = parts(state)
    except Exception as err:  # noqa: BLE001 - the export already ran
        log.warning(f"quality snapshot of step {state.step} failed: {err}")
        return
    if is_main:
        safe_report(k, output_dir, snapshot, state.step, "final-export")


def chain_after_save(*callbacks):
    """One `after_save(step, path, parts)` that calls `callbacks` in order."""

    def after_save(step: int, path, parts: dict | None = None) -> None:
        for cb in callbacks:
            cb(step, path, parts)

    return after_save
