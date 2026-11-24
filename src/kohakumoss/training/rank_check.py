"""One-shot spectral audit of the adapter at a chosen step, through the optional
`loraspectra` CLI (`cargo install loraspectra`); skipped with one warning when the
tool, a plain layout or LoRA factors are missing. See docs/internals/rank-check.md."""

import contextlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from safetensors.torch import save_file

from kohakumoss.logging.log import get_logger

log = get_logger(__name__)

DEFAULTS = {"step": 10, "layers": True, "keep": False, "timeout": 300.0}
TOOL = "loraspectra"
ENV = "LORASPECTRA"
REPORT = "rank_check.json"
FACTORS = (".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale")
FACTOR_ALGOS = frozenset({"lora", "locon", "nara", "tlora", "dylora"})
ROW = re.compile(
    r"^(?P<name>\S+)\s+(?P<rank>\d+)\s+(?P<r_eff>[\d.]+)\s+(?P<r_stable>[\d.]+)\s+"
    r"(?P<k99>\d+)\s+(?P<k_mp>\d+)\s+(?P<norm_f>[\d.]+)\s+(?P<status>[A-Z_]+)\s*$"
)
SUMMARY = re.compile(
    r"Summary: (?P<layers>\d+) layers \| Mean rank ([\d.]+) -> k_99% ([\d.]+) / "
    r"k_MP ([\d.]+) \(r_eff ([\d.]+)\)"
)


def options(k: dict) -> dict | None:
    """RANK_CHECK normalized: None when off, else the defaults merged with the
    given dict."""
    v = k.get("RANK_CHECK", False)
    if v is None or v is False:
        return None
    if v is True:
        return dict(DEFAULTS)
    return {**DEFAULTS, **v}


def option_problems(k: dict) -> list[str]:
    """RANK_CHECK as a bool or a dict over DEFAULTS: int step >= 1, bool layers /
    keep, number timeout > 0."""
    v = k.get("RANK_CHECK", False)
    if v is None or isinstance(v, bool):
        return []
    if not isinstance(v, dict):
        return [f"RANK_CHECK must be a bool or a dict, got {v!r}"]
    p = []
    unknown = sorted(set(v) - set(DEFAULTS))
    if unknown:
        p.append(f"RANK_CHECK: unknown keys {unknown}, allowed {sorted(DEFAULTS)}")
    step = v.get("step", DEFAULTS["step"])
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        p.append(f"RANK_CHECK['step'] must be an int >= 1, got {step!r}")
    for name in ("layers", "keep"):
        val = v.get(name, DEFAULTS[name])
        if not isinstance(val, bool):
            p.append(f"RANK_CHECK[{name!r}] must be a bool, got {val!r}")
    timeout = v.get("timeout", DEFAULTS["timeout"])
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        p.append(f"RANK_CHECK['timeout'] must be a number > 0, got {timeout!r}")
    return p


def tool_path() -> str | None:
    """The loraspectra executable: $LORASPECTRA, else the PATH entry."""
    named = os.environ.get(ENV)
    if named and Path(named).exists():
        return named
    return shutil.which(TOOL)


def factor_tensors(state: dict) -> dict:
    """The saved adapter's spectral factors: Wdown / Wup pairs plus alpha and DoRA
    magnitude, on the host."""
    out = {}
    for name, t in state.items():
        if not name.endswith(FACTORS) or not hasattr(t, "detach"):
            continue
        if name.endswith((".lora_down.weight", ".lora_up.weight")) and t.dim() != 2:
            continue
        out[name] = t.detach().to("cpu").contiguous()
    return out


def run_tool(
    exe: str, path: Path, timeout: float, *args: str
) -> subprocess.CompletedProcess:
    """`loraspectra audit` with `args`, UTF-8 captured; the caller handles errors."""
    return subprocess.run(
        [exe, "audit", str(path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def summary_of(stdout: str) -> dict:
    """The `audit --json` payload."""
    return json.loads(stdout)


def table_of(stdout: str) -> tuple[list[dict], list[str]]:
    """The per-layer rows and the `[WARNING]` lines of an `audit` table; empty rows
    when the table is not recognized."""
    layers, warnings = [], []
    for line in stdout.splitlines():
        strip = line.strip()
        if strip.startswith("[WARNING]"):
            warnings.append(strip)
        match = ROW.match(strip)
        if match:
            row = match.groupdict()
            layers.append(
                {
                    "layer": row["name"],
                    "rank": int(row["rank"]),
                    "r_eff": float(row["r_eff"]),
                    "r_stable": float(row["r_stable"]),
                    "k99": int(row["k99"]),
                    "k_mp": int(row["k_mp"]),
                    "norm_f": float(row["norm_f"]),
                    "status": row["status"],
                }
            )
    return layers, warnings


def guidance_of(summary: dict, layers: list[dict], dim: int) -> dict:
    """How far the current adapter uses its nominal rank: the mean 99%-energy rank
    and MP rank against DIM, plus the per-layer status counts."""
    counts = {}
    for row in layers:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    k99 = summary.get("mean_rank_at_99_energy")
    mp = summary.get("mean_rank_marchenko_pastur")
    return {
        "dim": dim,
        "energy_use": None if not dim or k99 is None else k99 / dim,
        "mp_share": None if not dim or mp is None else mp / dim,
        "status_counts": counts,
    }


class RankCheck:
    """One-shot: at `step`, write the trainable factors to a temporary safetensors and
    audit it with loraspectra; the report lands in `<output>/rank_check.json`."""

    every = 1

    def __init__(self, k: dict, parts, output_dir, is_main: bool):
        opts = options(k)
        if opts is None or not is_main:
            self.every = 0
            return
        if k["PARALLEL"] != "none":
            log.warning(
                f"RANK_CHECK needs PARALLEL='none' (rank 0 holds the full adapter); "
                f"skipped under PARALLEL={k['PARALLEL']!r}"
            )
            self.every = 0
            return
        self.exe = tool_path()
        if not self.exe:
            log.warning(
                f"RANK_CHECK is on but {TOOL} was not found ($PATH or ${ENV}); "
                f"install with `cargo install {TOOL}`, skipped"
            )
            self.every = 0
            return
        if k["ALGO"] not in FACTOR_ALGOS:
            log.warning(
                f"RANK_CHECK needs a LoRA-style ALGO (lora_down factors), "
                f"got ALGO={k['ALGO']!r}; skipped"
            )
            self.every = 0
            return
        if k["STEPS"] < opts["step"]:
            log.warning(
                f"RANK_CHECK step {opts['step']} is past the run's {k['STEPS']} steps; "
                f"skipped"
            )
            self.every = 0
            return
        self.k = k
        self.parts = parts
        self.output_dir = Path(output_dir)
        self.opts = opts
        self.done = False

    def on_step_end(self, state, out) -> bool:
        if self.done:
            return True
        if state.step < self.opts["step"]:
            return False
        if state.step > self.opts["step"]:
            log.info(
                f"rank check step {self.opts['step']} skipped (resumed at "
                f"{state.step})"
            )
            self.done = True
            return True
        self.done = True
        try:
            self._run(state.step)
        except Exception as err:  # noqa: BLE001 - a report never fails the run
            log.warning(f"rank check of step {state.step} failed: {err}")
        return True

    def _run(self, step: int) -> None:
        tensors = factor_tensors(self.parts.trainable.state_dict())
        if not tensors:
            log.warning(
                "rank check: no lora_down factors (RANK_CHECK needs a LoRA-style "
                "ALGO); skipped"
            )
            return
        root = self.output_dir / "rank_check"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"step_{step:08d}.safetensors"
        save_file(tensors, path, metadata={"kohakumoss_step": str(step)})
        timeout = float(self.opts["timeout"])
        try:
            summary = summary_of(run_tool(self.exe, path, timeout, "--json").stdout)
            layers, warnings = [], []
            if self.opts["layers"]:
                plain = run_tool(self.exe, path, timeout).stdout
                layers, warnings = table_of(plain)
            report = {
                "step": step,
                "tool": TOOL,
                "summary": summary,
                "layers": layers,
                "warnings": warnings,
                "guidance": guidance_of(summary, layers, int(self.k.get("DIM") or 0)),
                "adapter_file": str(path) if self.opts["keep"] else None,
            }
            (self.output_dir / REPORT).write_text(json.dumps(report, indent=1))
            self._log(step, report)
        finally:
            if not self.opts["keep"]:
                path.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    root.rmdir()

    @staticmethod
    def _log(step: int, report: dict) -> None:
        s, g = report["summary"], report["guidance"]
        use = "" if g["energy_use"] is None else f" ({g['energy_use']:.0%} of DIM)"
        log.info(
            f"rank check step {step}: {s.get('layer_count')} layers, "
            f"stable {s.get('mean_stable_rank', float('nan')):.2f}, "
            f"k99 {s.get('mean_rank_at_99_energy', float('nan')):.1f}{use}, "
            f"MP {s.get('mean_rank_marchenko_pastur', float('nan')):.1f}, "
            f"statuses {g['status_counts']}, "
            f"{len(report['warnings'])} warning(s); early-delta utilization, "
            f"see {REPORT}"
        )
