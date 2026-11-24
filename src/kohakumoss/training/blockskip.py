"""DiT-BlockSkip (arXiv 2603.20755): memory-efficient adapter fine-tuning.

Skip the first `n` and last `m` joint blocks; LoRA trains only the kept middle
blocks. Two stages share the run's seeded data / sigma / noise streams:

  precompute: a full no-grad forward per micro-batch stores the joint stream at
    the first kept block (absolute feature: the prefix is frozen, so it is exact)
    and the skipped tail's residual delta f_L - f_{L-m} (Eq. 3), one file per
    optimiser step under BLOCKSKIP['cache'].
  train: the step's forward runs only blocks [n, L-m); the stored head feature
    replaces the frozen prefix, the tail delta is added after the last kept
    block. Skipped blocks are never entered: their stream swaps, activations
    and backward all disappear. The sigma of every micro-batch is checked
    against the stored one, so a data/RNG drift fails loudly instead of training
    on misaligned features.

BLOCKSKIP = {"mode": "precompute" | "train", "skip": 0.5 | (n, m), "cache": dir}
"""

import copy
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from kohakumoss.adapters.build.presets import resolve_preset
from kohakumoss.diffusion.timestep.moss import MossTimestep
from kohakumoss.logging.log import get_logger, hl
from kohakumoss.training.drivers.common import grouped
from kohakumoss.types.context import BlockCtx

log = get_logger(__name__)


@dataclass(frozen=True)
class BlockSkipSpec:
    mode: str
    n: int
    m: int
    cache: str


def spec(k: dict, depth: int) -> BlockSkipSpec | None:
    """The run's BlockSkipSpec (None when BLOCKSKIP is off); raises on a combination
    the method does not support."""
    bs = k.get("BLOCKSKIP") or {}
    if not bs:
        return None
    mode = bs.get("mode", "train")
    if mode not in ("precompute", "train"):
        raise ValueError(f"BLOCKSKIP['mode'] must be 'precompute' or 'train': {mode!r}")
    cache = bs.get("cache", "")
    if not cache:
        raise ValueError("BLOCKSKIP needs 'cache': the feature directory")
    skip = bs.get("skip", 0.5)
    if isinstance(skip, (tuple, list)):
        n, m = (int(v) for v in skip)
    else:
        total = round(depth * float(skip))
        n = round(0.6 * total)  # keep the middle: front-heavy split (paper Sec. 3.2)
        m = total - n
    if n < 0 or m < 0 or n + m >= depth:
        raise ValueError(f"BLOCKSKIP skip ({n}, {m}) leaves no kept block of {depth}")
    if k.get("PARALLEL", "none") != "none":
        raise ValueError("BLOCKSKIP needs PARALLEL='none'")
    if k.get("ALGO") == "full":
        raise ValueError("BLOCKSKIP is adapter-only (ALGO != 'full')")
    if k.get("STREAM_LOSS") or k.get("CONCENTRATION_LOSS"):
        raise ValueError("BLOCKSKIP does not support stream penalties (skipped blocks)")
    if k.get("EVA"):
        raise ValueError("BLOCKSKIP does not support EVA calibration")
    return BlockSkipSpec(mode, n, m, str(cache))


def restrict_preset(preset, n: int, m: int, depth: int):
    """PRESET with every `layers.\\d+` pattern restricted to the kept blocks
    [n, depth-m); non-block targets (modulation, embeds, text refiner) are dropped.
    A toml path is returned unchanged (the user restricts it themselves)."""
    kept = "|".join(str(i) for i in range(n, depth - m))
    block_pat = r"layers\.(?:" + kept + r")"

    def rewrite(patterns):
        out = []
        for p in patterns:
            if r"layers\.\d+" in p:
                out.append(p.replace(r"layers\.\d+", block_pat))
        return out

    if isinstance(preset, dict):
        out = copy.deepcopy(preset)
        out["target_name"] = rewrite(out.get("target_name", []))
        if "norm_name" in out:
            out["norm_name"] = rewrite(out["norm_name"])
        return out
    if (
        isinstance(preset, str)
        and not preset.endswith(".toml")
        and not os.path.isfile(preset)
    ):
        full, norms = resolve_preset(preset, prefix="")
        full["target_name"] = rewrite(full["target_name"])
        full["norm_name"] = rewrite(norms)
        return full
    log.warning(
        "BLOCKSKIP: PRESET is a toml path; restrict its block patterns yourself"
    )
    return preset


def sigma_table(k: dict) -> dict:
    """The single-stage moss sampler's (sigma grid, cdf) for the patch planner."""
    if k.get("TIMESTEP", "moss") != "moss":
        raise ValueError("RESOLUTION_MODE='patch' needs TIMESTEP='moss'")
    sampler = MossTimestep(**k.get("TIMESTEP_KWARGS", {}))
    if len(sampler.stages) > 1:
        raise ValueError(
            "RESOLUTION_MODE='patch' needs a single-stage timestep schedule"
        )
    stage = sampler.stages[0]
    return {"sigma_grid": stage.sigma.tolist(), "sigma_cdf": stage.cdf.tolist()}


class CaptureRunner:
    """Wraps the run's runner; records the joint stream entering block `n` (head)
    and the tail segment's input / output during a full forward."""

    def __init__(self, inner, n: int, m: int, depth: int):
        self.inner = inner
        self.n = n
        self.tail_start = depth - m
        self.last = depth - 1
        self.head: Tensor | None = None
        self.tail_in: Tensor | None = None
        self.tail_out: Tensor | None = None

    def run(self, index: int, block, x: Tensor, ctx: BlockCtx) -> Tensor:
        if index == self.n:
            self.head = x.detach().clone()
        if index == self.tail_start:
            self.tail_in = x.detach().clone()
        y = self.inner.run(index, block, x, ctx)
        if index == self.last:
            self.tail_out = y.detach().clone()
        return y


def _feature_path(root: str, step: int) -> Path:
    return Path(root) / f"feat_{step:06d}.pt"


def save_features(root: str, step: int, n: int, m: int, records: list) -> None:
    path = _feature_path(root, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"n": n, "m": m, "records": records}, path)


def load_features(root: str, step: int, n: int, m: int) -> list:
    path = _feature_path(root, step)
    if not path.is_file():
        raise FileNotFoundError(
            f"BlockSkip features for step {step} missing ({path}); run the "
            "precompute stage with the same seed / data / STEPS first"
        )
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob["n"] != n or blob["m"] != m:
        raise ValueError(
            f"{path}: stored skip ({blob['n']}, {blob['m']}) != run's ({n}, {m})"
        )
    return blob["records"]


class FeatureReader:
    """Per-step feature files with a background prefetch of the next step."""

    def __init__(self, root: str, n: int, m: int):
        self.root = root
        self.n = n
        self.m = m
        self._loaded: dict[int, object] = {}
        self._pending: set[int] = set()
        self._lock = threading.Lock()
        self._req: queue.Queue = queue.Queue()
        self._cur_step = -1
        self._cur: list = []
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self) -> None:
        while True:
            step = self._req.get()
            try:
                out = load_features(self.root, step, self.n, self.m)
            except Exception as err:  # noqa: BLE001 - re-raised on the training thread
                out = err
            with self._lock:
                self._loaded[step] = out
                self._pending.discard(step)

    def _prefetch(self, step: int) -> None:
        with self._lock:
            if step not in self._loaded and step not in self._pending:
                self._pending.add(step)
                self._req.put(step)

    def get(self, step: int, micro: int) -> dict:
        if step != self._cur_step:
            self._prefetch(step)
            while True:
                with self._lock:
                    if step in self._loaded:
                        out = self._loaded.pop(step)
                        for old in list(self._loaded):
                            if old < step:
                                del self._loaded[old]
                        break
                time.sleep(0.002)
            if isinstance(out, Exception):
                raise out
            self._cur, self._cur_step = out, step
            self._prefetch(step + 1)
        return self._cur[micro]


def precompute(bs: BlockSkipSpec, k, built, step, loader, batcher, device, autocast):
    """The precompute stage: replay the training loop's batches, sigmas and noise
    with a full no-grad forward, storing the boundary features of every micro-batch."""
    model = built.backbone
    depth = model.cfg.depth
    inner = model.runner
    capture = CaptureRunner(inner, bs.n, bs.m, depth)
    model.set_runner(capture)
    groups = grouped(loader, k["GRAD_ACC"])
    steps = k["STEPS"]
    every = max(1, steps // 10)
    start = time.perf_counter()
    log.info(
        f"blockskip precompute: skip first {hl(bs.n)} + last {hl(bs.m)} of {depth} "
        f"blocks, {steps} steps -> {hl(bs.cache, 'path')}"
    )
    try:
        for index in range(steps):
            group = next(groups)
            records = []
            for hb in group:
                batch = batcher(hb)
                p = step.prepare(batch, index)
                with torch.no_grad(), autocast():
                    model(p.z, p.text, p.sigma, p.seq)
                tail = None
                if bs.m:
                    tail = (
                        (capture.tail_out.float() - capture.tail_in.float())
                        .to(capture.head.dtype)
                        .cpu()
                    )
                records.append(
                    {
                        "sigma": p.sigma.detach().cpu(),
                        "head": capture.head.cpu(),
                        "tail": tail,
                    }
                )
            save_features(bs.cache, index, bs.n, bs.m, records)
            if (index + 1) % every == 0 or index + 1 == steps:
                elapsed = time.perf_counter() - start
                log.info(
                    f"blockskip precompute {index + 1}/{steps} "
                    f"({elapsed:.1f}s, {elapsed / (index + 1):.2f}s/step)"
                )
    finally:
        model.set_runner(inner)
    log.info(f"blockskip precompute done in {time.perf_counter() - start:.1f}s")


class BlockSkipForward:
    """The training forward: cached head feature -> kept blocks -> + tail delta."""

    def __init__(self, model, reader: FeatureReader, n: int, m: int, device):
        self.model = model
        self.reader = reader
        self.n = n
        self.m = m
        self.depth = model.cfg.depth
        self.device = device
        self.step = -1
        self.micro = 0

    def begin(self, step: int) -> None:
        if step != self.step:
            self.step, self.micro = step, 0

    def __call__(self, img, text, sigma, seq) -> Tensor:
        rec = self.reader.get(self.step, self.micro)
        self.micro += 1
        if not torch.equal(sigma, rec["sigma"].to(sigma.device)):
            raise RuntimeError(
                f"BlockSkip step {self.step} micro {self.micro - 1}: sigma mismatch "
                "with the precomputed features (data order / seed / RESOLUTION_MODE "
                "differs between the precompute and train runs)"
            )
        model = self.model
        mod, head = model.conditioning(sigma)
        x = rec["head"].to(self.device, model._stream_dtype, non_blocking=True)
        ctx = BlockCtx(seq, mod, model.rope.bands, model.residual_scale)
        for i in range(self.n, self.depth - self.m):
            x = model.runner.run(i, model.layers[i], x, ctx)
        if self.m:
            x = x + rec["tail"].to(self.device, x.dtype, non_blocking=True)
        return model.head(x, head, seq)


def install_train(bs: BlockSkipSpec, built, step, device) -> None:
    """Replace the TrainStep's forward with the block-skip path and chain its
    per-micro-batch index ahead of the adapter step state."""
    reader = FeatureReader(bs.cache, bs.n, bs.m)
    forward = BlockSkipForward(built.backbone, reader, bs.n, bs.m, device)
    step.backbone = forward
    inner = step.pre_forward

    def pre_forward(sigma, seq, index):
        forward.begin(index)
        inner(sigma, seq, index)

    step.pre_forward = pre_forward
    log.info(
        f"blockskip train: blocks [{bs.n}, {built.backbone.cfg.depth - bs.m}) kept, "
        f"features from {hl(bs.cache, 'path')}"
    )
