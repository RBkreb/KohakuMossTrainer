"""AdapterConfig: what to train and how, resolved from the train-script knobs."""

import inspect
from dataclasses import dataclass, field

from lycoris.wrapper import LycorisNetwork

from kohakumoss.adapters.kinds import (
    BY_ALGO,
    lycoris_algo,
    nara_network_algo,
    nara_record_of,
)
from kohakumoss.adapters.nara.hyper import EMBEDS
from kohakumoss.adapters.train.tlora import MASK_KEYS

MODES = ("bypass", "merge")
BACKENDS = ("auto", "triton", "tilelang", "compile", "torch")
COMPILE_DYNAMIC = ("tokens", "static", "dynamic")
RESERVED = {
    "lora_dim",
    "alpha",
    "dropout",
    "rank_dropout",
    "module_dropout",
    "multiplier",
    "network_module",
}


def _parameters(fn) -> set[str]:
    kinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    sig = inspect.signature(fn)
    return {n for n, p in sig.parameters.items() if p.kind not in kinds} - {"self"}


def accepted_kwargs_of(module: type) -> set[str]:
    """ALGO_KWARGS names `module` (or the LyCORIS network) takes."""
    return (_parameters(module.__init__) | _parameters(LycorisNetwork.__init__)) - {
        "lora_name",
        "org_module",
        "module",
    }


def accepted_kwargs(algo: str) -> set[str]:
    """ALGO_KWARGS names the LyCORIS module (or network) of `algo` takes."""
    return accepted_kwargs_of(BY_ALGO[algo].module)


@dataclass(frozen=True)
class AdapterConfig:
    """One adapter configuration; `algo_kwargs` go to the LyCORIS module constructor
    (T-LoRA's `mask_*` keys configure its timestep mask instead). `backend` "auto"
    autotunes each kernel op against its compiled body; `compile_dynamic` is the compile
    candidate's shape mode; `fuse_base_add` puts the base add inside the op."""

    algo: str = "lora"
    preset: str | dict = "attn-mlp"
    dim: int = 32
    alpha: float = 32.0
    algo_kwargs: dict = field(default_factory=dict)
    dora: bool = False
    dora_on_output: bool = True
    dora_detach: bool = False
    nara: bool = False
    sora_post_rank: int = 0
    sora_initial_s_p: float = 1.0
    sora_pre_diag: bool = False
    eva: dict | None = None
    dropout: float = 0.0
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    mode: str = "bypass"
    train_norms: bool = False
    multiplier: float = 1.0
    prefix: str = "lora_unet"
    backend: str = "auto"
    compile_dynamic: str = "tokens"
    lr_ratio_up: float = 1.0
    seed: int = 0
    fuse_base_add: bool = True

    @property
    def lycoris_algo(self) -> str:
        return nara_network_algo(self.algo) if self.nara else lycoris_algo(self.algo)

    @property
    def record(self):
        """The kind record the configuration trains: the NARA switch swaps in the
        algo's noise-conditioned variant."""
        rec = BY_ALGO[self.algo]
        return nara_record_of(rec) if self.nara else rec

    @property
    def has_step_state(self) -> bool:
        """The adapters carry per-step state: T-LoRA masks, NaRA's noise level,
        DyLoRA ranks or module-dropout draws."""
        rec = BY_ALGO.get(self.algo)
        drawn = rec is not None and (
            rec.per_token_mask
            or rec.needs_sigma
            or self.nara
            or rec.draw_step is not None
        )
        return drawn or self.module_dropout > 0

    def mask_kwargs(self) -> dict:
        """T-LoRA mask settings (min_rank / alpha / max_timestep) from ALGO_KWARGS."""
        return {MASK_KEYS[k]: v for k, v in self.algo_kwargs.items() if k in MASK_KEYS}

    def validate(self) -> list[str]:
        """Every problem with this configuration (empty when valid)."""
        problems = []
        rec = BY_ALGO.get(self.algo)
        if rec is None:
            problems.append(f"ALGO={self.algo!r} not in {sorted(BY_ALGO)}")
        if self.mode not in MODES:
            problems.append(f"ADAPTER_MODE={self.mode!r} not in {MODES}")
        if self.backend not in BACKENDS:
            problems.append(f"adapter backend {self.backend!r} not in {BACKENDS}")
        if self.compile_dynamic not in COMPILE_DYNAMIC:
            problems.append(
                f"compile_dynamic={self.compile_dynamic!r} not in {COMPILE_DYNAMIC}"
            )
        if self.dim <= 0 and not (rec is not None and rec.rankless):
            problems.append(f"DIM must be positive, got {self.dim}")
        if self.dora and self.dropout > 0:
            problems.append("DORA ignores input dropout; set DROPOUT['input']=0")
        if self.nara and (rec is None or not rec.supports_nara):
            problems.append(f"NARA does not apply to ALGO={self.algo!r}")
        problems += self._sora_problems(rec)
        if self.eva is not None:
            if rec is None or not rec.supports_eva:
                problems.append(f"EVA does not apply to ALGO={self.algo!r}")
            unknown = sorted(set(self.eva) - {"batches", "min_batches", "sim_thresh"})
            if unknown:
                problems.append(f"EVA has unknown keys {unknown}")
            if self.module_dropout > 0:
                problems.append(
                    "EVA calibration needs every layer's activations; "
                    "set DROPOUT['module']=0"
                )
        for name in ("dropout", "rank_dropout", "module_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                problems.append(f"{name}={value} outside [0, 1)")
        if rec is not None:
            problems += self._kind_problems(rec)
        return problems

    def _sora_problems(self, rec) -> list[str]:
        """Post-Ortho stacks with DoRA; Pre-Diag replaces it (one conditioning)."""
        out = []
        if not (self.sora_post_rank or self.sora_pre_diag):
            return out
        if rec is None or not rec.supports_sora:
            out.append(f"SORA does not apply to ALGO={self.algo!r}")
        if self.sora_post_rank < 0:
            out.append("SORA_POST_ORTHO['rank'] must be >= 0 (0 = off)")
        if self.sora_initial_s_p <= 0:
            out.append("SORA_POST_ORTHO['initial_s_p'] must be positive")
        if self.sora_pre_diag:
            if self.dora:
                out.append("SORA_PRE_DIAG replaces DoRA; set DORA=False")
            if self.dropout > 0 or self.rank_dropout > 0:
                out.append(
                    "SORA_PRE_DIAG reads the delta twice per forward; set "
                    "DROPOUT['input']=0 and DROPOUT['rank']=0"
                )
        return out

    def _kind_problems(self, rec) -> list[str]:
        out = []
        if self.dora and not rec.supports_dora:
            out.append(f"DORA does not apply to ALGO={self.algo!r}")
        if rec.per_token_mask and self.mode == "merge":
            out.append(
                "ALGO='tlora' masks ranks per token, which no merged weight expresses; "
                "use ADAPTER_MODE='bypass'"
            )
        if rec.needs_sigma or self.nara:
            out += self._sigma_problems(rec)
        if self.nara and nara_record_of(rec) is None:
            # The switch itself is rejected above; there is no variant module to
            # check the kwargs against.
            return out
        return out + self._kwarg_problems()

    def _sigma_problems(self, rec) -> list[str]:
        """A noise-conditioned delta: bypass only, no rank dropout, no extra factors."""
        out = []
        if self.mode == "merge":
            out.append(
                f"ALGO={self.algo!r} conditions its delta on the noise level, which no "
                "merged weight expresses; use ADAPTER_MODE='bypass'"
            )
        if self.rank_dropout > 0:
            out.append(f"ALGO={self.algo!r} does not support rank dropout")
        for key in ("use_tucker", "use_scalar", "rs_lora"):
            if self.algo_kwargs.get(key):
                out.append(f"ALGO={self.algo!r} does not take ALGO_KWARGS[{key!r}]")
        embed = self.algo_kwargs.get("nara_embed", "fourier")
        if embed not in EMBEDS:
            out.append(f"ALGO_KWARGS['nara_embed']={embed!r} not in {EMBEDS}")
        return out

    def _kwarg_problems(self) -> list[str]:
        own = set(MASK_KEYS) if self.record.per_token_mask else set()
        accepted = accepted_kwargs_of(self.record.module)
        out = []
        for key in self.algo_kwargs:
            if key in own:
                continue
            if key in RESERVED:
                out.append(f"ALGO_KWARGS[{key!r}] is set by its own knob")
            elif key not in accepted:
                out.append(f"ALGO_KWARGS[{key!r}] is not a {self.algo} module argument")
        return out

    def module_kwargs(self) -> dict:
        """Constructor kwargs shared by every LyCORIS module of this configuration."""
        kwargs = {k: v for k, v in self.algo_kwargs.items() if k not in MASK_KEYS}
        kwargs.setdefault("bypass_mode", self.mode == "bypass")
        if self.dora:
            kwargs["weight_decompose"] = True
            kwargs["wd_on_out"] = self.dora_on_output
        return kwargs


def _sora_post(value) -> tuple[int, float]:
    """(rank, initial_s_p) of the SORA_POST_ORTHO knob: a rank dict, a bare rank,
    or a bool (True = rank 1)."""
    if isinstance(value, dict):
        return int(value.get("rank", 1)), float(value.get("initial_s_p", 1.0))
    if isinstance(value, bool):
        return int(value), 1.0
    return int(value or 0), 1.0


def config_from_knobs(cfg: dict) -> AdapterConfig:
    """AdapterConfig from the train-script globals (ALGO, PRESET, DIM, ...)."""
    dropout = cfg.get("DROPOUT", {}) or {}
    preset = cfg.get("PRESET", "attn-mlp")
    if cfg.get("TRAIN_TEXT_REFINER", False) and isinstance(preset, str):
        if "+text-refiner" not in preset and not preset.endswith(".toml"):
            preset = f"{preset}+text-refiner"
    algo_kwargs = dict(cfg.get("ALGO_KWARGS", {}) or {})
    post_rank, initial_s_p = _sora_post(cfg.get("SORA_POST_ORTHO", 0))
    raw_eva = cfg.get("EVA")
    eva = None if raw_eva is None or raw_eva is False else dict(raw_eva or {})
    return AdapterConfig(
        algo=cfg.get("ALGO", "lora"),
        preset=preset,
        dim=int(cfg.get("DIM", 32)),
        alpha=float(cfg.get("ALPHA", 32.0)),
        algo_kwargs=algo_kwargs,
        dora=bool(cfg.get("DORA", False)),
        dora_on_output=bool(algo_kwargs.pop("wd_on_out", True)),
        dora_detach=bool(cfg.get("DORA_DETACH", False)),
        nara=bool(cfg.get("NARA", False)),
        sora_post_rank=post_rank,
        sora_initial_s_p=initial_s_p,
        sora_pre_diag=bool(cfg.get("SORA_PRE_DIAG", False)),
        eva=eva,
        dropout=float(dropout.get("input", 0.0)),
        rank_dropout=float(dropout.get("rank", 0.0)),
        module_dropout=float(dropout.get("module", 0.0)),
        mode=cfg.get("ADAPTER_MODE", "bypass"),
        train_norms=bool(cfg.get("TRAIN_NORMS", False)),
        backend=(cfg.get("KERNELS", {}) or {}).get("lycoris", "auto"),
        lr_ratio_up=float(cfg.get("LORAPLUS_RATIO", 1.0)),
        seed=int(cfg.get("SEED", 0)),
    )


def inference_config(fields: dict, multiplier: float) -> AdapterConfig:
    """A saved configuration rebuilt for inference / export on any device: no dropout,
    the device's own kernel backend (a training-time backend pin does not carry over).
    """
    known = set(AdapterConfig.__dataclass_fields__)
    kept = {k: v for k, v in fields.items() if k in known}
    quiet = {"dropout": 0.0, "rank_dropout": 0.0, "module_dropout": 0.0}
    return AdapterConfig(
        **{**kept, **quiet, "backend": "auto", "multiplier": multiplier}
    )
