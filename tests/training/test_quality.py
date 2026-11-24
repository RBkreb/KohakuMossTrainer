"""QUALITY_CHECK: SVIR / EK of synthetic Wdown factors, the option checks, the after_save
row write and the callback chain."""

import json

import pytest
import torch

from kohakumoss.training.quality import (
    DEFAULTS,
    after_save_quality,
    chain_after_save,
    inspect_state,
    layer_metrics,
    option_problems,
    options,
)

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]

SUFFIX = ".lora_down.weight"


def _uniform(n_layers=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        f"lora_unet_layers_{i}_proj_k{SUFFIX}": torch.rand((32, 1024), generator=g)
        - 0.5
        for i in range(n_layers)
    }


def _spectrum(shape, values):
    """A matrix with the given singular values (orthonormal U, V)."""
    g = torch.Generator().manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn((shape[0], len(values)), generator=g))
    v, _ = torch.linalg.qr(torch.randn((shape[1], len(values)), generator=g))
    return (u * torch.tensor(values)) @ v.T


def test_uniform_layers_are_healthy():
    s = inspect_state(_uniform(), dict(DEFAULTS))
    assert s["layers"] == 3
    assert s["ek_median"] < -1.0
    assert s["ek_frac_pos"] == 0.0
    assert s["svir_mean"] == 0.0
    assert s["layers_with_intrusions"] == 0
    assert s["watch"] == []


def test_heavy_tail_turns_ek_positive():
    state = _uniform(1)
    name = next(iter(state))
    state[name][0, 7] = 8.0
    s = inspect_state(state, dict(DEFAULTS))
    assert s["ek_median"] > 0
    assert s["watch"] == ["ek"]
    assert s["top"][0]["layer"] == name[: -len(SUFFIX)]
    assert s["top"][0]["intrusions"] == 0


def test_spectral_spike_flags_svir():
    w = _spectrum((32, 1024), [4.0] + [1.0] * 31)
    m = layer_metrics(w, 3.0, 3.5)
    assert m["intrusions"] == 1
    assert m["r_eff"] == pytest.approx(47 / 16)
    assert m["svir"] == pytest.approx(16 / 47)
    s = inspect_state({f"x{SUFFIX}": w}, dict(DEFAULTS))
    assert "svir" in s["watch"]


def test_low_rank_uses_its_own_tau():
    w = _spectrum((4, 256), [3.2, 1.0, 1.0, 1.0])
    assert layer_metrics(w, 3.0, 3.5)["intrusions"] == 0
    assert layer_metrics(w, 3.0, 3.0)["intrusions"] == 1


def test_non_lora_factors_are_skipped():
    a = torch.rand((32, 1024))
    assert inspect_state({"lora_unet_x.hada_w1_a": a}, dict(DEFAULTS)) is None
    assert layer_metrics(torch.zeros(32, 64), 3.0, 3.5) is None
    assert layer_metrics(torch.rand(4, 4, 3, 3), 3.0, 3.5) is None


def test_options_and_problems():
    assert options({"QUALITY_CHECK": False}) is None
    assert options({"QUALITY_CHECK": True}) == DEFAULTS
    assert options({"QUALITY_CHECK": {"top": 0}})["top"] == 0
    assert option_problems({"QUALITY_CHECK": True}) == []
    p = option_problems({"QUALITY_CHECK": {"tau": -1, "bogus": 1, "ema": 1, "top": -2}})
    assert len(p) == 4


def test_after_save_rows(tmp_path):
    k = {"QUALITY_CHECK": {"ema": True}}
    state = {"trainable": _uniform(2), "ema": _uniform(2, seed=1)}
    after_save_quality(k, tmp_path, True)(100, tmp_path / "step_00000100", state)
    path = tmp_path / "quality.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["source"] for r in rows] == ["trainable", "ema"]
    assert all(r["step"] == 100 and r["kind"] == "checkpoint" for r in rows)
    after_save_quality(k, tmp_path, False)(200, None, state)
    assert len(path.read_text().splitlines()) == 2


def test_chain_after_save_order():
    seen = []
    cb = chain_after_save(
        lambda step, path, parts=None: seen.append(("a", step)),
        lambda step, path, parts=None: seen.append(("b", step)),
    )
    cb(7, None, {})
    assert seen == [("a", 7), ("b", 7)]
