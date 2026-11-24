"""RANK_CHECK: option checks, factor selection, the loraspectra table / JSON parsers
and the one-shot callback (tool lookup, step gating, report write)."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from kohakumoss.training import rank_check
from kohakumoss.training.rank_check import (
    DEFAULTS,
    RankCheck,
    factor_tensors,
    guidance_of,
    option_problems,
    options,
    table_of,
)

pytestmark = [pytest.mark.cpu, pytest.mark.budget(10)]

TABLE = """\
LAYER                                          RANK    R_EFF R_STABLE   K_99%   K_MP   NORM_F STATUS
----------------------------------------------------------------------------------------------------
lora_unet_layers_0_proj_k                      32    25.91     3.40      30      5   0.3657 OPTIMAL
lora_unet_layers_0_proj_o                      32    19.83     1.72      16      7   0.3888 OVER_PARAMETERIZED
  [WARNING] DORA_EXTREME_DISPERSION (lora_unet_layers_41_proj_k): extreme
"""
SUMMARY = {
    "layer_count": 2,
    "mean_rank_at_99_energy": 23.0,
    "mean_rank_marchenko_pastur": 6.0,
    "mean_stable_rank": 2.56,
}


def _parts(state):
    return SimpleNamespace(trainable=SimpleNamespace(state_dict=lambda: state))


def _k(**over):
    k = {
        "RANK_CHECK": True,
        "PARALLEL": "none",
        "STEPS": 5,
        "DIM": 32,
        "ALGO": "lora",
    }
    k.update(over)
    return k


def _state():
    return {
        "lora_unet_x.lora_down.weight": torch.randn(4, 8),
        "lora_unet_x.lora_up.weight": torch.randn(8, 4),
    }


def test_options_and_problems():
    assert options({"RANK_CHECK": False}) is None
    assert options({"RANK_CHECK": True}) == DEFAULTS
    assert option_problems({"RANK_CHECK": True}) == []
    p = option_problems({"RANK_CHECK": {"step": 0, "nope": 1, "keep": 1, "timeout": 0}})
    assert len(p) == 4


def test_factor_tensors_filters():
    state = {
        "a.lora_down.weight": torch.zeros(2, 3),
        "a.lora_up.weight": torch.zeros(3, 2),
        "a.alpha": torch.tensor(2.0),
        "a.dora_scale": torch.zeros(3, 1),
        "a.sora_diag": torch.ones(3),
        "nara_hyper.fourier_w": torch.ones(1),
    }
    assert sorted(factor_tensors(state)) == [
        "a.alpha",
        "a.dora_scale",
        "a.lora_down.weight",
        "a.lora_up.weight",
    ]


def test_table_and_guidance_parsers():
    layers, warnings = table_of(TABLE)
    assert len(layers) == 2
    assert layers[0]["layer"] == "lora_unet_layers_0_proj_k"
    assert layers[0]["k99"] == 30
    assert layers[1]["status"] == "OVER_PARAMETERIZED"
    assert len(warnings) == 1
    g = guidance_of(SUMMARY, layers, 32)
    assert g["energy_use"] == pytest.approx(23 / 32)
    assert g["mp_share"] == pytest.approx(6 / 32)
    assert g["status_counts"] == {"OPTIMAL": 1, "OVER_PARAMETERIZED": 1}


def test_callback_one_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_check, "tool_path", lambda: "loraspectra-fake")
    calls = []

    def fake_run(exe, path, timeout, *args):
        calls.append(args)
        out = json.dumps(SUMMARY) if "--json" in args else TABLE
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(rank_check, "run_tool", fake_run)
    cb = RankCheck(_k(RANK_CHECK={"step": 2}), _parts(_state()), tmp_path, True)
    assert cb.every == 1
    assert cb.on_step_end(SimpleNamespace(step=1), None) is False
    assert cb.on_step_end(SimpleNamespace(step=2), None) is True
    report = json.loads((tmp_path / "rank_check.json").read_text())
    assert report["step"] == 2
    assert report["layers"][0]["status"] == "OPTIMAL"
    assert report["guidance"]["energy_use"] == pytest.approx(23 / 32)
    assert report["adapter_file"] is None
    assert calls == [("--json",), ()]
    assert cb.on_step_end(SimpleNamespace(step=3), None) is True
    assert not (tmp_path / "rank_check" / "step_00000002.safetensors").exists()
    assert not (tmp_path / "rank_check").exists()


def test_callback_keeps_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_check, "tool_path", lambda: "fake")

    def fake_run(exe, path, timeout, *args):
        out = json.dumps(SUMMARY) if "--json" in args else TABLE
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(rank_check, "run_tool", fake_run)
    cb = RankCheck(
        _k(RANK_CHECK={"step": 1, "keep": True, "layers": False}),
        _parts(_state()),
        tmp_path,
        True,
    )
    cb.on_step_end(SimpleNamespace(step=1), None)
    kept = tmp_path / "rank_check" / "step_00000001.safetensors"
    assert kept.exists()
    report = json.loads((tmp_path / "rank_check.json").read_text())
    assert report["adapter_file"] == str(kept)
    assert report["layers"] == []


def test_callback_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_check, "tool_path", lambda: None)
    assert RankCheck(_k(), _parts(_state()), tmp_path, True).every == 0
    monkeypatch.setattr(rank_check, "tool_path", lambda: "fake")
    assert RankCheck(_k(PARALLEL="fsdp"), _parts(_state()), tmp_path, True).every == 0
    assert (
        RankCheck(
            _k(STEPS=5, RANK_CHECK={"step": 10}), _parts(_state()), tmp_path, True
        ).every
        == 0
    )
    assert RankCheck(_k(ALGO="lokr"), _parts(_state()), tmp_path, True).every == 0
    assert RankCheck(_k(), _parts(_state()), tmp_path, False).every == 0


def test_callback_resume_past_step(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_check, "tool_path", lambda: "fake")
    cb = RankCheck(_k(RANK_CHECK={"step": 2}), _parts(_state()), tmp_path, True)
    assert cb.on_step_end(SimpleNamespace(step=3), None) is True
    assert not (tmp_path / "rank_check.json").exists()


def test_callback_without_lora_factors(tmp_path, monkeypatch):
    monkeypatch.setattr(rank_check, "tool_path", lambda: "fake")
    monkeypatch.setattr(rank_check, "run_tool", lambda *a, **kw: None)
    cb = RankCheck(
        _k(RANK_CHECK={"step": 1}), _parts({"x": torch.ones(3)}), tmp_path, True
    )
    assert cb.on_step_end(SimpleNamespace(step=1), None) is True
    assert not (tmp_path / "rank_check.json").exists()
