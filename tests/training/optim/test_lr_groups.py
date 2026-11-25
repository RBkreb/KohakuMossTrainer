"""LR_GROUPS: name regex -> lr multiplier on the optimiser groups. The first matching
pattern wins, the scale stacks on the LoRA+ ratio in adapter runs, unmatched patterns
are logged, and the knob validation rejects bad patterns / multipliers."""

import logging
import sys
from pathlib import Path

import pytest
import torch

from kohakumoss.adapters.train.groups import build as build_adapter_groups
from kohakumoss.cli.config.validate import TRAIN, compose
from kohakumoss.training.knobs import validate
from kohakumoss.training.optim import build_param_groups

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[3]
LR = 1e-3


def _named():
    return [
        ("layers.0.ffn_up_gate.weight", torch.nn.Parameter(torch.zeros(4, 4))),
        ("layers.0.proj.q.weight", torch.nn.Parameter(torch.zeros(4, 4))),
        ("layers.0.ffn_down.weight", torch.nn.Parameter(torch.zeros(4, 4))),
    ]


def _lr_by_name(groups) -> dict[str, float]:
    return {name: g["lr"] for g in groups for name in g["names"]}


def test_full_groups_scale_by_name():
    groups = build_param_groups(_named(), LR, 0.0, "none", lr_groups={r"\.ffn_": 0.5})
    lr = _lr_by_name(groups)
    assert lr["layers.0.ffn_up_gate.weight"] == pytest.approx(5e-4)
    assert lr["layers.0.ffn_down.weight"] == pytest.approx(5e-4)
    assert lr["layers.0.proj.q.weight"] == pytest.approx(LR)
    assert len(groups) == 2


def test_first_matching_pattern_wins():
    rules = {r"layers\.": 0.5, r"\.ffn_": 0.1}
    groups = build_param_groups(_named(), LR, 0.0, "none", lr_groups=rules)
    assert all(g["lr"] == pytest.approx(5e-4) for g in groups)


def test_unmatched_pattern_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="kohakumoss.training.optim.groups"):
        build_param_groups(_named(), LR, 0.0, "none", lr_groups={r"\.missing_": 0.5})
    assert any(
        "matched no trainable parameter" in r.getMessage() for r in caplog.records
    )


def test_adapter_groups_stack_scale_on_loraplus():
    up = torch.nn.Parameter(torch.zeros(4, 4))
    down = torch.nn.Parameter(torch.zeros(4, 4))
    hyper = torch.nn.Parameter(torch.zeros(8, 8))
    entries = [
        ("blocks.0.lora_up.weight", "lora_up.weight", up, ("lora_up.weight",)),
        ("blocks.0.lora_down.weight", "lora_down.weight", down, ("lora_up.weight",)),
        ("nara_hyper.mapper.0.weight", "mapper.0.weight", hyper, ()),
    ]
    groups = build_adapter_groups(
        entries, LR, 0.0, "none", False, 4.0, {r"^nara_hyper\.": 0.1}
    )
    lr = _lr_by_name(groups)
    assert lr["blocks.0.lora_up.weight"] == pytest.approx(4e-3)
    assert lr["blocks.0.lora_down.weight"] == pytest.approx(LR)
    assert lr["nara_hyper.mapper.0.weight"] == pytest.approx(1e-4)


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")


def test_knob_validation(cpu_only):
    k = compose(ROOT, TRAIN, [], {"LR_GROUPS": {r"\.ffn_": 0.5}})
    assert [m for m in validate(k) if "LR_GROUPS" in m] == []
    bad = {**k, "LR_GROUPS": {"(": 0.5, r"\.ffn_": -1, "ok": True}}
    msgs = [m for m in validate(bad) if "LR_GROUPS" in m]
    assert len(msgs) == 3, msgs
    not_dict = {**k, "LR_GROUPS": ["layers.", 0.5]}
    assert len([m for m in validate(not_dict) if "LR_GROUPS" in m]) == 1
