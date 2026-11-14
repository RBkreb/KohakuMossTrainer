"""Algorithm layer for scripts/train/moss.py: NaRA r16 on the attention + MLP linears.
Composed after configs/moss/_base/base_lora.py; see docs/guides/configs.md and
docs/internals/adapter-nara.md."""

ALGO = "nara"
PRESET = "attn-mlp"
DIM = 16
ALPHA = 16.0
LR = 1e-4
ALGO_KWARGS = {}
# LyCORIS dropout / rank_dropout / module_dropout (nara: rank dropout unsupported)
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
