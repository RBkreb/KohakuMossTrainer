"""Algorithm layer for scripts/train/moss.py: LoKr-NaRA, the NARA switch on ALGO=lokr.
Factor 8 keeps w2 factorized for the shared core on every MOSS dimension (a layer's
w2 rank needs DIM < dim / factor / 2; see docs/internals/adapter-nara.md). Composed
after configs/moss/_base/base_lora.py."""

ALGO = "lokr"
NARA = True
DIM = 16
ALPHA = 16.0
ALGO_KWARGS = {
    "factor": 8,
    "decompose_both": False,
    "full_matrix": False,
    "use_tucker": False,
    "rank_dropout_scale": False,
}
LR = 2e-4
# LyCORIS dropout / rank_dropout / module_dropout
DROPOUT = {"input": 0.0, "rank": 0.0, "module": 0.0}
