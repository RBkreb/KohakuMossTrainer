"""Per-algorithm records (`Kind`), registered by kind name, module class and ALGO."""

from kohakumoss.adapters.kinds import lowrank, nara, other, rotation
from kohakumoss.adapters.kinds.nara import nara_network_algo, nara_record_of
from kohakumoss.adapters.kinds.record import (
    BY_ALGO,
    BY_MODULE,
    REGISTRY,
    Kind,
    lycoris_algo,
    record_of,
    register,
)

__all__ = [
    "BY_ALGO",
    "BY_MODULE",
    "REGISTRY",
    "Kind",
    "lowrank",
    "lycoris_algo",
    "nara",
    "nara_network_algo",
    "nara_record_of",
    "other",
    "record_of",
    "register",
    "rotation",
]
