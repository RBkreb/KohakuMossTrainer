"""The noise level of the forward being run: (sigma, seq) installed by the trainer's
begin_step or the sampler's on_sigma hook, read by every NaRA module."""

from torch import Tensor

from kohakumoss.types.seqinfo import SeqInfo

_context: tuple[Tensor, SeqInfo] | None = None


def set_context(sigma: Tensor, seq: SeqInfo) -> None:
    """Install the step's per-sample noise levels and packed layout."""
    global _context
    _context = (sigma, seq)


def get_context() -> tuple[Tensor, SeqInfo]:
    """The installed (sigma, seq); RuntimeError when no step installed one."""
    if _context is None:
        raise RuntimeError(
            "NaRA needs the step's noise level: training installs it in begin_step; "
            "sampling passes on_sigma to sample_latents"
        )
    return _context


def clear_context() -> None:
    global _context
    _context = None


def saved_context() -> tuple[Tensor, SeqInfo] | None:
    """The installed context (for a save / clear / restore around a micro-batch)."""
    return _context


def restore_context(state: tuple[Tensor, SeqInfo] | None) -> None:
    global _context
    _context = state
