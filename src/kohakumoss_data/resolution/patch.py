"""`patch`: DiT-BlockSkip dynamic patch sampling (arXiv 2603.20755, Sec. 3.1).

Every sample trains at a fixed low resolution (smin x smin); the crop side grows
linearly with its pre-drawn sigma from smin to smax (rounded to the latent factor),
so high-noise draws see global structure and low-noise draws see fine detail. The
sigma is drawn here (host, seeded per occurrence) from the run's timestep pmf and
carried to the TrainStep, which uses it instead of sampling its own.
"""

import numpy as np

from kohakumoss_data.registry import PLANNERS
from kohakumoss_data.resolution.base import Planner
from kohakumoss_data.seeds import rng_of
from kohakumoss_data.types import PlanItem


@PLANNERS.register("patch")
class PatchPlanner(Planner):
    """Square smin targets; `sigma_grid` / `sigma_cdf` are the timestep sampler's
    shifted sigma grid and its cumulative pmf (injected by the training entry)."""

    def __init__(
        self,
        *args,
        smin: int = 512,
        smax: int = 1024,
        sigma_grid=None,
        sigma_cdf=None,
        **kwargs,
    ):
        self.smin = int(smin)
        self.smax = int(smax)
        super().__init__(*args, **kwargs)
        lf = self.geometry.latent_factor
        if smin % lf or smax % lf or not (lf <= smin <= smax):
            raise ValueError(
                f"patch smin/smax must be multiples of {lf} with smin <= smax, "
                f"got {smin}/{smax}"
            )
        if sigma_grid is None or sigma_cdf is None:
            raise ValueError(
                "RESOLUTION_MODE='patch' needs the timestep sigma table; run through "
                "the training entry (it injects sigma_grid/sigma_cdf)"
            )
        self.sigma_grid = np.asarray(sigma_grid, dtype=np.float32)
        self.sigma_cdf = np.asarray(sigma_cdf, dtype=np.float32)

    def target(self, row_id: int) -> tuple[int, int]:
        return self.smin, self.smin

    def item(self, row_id: int, epoch: int, position: int, target) -> PlanItem:
        w, h = int(self.catalog.w[row_id]), int(self.catalog.h[row_id])
        rng = rng_of("item", self.seed, epoch, position, row_id)
        u = float(rng.random())
        idx = min(int(np.searchsorted(self.sigma_cdf, u)), len(self.sigma_grid) - 1)
        sigma = float(self.sigma_grid[idx])
        lf = self.geometry.latent_factor
        side = self.smin + sigma * (self.smax - self.smin)
        side = min(self.smax, max(self.smin, round(side / lf) * lf))
        frac = side / self.smax
        cw, ch = self.geometry.box_size(w, h, target)
        pw, ph = max(1, round(cw * frac)), max(1, round(ch * frac))
        left = int(rng.integers(0, w - pw + 1))
        top = int(rng.integers(0, h - ph + 1))
        box = (left, top, left + pw, top + ph)
        flip = self._flip(rng)
        variant = int(rng.integers(self.caption_variants))
        return PlanItem(row_id, tuple(target), box, flip, variant, sigma=sigma)
