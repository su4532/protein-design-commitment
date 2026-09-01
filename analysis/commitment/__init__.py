"""Commitment-curve measurement for the Complexa denoiser.

Measures *when during denoising* each scored design quality becomes determined,
and how much of what is determined a linear readout of the denoiser's own
activations can see.

The package has two halves:

- :mod:`commitment.capture` — the GPU harvest. A **no-search** capture: plain
  trajectories from noise (no beam, no reward filtering), a clean three-level
  nesting of ``root -> child -> completion``, activation capture at the capture
  step, and ``k`` independent AF2-folded completions per child.
- :mod:`commitment.variance` / :mod:`commitment.probes` — the CPU analysis. A
  three-level variance decomposition with a split-half correction for finite
  ``k``, and a root-blocked ridge-probe ladder scored on the same
  variance-explained scale as the decomposition's ceiling.

Complexa (``proteinfoundation``) imports are deferred into the functions that
need them, so the analysis half imports and runs on CPU without Complexa
present.
"""

__all__ = ["capture", "variance", "probes"]
