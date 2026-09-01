"""The capture grid, as a single place every stage reads it from.

The source analysis (the recorded source analysis) hard-coded its seven capture
steps in six separate modules. This experiment adds 150, 250 and 350 and
re-derives on the combined ten-step grid, and it also has to reproduce the
seven-step derivation on exactly the same re-analysed cells, so the grid has to
be a run-time choice rather than a literal.

``CAPTURE_STEPS`` in the environment overrides the default, e.g.::

    CAPTURE_STEPS=10,25,50,75,100,200,300   # the parent's grid

Nothing else about the protocol is configurable here on purpose: roots,
children, completions, seeds, sampler and AlphaFold settings all stay exactly
where the parent left them.
"""

from __future__ import annotations

import os

#: the source analysis's grid
PARENT_STEPS: tuple[int, ...] = (10, 25, 50, 75, 100, 200, 300)

#: the three steps this experiment adds
NEW_STEPS: tuple[int, ...] = (150, 250, 350)

#: the combined grid this experiment re-derives on
FULL_STEPS: tuple[int, ...] = tuple(sorted(PARENT_STEPS + NEW_STEPS))


def steps() -> tuple[int, ...]:
    """The capture grid for this process."""
    raw = os.environ.get("CAPTURE_STEPS", "").strip()
    if not raw:
        return FULL_STEPS
    return tuple(sorted(int(x) for x in raw.replace(",", " ").split()))
