#!/usr/bin/env python
"""Readability with the ruler's error taken out of the probe's target.

The source analysis already disattenuates the probe: it divides the observed
correlation by ``sqrt(reliability)`` where

    reliability = S / (S + sigma2_within / k),   S = sigma2_root + sigma2_child

so the finite-``k`` noise in the child mean is accounted for. That correction is
complete only if every bit of the ruler's error lives in the within-child term.
``nested_split.py`` says it does not: at the late capture steps most of the
AlphaFold reading error is shared by the eight completions of a child, so it sits
in ``sigma2_child`` and the parent's ``reliability`` counts it as signal. The
probe is then asked to predict a target that is partly unpredictable noise, and
its ``r2_corrected`` is still attenuated.

Redo the disattenuation against the ruler-free signal:

    reliability' = (S - eps_root - eps_child) / (S + sigma2_within / k)

and put the result back on the total-variance scale with the *corrected*
ceiling. ``r_observed`` is the parent's own out-of-fold correlation; no probe is
refitted, because removing noise from a target does not change the fit, only the
scale the fit is scored on.

Fold identity has zero ruler error by construction, so its numbers pass through
unchanged and act as the internal control: if the readability ordering is a ruler
artefact, the gap between fold identity and interface quality has to shrink here.

    readability_corrected.py <corrected_dir> <out_json> [--per-target2 DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
