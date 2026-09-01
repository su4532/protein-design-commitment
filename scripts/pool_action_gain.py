#!/usr/bin/env python3
"""Command-line wrapper for :mod:`analysis.action.pool_gain`."""

import _path  # noqa: F401

from analysis.action.pool_gain import main


if __name__ == "__main__":
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
