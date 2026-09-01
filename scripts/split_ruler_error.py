#!/usr/bin/env python3
"""Command-line wrapper for :mod:`analysis.scoring.nested_split`."""

import _path  # noqa: F401

from analysis.scoring.nested_split import main


if __name__ == "__main__":
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
