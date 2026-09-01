#!/usr/bin/env python3
"""Recompute the compact manuscript-result checks from frozen fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _path  # noqa: F401

from analysis.golden import collect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    parser.add_argument("--out", type=Path, default=Path("generated_results.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(collect(args.fixtures), indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
