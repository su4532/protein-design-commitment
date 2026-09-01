#!/usr/bin/env python3
"""Build and clean-room verify the deterministic anonymous release archive."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = (
    "analysis", "scripts", "configs", "schemas", "fixtures", "tests",
    "generated_figures", "generated_tables",
)
TOP_LEVEL = (
    "README.md", "ANONYMOUS_GITHUB_SETUP.md", "ANONYMITY_AUDIT.json",
    "CODE_COMPLETENESS_MANIFEST.json", "pyproject.toml", "requirements-cpu.txt",
)


def include(path: Path) -> bool:
    return (
        path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def stage(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in TOP_LEVEL:
        shutil.copy2(ROOT / name, destination / name)
    for name in DIRECTORIES:
        source = ROOT / name
        assert source.is_dir(), f"missing release directory: {name}"
        for path in sorted(source.rglob("*")):
            if include(path):
                target = destination / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)


def write_checksums(staged: Path) -> None:
    rows = []
    for path in sorted(path for path in staged.rglob("*") if path.is_file()):
        relative = path.relative_to(staged).as_posix()
        if relative == "CHECKSUMS.sha256":
            continue
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    (staged / "CHECKSUMS.sha256").write_text("\n".join(rows) + "\n")


def write_zip(staged: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(item for item in staged.rglob("*") if item.is_file()):
            relative = path.relative_to(staged).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            executable = path.suffix == ".py" or path.parent.name == "scripts"
            info.external_attr = ((0o100755 if executable else 0o100644) << 16)
            handle.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def clean_room_verify(archive: Path) -> str:
    with (
        tempfile.TemporaryDirectory(prefix="anonymous-release-") as tmp_text,
        tempfile.TemporaryDirectory(prefix="anonymous-matplotlib-") as mpl_text,
    ):
        tmp = Path(tmp_text)
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            assert names and not any(".git" in Path(name).parts for name in names)
            handle.extractall(tmp)
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = mpl_text
        result = subprocess.run(
            [sys.executable, "scripts/verify_anonymous_release.py"],
            cwd=tmp, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
        )
        if result.returncode:
            raise SystemExit(result.stdout)
        return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    stage(args.staging)
    write_checksums(args.staging)
    write_zip(args.staging, args.archive)
    output = clean_room_verify(args.archive)
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    print(output, end="")
    print(f"archive: {args.archive} ({args.archive.stat().st_size} bytes)")
    print(f"archive sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
