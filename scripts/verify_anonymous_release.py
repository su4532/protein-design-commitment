#!/usr/bin/env python3
"""Offline completeness, golden-output, checksum, and anonymity verification."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIGURES = (
    "fig1_schedule", "fig2_readout", "figA1_commitment", "figA2_consequence",
    "figA3_grid", "figA4_cv", "fig2_readability", "fig3_robustness", "fig4_schedule",
)
EXPECTED_TABLES = ("clocks", "order-null", "gap-scale", "cv", "pairwise", "oa", "thresholds", "boot")
CPU_MODULES = (
    "analysis.commitment.boot", "analysis.commitment.grid", "analysis.commitment.metrics",
    "analysis.commitment.nested", "analysis.commitment.probes", "analysis.commitment.variance",
    "analysis.action.compare_grids", "analysis.action.features", "analysis.action.objective",
    "analysis.action.pool_gain", "analysis.action.ridge", "analysis.action.score_curves",
    "analysis.scoring.compare", "analysis.scoring.correct_target", "analysis.scoring.nested_split",
    "analysis.scoring.pool_quant", "analysis.scoring.quantisation",
    "analysis.scoring.readability_corrected", "analysis.scoring.recompute_objective",
    "analysis.scoring.ruler", "analysis.cell_analysis", "analysis.clock_remap",
    "analysis.designability", "analysis.fold_clustering", "analysis.golden",
    "analysis.grid_comparison", "analysis.joint_bootstrap", "analysis.manuscript_figures",
    "analysis.manuscript_tables", "analysis.mlp_retune", "analysis.opportunity",
    "analysis.order_simulations", "analysis.pooling", "analysis.spread",
    "analysis.spread_raw", "analysis.transfer",
)


def _run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode:
        raise AssertionError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_completeness() -> dict[str, int]:
    manifest = _json(ROOT / "CODE_COMPLETENESS_MANIFEST.json")
    coverage = _json(ROOT / "configs/paper_coverage.json")
    claims = _json(ROOT / "fixtures/manuscript_claims.json")["claims"]
    assert manifest["version"] == 1
    entries = manifest["entries"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids)), "duplicate completeness IDs"
    assert set(ids) == set(coverage["expected_ids"]), "coverage and manifest IDs differ"
    required = {
        "id", "kind", "title", "entrypoint", "configuration",
        "source_experiment", "required_inputs", "verification_fixture",
    }
    aliases = set(manifest["source_experiment_aliases"])
    counts: dict[str, int] = {}
    for entry in entries:
        missing = required.difference(entry)
        assert not missing, f"{entry.get('id')}: missing fields {sorted(missing)}"
        assert entry["source_experiment"] in aliases, f"{entry['id']}: unknown source alias"
        assert (ROOT / entry["entrypoint"]).is_file(), f"{entry['id']}: missing entrypoint"
        assert (ROOT / entry["configuration"]).is_file(), f"{entry['id']}: missing configuration"
        fixture = entry["verification_fixture"].split("#", 1)[0]
        assert (ROOT / fixture).is_file(), f"{entry['id']}: missing verification fixture {fixture}"
        counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
    assert counts == coverage["counts"]
    claim_ids = {"result:" + row["id"] for row in claims}
    result_ids = {entry["id"] for entry in entries if entry["kind"] == "result"}
    assert result_ids == claim_ids, "reported-result coverage is incomplete"
    assert len(EXPECTED_FIGURES) == counts["figure"]
    assert len(EXPECTED_TABLES) == counts["table"]
    return counts


def verify_imports_and_tests() -> None:
    sys.path.insert(0, str(ROOT))
    importlib.import_module("analysis")
    for name in CPU_MODULES:
        importlib.import_module(name)
    _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])


def verify_golden_outputs() -> None:
    expected = _json(ROOT / "fixtures/golden_results.json")
    expected_tables = _json(ROOT / "fixtures/manuscript_tables.json")
    with tempfile.TemporaryDirectory(prefix="denoising-verify-") as tmp_text:
        tmp = Path(tmp_text)
        compact = tmp / "compact.json"
        _run([
            sys.executable, "scripts/regenerate_compact_results.py",
            "--fixtures", "fixtures", "--out", str(compact),
        ])
        assert _json(compact) == expected, "compact golden-output mismatch"

        table_dir = tmp / "tables"
        _run([
            sys.executable, "scripts/make_manuscript_tables.py",
            "--fixtures", "fixtures", "--out", str(table_dir),
        ])
        assert _json(table_dir / "manuscript_tables.json") == expected_tables
        assert {path.stem for path in table_dir.glob("*.csv")} == set(EXPECTED_TABLES)

        figure_dir = tmp / "figures"
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env["SOURCE_DATE_EPOCH"] = "0"
        _run([
            sys.executable, "scripts/make_manuscript_figures.py",
            "--data", "fixtures/figure_data.json", "--out", str(figure_dir),
        ], env=env)
        observed_pdf = {path.stem for path in figure_dir.glob("*.pdf")}
        observed_png = {path.stem for path in figure_dir.glob("*.png")}
        assert observed_pdf == set(EXPECTED_FIGURES), "manuscript PDF figure set differs"
        assert observed_png == set(EXPECTED_FIGURES), "manuscript PNG figure set differs"
        for name in EXPECTED_FIGURES:
            assert (figure_dir / f"{name}.pdf").read_bytes().startswith(b"%PDF")
            assert (figure_dir / f"{name}.png").read_bytes().startswith(b"\x89PNG")


def _release_files() -> list[Path]:
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.name != "CHECKSUMS.sha256"
    )


def verify_checksums() -> None:
    checksum_path = ROOT / "CHECKSUMS.sha256"
    assert checksum_path.is_file(), "CHECKSUMS.sha256 is missing"
    recorded: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        recorded[relative] = digest
    expected_paths = {path.relative_to(ROOT).as_posix() for path in _release_files()}
    assert set(recorded) == expected_paths, "checksum path set differs from release contents"
    for relative, expected in recorded.items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, f"checksum mismatch: {relative}"


def anonymity_findings(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    patterns = {
        "internal experiment identifier": re.compile(r"\b" + "exp" + r"_[a-z0-9]{8,}\b", re.I),
        "absolute research path": re.compile(r"/(?:" + "srv|mnt|home|Users|opt/" + "sili" + "co" + r")/", re.I),
        "scheduler output identifier": re.compile(r"(?:" + "slurm" + r"[-_]|job[-_ ]?id\D{0,3})\d{4,}", re.I),
        "tracking URL": re.compile(r"(?:wand" + "b" + r"\.ai|api\.wand" + "b" + r"\.)", re.I),
        "platform identifier": re.compile(r"\b" + "sili" + "co" + r"\b", re.I),
        "approval provenance": re.compile(r"researcher\s+" + "approval", re.I),
        "unrelated project reference": re.compile(r"\bprun" + "ing\b", re.I),
    }
    email = re.compile(rb"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
    neutral_emails = {b"anonymous@users.noreply.github.com"}
    github = re.compile(rb"https?://github\.com/([A-Z0-9_.-]+)/([A-Z0-9_.-]+)", re.I)
    allowed_github = {(b"nvidia-digital-bio", b"proteina-complexa")}

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if ".git" in path.parts:
            findings.append({"path": relative, "kind": "Git metadata"})
            continue
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        for label, regex in patterns.items():
            if regex.search(relative):
                findings.append({"path": relative, "kind": label})
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        for label, regex in patterns.items():
            if regex.search(text):
                findings.append({"path": relative, "kind": label})
        for match in email.findall(data):
            if match.lower() not in neutral_emails:
                findings.append({"path": relative, "kind": "email address"})
        for owner, repo in github.findall(data):
            if (owner.lower(), repo.lower().rstrip(b"./")) not in allowed_github:
                findings.append({"path": relative, "kind": "unapproved repository URL"})
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}:
            findings.append({"path": relative, "kind": "nested archive"})
    unique = {(row["path"], row["kind"]): row for row in findings}
    return [unique[key] for key in sorted(unique)]


def verify_anonymity() -> None:
    audit = _json(ROOT / "ANONYMITY_AUDIT.json")
    assert audit["status"] == "pass" and audit["finding_count"] == 0
    findings = anonymity_findings(ROOT)
    assert not findings, "anonymity findings: " + json.dumps(findings[:20], indent=2)


def verify_archive_if_present() -> None:
    archives = list(ROOT.parent.glob("anonymous-code.zip"))
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            assert names and all(".git" not in Path(name).parts for name in names)


def main() -> int:
    counts = validate_completeness()
    print(f"completeness: {sum(counts.values())} entries {counts}")
    verify_imports_and_tests()
    print("unit tests and imports: pass")
    verify_golden_outputs()
    print("compact results, eight tables, and nine figures: pass")
    verify_checksums()
    print("checksums: pass")
    verify_anonymity()
    print("anonymity: 0 findings")
    verify_archive_if_present()
    print("anonymous release verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
