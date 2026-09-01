#!/usr/bin/env python
"""Cluster the benchmark TARGET structures, so target-level intervals resample structures.

Several benchmark entries are the same or homologous proteins (two CbAgo crops,
two PD-L1 preparations, the Der f allergens). Bootstrapping over target *names*
would treat those as independent evidence. This clusters the target structures
themselves with Foldseek (TM-align, threshold 0.5) and writes
``{task_name: cluster_representative}`` for ``pool_targets.py`` to resample.

    python cluster_targets.py <targets.json> <complexa_repo> <out.json>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    targets_json, repo, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    foldseek = os.environ.get("FOLDSEEK_BIN", "foldseek")
    sel = json.load(open(targets_json))
    rows = {r["task"]: r for r in sel["selected_rows"]}
    keep = sys.argv[4].split(",") if len(sys.argv) > 4 else list(rows)

    work = tempfile.mkdtemp()
    ind = os.path.join(work, "in")
    os.makedirs(ind)
    name_map = {}
    for t in keep:
        r = rows.get(t)
        if not r:
            continue
        src = os.path.join(repo, r["structure"].lstrip("./"))
        if not os.path.exists(src):
            print(f"  missing structure for {t}: {src}", flush=True)
            continue
        stem = t.replace("/", "_")
        # Foldseek splits a multi-chain PDB into one entry per chain, which would
        # give a two-chain target two cluster memberships. Relabel every ATOM to
        # chain A so each target is a single structure.
        lines = [ln for ln in open(src).read().splitlines() if ln.startswith("ATOM")]
        with open(os.path.join(ind, stem + ".pdb"), "w") as fh:
            fh.write("\n".join(ln[:21] + "A" + ln[22:] for ln in lines) + "\nEND\n")
        name_map[stem] = t

    res = os.path.join(work, "clu")
    subprocess.run(
        [foldseek, "easy-cluster", ind, res, os.path.join(work, "tmp"),
         "--alignment-type", "1", "--tmscore-threshold", "0.5", "-c", "0.0", "-v", "0"],
        check=True, capture_output=True,
    )
    assign = {}
    with open(res + "_cluster.tsv") as fh:
        for line in fh:
            p = line.strip().split("\t")
            if len(p) >= 2:
                rep, mem = p[0].replace(".pdb", ""), p[1].replace(".pdb", "")
                assign[name_map.get(mem, mem)] = name_map.get(rep, rep)
    shutil.rmtree(work, ignore_errors=True)

    for t in keep:
        assign.setdefault(t, t)
    n_clusters = len(set(assign.values()))
    print(f"{len(assign)} targets -> {n_clusters} structural clusters", flush=True)
    for rep in sorted(set(assign.values())):
        members = sorted(t for t, r in assign.items() if r == rep)
        if len(members) > 1:
            print(f"  cluster {rep}: {members}", flush=True)
    with open(out_path, "w") as f:
        json.dump(assign, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
