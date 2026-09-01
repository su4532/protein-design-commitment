# Denoising dynamics in protein design

This repository contains the maintained analysis and compact verification package for a study of when protein-design qualities settle during a denoising trajectory. It includes the study-owned capture hooks, outcome analysis, probe fitting, pooled statistics, sensitivity analyses, action-timing estimators, and manuscript figure and table builders.

The default verification path is CPU-only and uses frozen anonymized fixtures. It performs no model inference, structure prediction, sequence redesign, activation harvesting, or network access.

## Verify the release

Use Python 3.11 or later in an environment with the packages in `requirements-cpu.txt`:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements-cpu.txt
python scripts/verify_anonymous_release.py
```

The installation command is an offline example. A normal package index may be used to prepare the environment before an anonymous review. The verification command itself does not use the network.

A passing run verifies all of the following:

- the `analysis` package and every CPU analysis module import;
- `CODE_COMPLETENESS_MANIFEST.json` covers every manuscript method, figure, table, and all 1,009 quantitative or categorical claim records;
- unit tests and compact golden-output comparisons pass exactly;
- all eight manuscript tables and all nine figures regenerate from the frozen fixtures;
- every packaged file matches `CHECKSUMS.sha256`;
- filenames, text, binary metadata, archive structure, and Git metadata have zero anonymity findings.

## Repository layout

- `analysis/`: importable estimators and analysis modules.
- `scripts/`: command-line entrypoints for capture, scoring, statistics, sensitivities, and manuscript outputs.
- `configs/`: fixed study, capture, analysis, action, sensitivity, and coverage configurations.
- `schemas/`: input and completeness-manifest schemas.
- `fixtures/`: anonymized aggregate records used by offline verification.
- `generated_figures/`: nine regenerated manuscript figures in PDF and PNG form.
- `generated_tables/`: eight regenerated manuscript tables in CSV and JSON form.
- `tests/`: estimator unit tests and frozen-output tests.
- `CODE_COMPLETENESS_MANIFEST.json`: paper-to-code traceability, including neutral source-run aliases.

## Full pipeline entrypoints

The raw-data stages require the data and external software described below. Paths shown as angle-bracketed values are supplied by the user.

### Capture and outcome preparation

```bash
python scripts/run_capture.py <capture overrides>
python scripts/cluster_folds.py <fold records> <cluster output>
python scripts/score_designability.py <completion structures> <score output>
python scripts/analyze_cell.py <trajectory shard> <outcome records> <cell output>
```

`run_capture.py` calls the public model interface and stores lineage, coordinates, activations, and completion outcomes. `cluster_folds.py` assigns fold identities. `score_designability.py` reads redesign-and-refold outputs. `analyze_cell.py` performs the continuous nested ANOVA, categorical Gini decomposition, commitment calculation, and nested probe evaluation.

### Target pooling and capture-grid analyses

```bash
python scripts/pool_targets.py <per-target directory> <pooled output> --target-clusters <clusters.json>
python scripts/loco_transfer.py <per-target directory> <transfer output> --target-clusters <clusters.json>
python scripts/compare_capture_grids.py <comparison inputs> <comparison output>
python scripts/run_order_simulations.py <pooled input> <simulation output> --n-sim 20000 --seed 42
```

These entrypoints produce cluster-weighted pooled curves, target-level and cluster-level settling-time spread, leave-one-cluster-out transfer, seven- versus ten-step grid sensitivity, and the shared-order simulation nulls.

### Scoring and ruler sensitivities

```bash
python scripts/score_ruler.py <per-cell records> <ruler output>
python scripts/split_ruler_error.py <per-cell records> <error output>
python scripts/correct_scores.py <per-cell records> <corrected output>
python scripts/continuous_fold_ruler.py <fold structures> <continuous-ruler output>
python scripts/pool_fold_ruler.py <per-target inputs> <pooled output>
python scripts/recompute_readability.py <corrected cells> <readability output>
python scripts/recompute_action_objective.py <corrected cells> <objective output>
python scripts/compare_score_sensitivities.py <comparison inputs> <comparison output>
```

The sensitivity path separates nested scoring error from model variability, compares categorical and continuous fold rulers, and recomputes readability and action objectives on corrected scores.

### Action timing and adaptive opportunity

```bash
python scripts/fit_action_curves.py <activation and outcome inputs> <curve output>
python scripts/pool_action_gain.py <target curves> <gain output> --target-clusters <clusters.json>
python scripts/compare_action_grids.py <seven-step result> <ten-step result> <comparison output>
python scripts/run_adaptive_opportunity.py --config configs/opportunity.yaml
python scripts/run_joint_opportunity_bootstrap.py --config configs/opportunity.yaml
```

The action estimators parent-center predictions and outcomes, hold out complete roots, reselect the deployed comparator inside each resample, and report available opportunity (`O`) and readable opportunity (`A`). The joint bootstrap resamples structural clusters, root rotations, and saved design replicates together.

### Other controls and manuscript outputs

```bash
python scripts/run_clock_remap.py <per-target directory> <output> --config <resolved capture config> --clusters <clusters.json>
python scripts/retune_mlp.py <per-cell inputs> <output>
python scripts/regenerate_compact_results.py --fixtures fixtures --out generated_results.json
python scripts/make_manuscript_tables.py --fixtures fixtures --out generated_tables
python scripts/make_manuscript_figures.py --data fixtures/figure_data.json --out generated_figures
```

Clock remapping re-interpolates the saved commitment curves on the backbone and local-latent schedules. The MLP control uses the same root-blocked nested selection as the linear readout. The final three commands reproduce the compact reported values, eight tables, and nine figures without raw data.

## Data boundary and schemas

Raw trajectories, activation tensors, completion structures, structure-prediction caches, and redesign caches are data products and are intentionally not packaged.

### Trajectory shards

`schemas/trajectory_shard.schema.json` defines the lineage and capture metadata. A shard is produced by `scripts/run_capture.py` from target structures and the fixed capture configuration. Each captured state records a target identifier, capture step, root, child, completion index, and paths or keys for model-owned arrays. Large arrays remain in the external data store.

The capture stage uses the public Complexa repository at:

- repository: `https://github.com/NVIDIA-Digital-Bio/proteina-complexa`
- commit: `916eaaedce5b`

The external repository, model checkpoint, structure predictor, sequence-redesign model, Foldseek installation, and their caches are not copied here. `configs/capture.json` records the sampler shape and the pinned public dependency.

### Per-target analysis records

`schemas/per_target_analysis.schema.json` defines the target, capture-step, outcome, decomposition, probe, and uncertainty fields consumed by the pooling and sensitivity scripts. These records are produced by the cell-analysis and pooling entrypoints from trajectory shards, outcomes, fold identities, target structural clusters, and activation tensors.

### Frozen fixtures

The fixtures contain aggregate arrays and result tables only. They retain the values needed to compare consolidated code against settled outputs, regenerate manuscript numbers, and rebuild every figure and table. They do not contain raw sequences, raw trajectories, activation tensors, structures, personal metadata, filesystem paths, run identifiers, credentials, or tracking links.

`fixtures/source_records.json` gives each settled source run a neutral alias and records the SHA-256 digest of the compact source result used for comparison. The private source-to-alias ledger is intentionally outside the anonymous archive.

## Completeness and provenance

`CODE_COMPLETENESS_MANIFEST.json` is validated against `configs/paper_coverage.json`. Each entry identifies a method, figure, table, or reported result and records:

- the maintained repository entrypoint;
- the fixed configuration;
- the neutral source-run alias;
- the required raw or aggregate inputs;
- the verification fixture or generated output.

`fixtures/manuscript_claims.json` is the anonymized claim-level ledger. The compact golden checks in `fixtures/golden_results.json` span the seven settled source-run families, while the claim ledger provides exhaustive paper-result coverage.

## Anonymous repository handoff

`ANONYMOUS_GITHUB_SETUP.md` gives the exact fresh-history, private-repository, pull-request, and read-only mirror settings. The release archive contains no `.git` directory and no license file.
