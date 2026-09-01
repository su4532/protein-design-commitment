#!/usr/bin/env python
"""Hydra entrypoint for the no-search commitment capture.

Runs against Complexa's own config tree, so every sampler setting (noise
schedule, guidance, self-conditioning, reward model, AF2 protocol) is the one the
search actually uses::

    python -m run_capture \
        --config-path "$COMPLEXA/configs" --config-name search_binder_local_pipeline \
        ++capture.mode=capture \
        ++capture.target_name=33_TrkA ++capture.capture_step=100 \
        ++capture.out_dir=$ART/harvest \
        ++generation.task_name=33_TrkA \
        ++generation.dataloader.dataset.nres.nsamples=16 \
        ++generation.dataloader.batch_size=16 \
        ++generation.args.nsteps=400

``capture.mode``:
  ``capture``  one full capture cell (default)
  ``verify``   the trunk determinism / step-nesting instrument check only
  ``timing``   a capture cell with shard + structure writing disabled, used to
               measure the denoise-vs-fold split before sizing the full harvest
"""

from __future__ import annotations

import json
import os
import sys


import _path  # noqa: F401

from analysis.commitment.capture import CaptureSpec, run_capture, verify_trunk_determinism  # noqa: E402


def _spec_from_cfg(cfg) -> CaptureSpec:
    c = cfg.get("capture", {})
    kwargs = {}
    for f in CaptureSpec.__dataclass_fields__:
        if f == "extras":
            continue
        if f in c:
            kwargs[f] = c[f]
    kwargs.setdefault("nsteps", int(cfg.generation.args.nsteps))
    return CaptureSpec(**kwargs)


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=None, config_name=None)
    def _entry(cfg):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass
        spec = _spec_from_cfg(cfg)
        mode = str(cfg.get("capture", {}).get("mode", "capture"))
        print(f"[run_capture] mode={mode} spec={spec}", flush=True)
        if mode == "verify":
            out = verify_trunk_determinism(cfg, spec, split_at=int(cfg.capture.get("split_at", 5)))
            print("[run_capture] verify:", json.dumps(out, indent=2), flush=True)
            os.makedirs(spec.out_dir, exist_ok=True)
            with open(os.path.join(spec.out_dir, f"verify_{spec.target_name}_step{spec.capture_step}.json"), "w") as f:
                json.dump(out, f, indent=2)
            bad = [k for k in ("repeat_max_abs_diff", "prefix_max_abs_diff", "split_endpoint_max_abs_diff") if out[k] != 0.0]
            if bad:
                raise SystemExit(f"trunk determinism/nesting check FAILED on {bad}: {out}")
            print("[run_capture] trunk determinism + step-nesting OK (exact)", flush=True)
            return
        if mode == "timing":
            spec.timing_only = True
            spec.save_structures = False
        man = run_capture(cfg, spec)
        os.makedirs(spec.out_dir, exist_ok=True)
        tag = f"{spec.target_name}_step{spec.capture_step}"
        with open(os.path.join(spec.out_dir, f"manifest_{mode}_{tag}.json"), "w") as f:
            json.dump(man, f, indent=2)
        print("[run_capture] timings:", json.dumps(man["timings"], indent=2), flush=True)

    _entry()


if __name__ == "__main__":
    main()
