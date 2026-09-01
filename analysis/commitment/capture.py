"""No-search commitment capture for the Complexa denoiser (GPU stage).

What this does, and why it is not the Phase-1 beam-search harvest
-----------------------------------------------------------------
The prior harvest captured states *inside a live beam search*, so a state alive
at step 300 had already survived two rounds of reward filtering while a state at
step 100 had survived none. Their spreads are not comparable across steps.

Here there is **no beam and no selection**. For one capture step ``t`` we draw
``R`` independent initialisations ("roots"), tile each one ``C`` times
("children"), and run every child on its own stochastic trajectory from noise to
``t``. That produces ``R*C`` states at step ``t`` with a clean three-level
nesting::

    root r          (independent initialisation: binder length + initial noise)
      └── child c   (independent trajectory 0 -> t from that same initialisation)
            └── completion j  (independent trajectory t -> nsteps, then AF2 fold)

The three-level variance decomposition of any completion-level outcome then
splits into: what the *initialisation* fixed (between roots), what the
*trajectory up to t* added (between children of a root), and what is still open
after ``t`` (within a child). The first two together are what the state at ``t``
determines; the middle term alone is what a search branching at initialisation
and selecting at ``t`` could act on.

Reproducibility / nesting across capture steps
----------------------------------------------
Roots and children are seeded independently of the capture step, and the trunk
simulation is chunked with a *per-chunk* seed rather than relying on one global
RNG stream. Both make the trunk deterministic, and together they make the states
at an earlier capture step exact prefixes of the trajectories captured at a later
one: the seven capture steps observe the *same* 128 trajectories at seven times,
not seven independent draws. (Verified by
:func:`verify_trunk_determinism`, which checks that a split simulation
``0->a->t`` reproduces a single ``0->t`` call bit-for-bit.)

All ``proteinfoundation`` imports are deferred into functions, so importing this
module does not require Complexa to be installed.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = [
    "CaptureSpec",
    "ActivationCapturer",
    "run_capture",
    "verify_trunk_determinism",
]

# Chunk-seed stride: any odd stride works; a prime keeps chunk streams apart.
_CHUNK_SEED_STRIDE = 7919
CA_ATOM_INDEX = 1  # atom37 ordering is N, CA, C, O, ...
NM_TO_ANG = 10.0


@dataclass
class CaptureSpec:
    """One capture cell: a (target, capture step) pair."""

    target_name: str
    capture_step: int
    nsteps: int = 400
    n_roots: int = 16
    n_children: int = 8
    k_completions: int = 8
    #: interface residue budget for the second-order (covariance) readout
    n_iface_binder: int = 24
    n_iface_target: int = 8
    #: seeds are deliberately independent of ``capture_step`` so the trunk
    #: trajectories are shared across every capture step of a target
    root_seed: int = 42
    trunk_seed: int = 424_242
    completion_seed: int = 84_242
    sim_chunk: int = 16
    capture_chunk: int = 8
    fold_chunk: int = 128
    out_dir: str = "./capture_out"
    save_structures: bool = True
    timing_only: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return self.n_roots * self.n_children

    @property
    def n_iface(self) -> int:
        return self.n_iface_binder + self.n_iface_target


class ActivationCapturer:
    """Forward hooks on the denoiser's transformer layers.

    Records each layer's output residual stream ``[B, n_ext, dim]`` from the most
    recent *armed* forward pass. Recording only happens while :attr:`armed` is
    true, so the hooks cost nothing during the many unarmed sampler forwards.
    """

    def __init__(self, nn_module: Any):
        self.layers = nn_module.transformer_layers
        self.nlayers = len(self.layers)
        self._buf: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self.armed = False
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            if not self.armed:
                return
            out = output[0] if isinstance(output, (tuple, list)) else output
            self._buf[idx] = out.detach()

        return hook

    def clear(self) -> None:
        self._buf = {}

    def stacked(self) -> torch.Tensor:
        """``[B, nlayers, n_ext, dim]`` from the last armed forward."""
        if len(self._buf) != self.nlayers:
            raise RuntimeError(f"captured {len(self._buf)}/{self.nlayers} layers")
        return torch.stack([self._buf[i] for i in range(self.nlayers)], dim=1)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


# --------------------------------------------------------------------------- #
# batch / device plumbing
# --------------------------------------------------------------------------- #


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _slice_batch(batch: dict[str, Any], s0: int, s1: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, (torch.Tensor, list)):
            out[k] = v[s0:s1]
        elif k == "nsamples":
            out[k] = s1 - s0
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# trunk simulation (per-chunk seeded, so it is reproducible and step-nested)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def seeded_partial_simulation(
    proteina: Any,
    search_ctx: Any,
    batch: dict[str, Any],
    xt: dict[str, torch.Tensor],
    x_1_pred: dict[str, torch.Tensor] | None,
    mask: torch.Tensor,
    start_step: int,
    end_step: int,
    chunk: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """``partial_simulation`` over chunks, each with its own deterministic seed.

    Chunk-local seeding is what makes an earlier capture step a strict prefix of
    a later one: chunk ``c`` always starts from ``seed + stride*c`` and consumes
    its own RNG stream, so how many steps a *different* chunk ran cannot shift
    it.
    """
    from proteinfoundation.utils.tensor_utils import concat_dict_tensors

    total = next(iter(xt.values())).shape[0]
    xt_parts: list[dict] = []
    pred_parts: list[dict] = []
    for ci, s0 in enumerate(range(0, total, chunk)):
        s1 = min(s0 + chunk, total)
        torch.manual_seed(seed + _CHUNK_SEED_STRIDE * ci)
        c_batch = _slice_batch(batch, s0, s1)
        c_xt = {k: v[s0:s1] for k, v in xt.items()}
        c_pred = {k: v[s0:s1] for k, v in x_1_pred.items()} if x_1_pred is not None else None
        r_xt, r_pred = proteina.fm.partial_simulation(
            batch=c_batch,
            x=c_xt,
            x_1_pred=c_pred,
            mask=mask[s0:s1],
            predict_for_sampling=search_ctx.predict_fn,
            start_step=start_step,
            end_step=end_step,
            ts=search_ctx.ts,
            gt=search_ctx.gt,
            self_cond=search_ctx.self_cond,
            simulation_step_params=search_ctx.simulation_step_params,
            device=search_ctx.device,
            guidance_w=search_ctx.guidance_w,
            ag_ratio=search_ctx.ag_ratio,
        )
        xt_parts.append(r_xt)
        if r_pred is not None:
            pred_parts.append(r_pred)
    return (
        concat_dict_tensors(xt_parts, dim=0),
        concat_dict_tensors(pred_parts, dim=0) if pred_parts else None,
    )


# --------------------------------------------------------------------------- #
# activation capture + pooling
# --------------------------------------------------------------------------- #


def _interface_residue_index(
    binder_ca_ang: torch.Tensor,
    binder_valid: torch.Tensor,
    target_ca_ang: torch.Tensor,
    hotspot_mask: torch.Tensor,
    n_iface_binder: int,
    n_iface_target: int,
    binder_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the interface residue set for one candidate, using only step-``t`` info.

    Returns ``(token_idx [K], kind [K], valid [K])`` where ``kind`` is 0 for a
    binder residue and 1 for a target hotspot residue, and ``token_idx`` indexes
    the extended token axis ``binder(binder_block) || target(n_target)``.
    ``binder_block`` is the width the *model* used for this forward pass - the
    longest binder in the chunk - not the padded array width.

    Binder residues are ranked by distance from their Cα to the nearest hotspot
    Cα *in the model's own clean prediction at the capture step*. At an early
    capture step that structure is close to noise, which is the honest situation:
    the selection is allowed to see nothing the probe could not see.
    """
    n_orig = binder_ca_ang.shape[0]
    n_target = target_ca_ang.shape[0]
    device = binder_ca_ang.device

    hot = hotspot_mask.bool()
    if hot.sum() == 0:
        hot = torch.ones(n_target, dtype=torch.bool, device=device)
    hot_idx = torch.nonzero(hot, as_tuple=False).flatten()
    hot_ca = target_ca_ang[hot_idx]  # [H, 3]

    d = torch.cdist(binder_ca_ang, hot_ca)  # [n_orig, H]
    dmin = d.min(dim=1).values
    dmin = torch.where(binder_valid.bool(), dmin, torch.full_like(dmin, float("inf")))
    nb = min(n_iface_binder, int(binder_valid.sum().item()))
    order = torch.argsort(dmin)[:nb]

    tok = torch.zeros(n_iface_binder + n_iface_target, dtype=torch.long, device=device)
    kind = torch.zeros_like(tok)
    valid = torch.zeros_like(tok, dtype=torch.bool)

    tok[:nb] = order
    kind[:nb] = 0
    valid[:nb] = True

    nt = min(n_iface_target, int(hot_idx.numel()))
    # nearest hotspots to the binder centroid, so the target slots are stable
    if nt > 0:
        centroid = binder_ca_ang[binder_valid.bool()].mean(dim=0, keepdim=True)
        hot_order = torch.argsort(torch.cdist(centroid, hot_ca).flatten())[:nt]
        sel = hot_idx[hot_order]
        tok[n_iface_binder : n_iface_binder + nt] = binder_block + sel
        kind[n_iface_binder : n_iface_binder + nt] = 1
        valid[n_iface_binder : n_iface_binder + nt] = True
    return tok, kind, valid


@torch.no_grad()
def capture_states(
    proteina: Any,
    search_ctx: Any,
    capturer: ActivationCapturer,
    batch_tiled: dict[str, Any],
    xt: dict[str, torch.Tensor],
    x_1_pred: dict[str, torch.Tensor] | None,
    mask_tiled: torch.Tensor,
    end_step: int,
    n_orig: int,
    n_target: int,
    hotspot_mask: torch.Tensor,
    binder_ca_ang: torch.Tensor,
    target_ca_ang: torch.Tensor,
    spec: CaptureSpec,
) -> dict[str, torch.Tensor]:
    """One dedicated hooked forward per chunk at the capture step; pool per candidate.

    Returns CPU tensors:
      ``binder_pool``/``iface_pool`` ``[N, L, D]`` float32 (mean over the valid
      binder residues / over the target hotspot residues — the Phase-1
      conventions), ``iface_res`` ``[N, L, K, D]`` float16 (the per-residue
      interface vectors the covariance readout consumes), plus the residue
      bookkeeping.
    """
    fm = proteina.fm
    ts = search_ctx.ts
    device = search_ctx.device
    n_total = next(iter(xt.values())).shape[0]

    binder_pools, iface_pools, res_acts = [], [], []
    tok_all, kind_all, valid_all = [], [], []
    blocks: list[int] = []

    for s0 in range(0, n_total, spec.capture_chunk):
        s1 = min(s0 + spec.capture_chunk, n_total)
        c = s1 - s0
        b = _slice_batch(batch_tiled, s0, s1)
        b["x_t"] = {k: v[s0:s1] for k, v in xt.items()}
        b["t"] = {dm: ts[dm][end_step] * torch.ones(c, device=device) for dm in fm.data_modes}
        b["mask"] = mask_tiled[s0:s1]
        if x_1_pred is not None:
            b["x_sc"] = {k: v[s0:s1] for k, v in x_1_pred.items()}
        capturer.clear()
        capturer.armed = True
        proteina.predict_for_sampling(b, mode="full", n_recycle=0)
        capturer.armed = False
        acts = capturer.stacked().float()  # [c, L, n_ext, D]
        # The model does NOT keep the padded binder width: it crops the binder
        # block to the longest binder in THIS chunk, so the extended token axis is
        # (max valid binder length in chunk) || (target). Assuming n_orig here
        # silently misaligns every target token whenever the chunk's longest
        # binder is shorter than the padded array.
        n_ext = int(acts.shape[2])
        binder_block = int(mask_tiled[s0:s1, :n_orig].sum(dim=1).max().item())
        if n_ext != binder_block + n_target:
            raise AssertionError(
                f"token axis {n_ext} != binder_block({binder_block}) + n_target({n_target}); "
                f"n_orig={n_orig}, chunk={s0}:{s1}"
            )
        for j in range(c):
            gi = s0 + j
            bvalid = mask_tiled[gi, :binder_block].bool()
            if bvalid.sum() == 0:
                bvalid = torch.ones(binder_block, dtype=torch.bool, device=device)
            binder_pools.append(acts[j, :, :binder_block, :][:, bvalid, :].mean(dim=1).cpu())

            hrow = hotspot_mask[gi % hotspot_mask.shape[0]].bool()
            if hrow.sum() == 0:
                hrow = torch.ones(n_target, dtype=torch.bool, device=device)
            tgt = acts[j, :, binder_block : binder_block + n_target, :]
            iface_pools.append(tgt[:, hrow, :].mean(dim=1).cpu())

            tok, kind, valid = _interface_residue_index(
                binder_ca_ang[gi],
                mask_tiled[gi, :n_orig],
                target_ca_ang[gi % target_ca_ang.shape[0]],
                hotspot_mask[gi % hotspot_mask.shape[0]],
                spec.n_iface_binder,
                spec.n_iface_target,
                binder_block,
            )
            sel = acts[j][:, tok, :]  # [L, K, D]
            sel = sel * valid[None, :, None].to(sel.dtype)
            res_acts.append(sel.half().cpu())
            tok_all.append(tok.cpu())
            kind_all.append(kind.cpu())
            valid_all.append(valid.cpu())
        blocks.append(binder_block)
        del acts
        capturer.clear()

    return {
        "binder_block_per_chunk": torch.tensor(blocks, dtype=torch.int32),
        "binder_pool": torch.stack(binder_pools),
        "iface_pool": torch.stack(iface_pools),
        "iface_res": torch.stack(res_acts),
        "iface_tok": torch.stack(tok_all),
        "iface_kind": torch.stack(kind_all).to(torch.int8),
        "iface_valid": torch.stack(valid_all),
    }


# --------------------------------------------------------------------------- #
# decoding / structure IO
# --------------------------------------------------------------------------- #


def _decode(proteina: Any, x: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    from proteinfoundation.utils.sample_utils import sample_formatting

    return sample_formatting(
        x=x,
        extra_info={"mask": mask},
        ret_mode="coors37_n_aatype",
        data_modes=list(proteina.cfg_exp.product_flowmatcher),
        autoencoder=getattr(proteina, "autoencoder", None),
    )


def _write_pdbs(sample_prots: dict[str, torch.Tensor], names: list[str], out_dir: str) -> list[str]:
    from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    n = sample_prots["coors"].shape[0]
    chain_index = sample_prots.get("chain_index")
    for i in range(n):
        p = os.path.join(out_dir, f"{names[i]}.pdb")
        write_prot_to_pdb(
            prot_pos=sample_prots["coors"][i].float().detach().cpu().numpy(),
            aatype=sample_prots["residue_type"][i].detach().cpu().numpy(),
            file_path=p,
            chain_index=chain_index[i].detach().cpu().numpy() if chain_index is not None else None,
            overwrite=True,
            no_indexing=True,
        )
        paths.append(p)
    return paths


def _tar_dir(src_dir: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tf:
        for name in sorted(os.listdir(src_dir)):
            tf.add(os.path.join(src_dir, name), arcname=name)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #


def _make_search_context(proteina: Any, inf_cfg: Any, batch: dict[str, Any]) -> Any:
    """Build Complexa's own ``SearchContext`` without running any search."""
    from proteinfoundation.search.base_search import BaseSearch

    class _NoSearch(BaseSearch):
        def search(self, batch):  # pragma: no cover - never called
            raise NotImplementedError("capture mode runs no search")

    return _NoSearch(proteina, inf_cfg).build_search_context(batch)


def _log(msg: str) -> None:
    print(f"[capture] {msg}", flush=True)


def _progress(step: int, total: int, phase: str) -> None:
    """Optional progress hook kept dependency-free for the public release."""
    return None


@torch.no_grad()
def run_capture(cfg: Any, spec: CaptureSpec) -> dict[str, Any]:
    """Run one capture cell end to end and write its shard.

    Cost per cell is dominated by ``n_states * k_completions`` AF2 folds; the
    denoising cost is ``n_states`` trunk steps ``0->t`` plus
    ``n_states * k_completions`` completion steps ``t->nsteps``.
    """
    import hydra
    import lightning as L
    from proteinfoundation.generate import load_ckpt_n_configure_inference, validate_checkpoint_paths
    from proteinfoundation.rewards.base_reward import TOTAL_REWARD_KEY
    from proteinfoundation.rewards.reward_utils import compute_reward_from_samples, initialize_reward_model
    from proteinfoundation.search.search_utils import generate_samples_to_completion, tile_batch, tile_tensor_dict
    from proteinfoundation.utils.sample_utils import prepend_target_to_samples

    t_job0 = time.time()
    timings: dict[str, float] = {}
    validate_checkpoint_paths(cfg)
    torch.set_float32_matmul_precision("high")

    # --- model ------------------------------------------------------------
    t0 = time.time()
    L.seed_everything(spec.root_seed)
    model = load_ckpt_n_configure_inference(cfg)
    device = torch.device("cuda")
    model = model.to(device).eval()
    model.reward_model = initialize_reward_model(model.inf_cfg)
    timings["model_load_s"] = time.time() - t0
    _log(f"model loaded in {timings['model_load_s']:.1f}s")

    # --- roots: seeded independently of the capture step -------------------
    L.seed_everything(spec.root_seed)
    dataloader = hydra.utils.instantiate(cfg.generation.dataloader)
    batch = next(iter(dataloader))
    batch = _to_device(batch, device)
    init_mask = batch["mask"]
    n_roots, n_orig = init_mask.shape
    if n_roots != spec.n_roots:
        raise ValueError(
            f"dataloader produced {n_roots} roots, spec wants {spec.n_roots}; "
            "set generation.dataloader.dataset.nres.nsamples and dataloader.batch_size"
        )
    if "target_hotspot_mask" not in batch:
        raise ValueError(f"target {spec.target_name} has no target_hotspot_mask (no hotspots defined)")
    hotspot_mask = batch["target_hotspot_mask"]
    n_target = int(hotspot_mask.shape[1])
    binder_len = init_mask.sum(dim=1).detach().cpu().numpy().astype(np.int64)
    _log(f"roots={n_roots} n_orig={n_orig} n_target={n_target} binder_len={binder_len.tolist()}")

    search_ctx = _make_search_context(model, cfg.generation, batch)
    if search_ctx.nsteps != spec.nsteps:
        raise ValueError(f"cfg nsteps {search_ctx.nsteps} != spec.nsteps {spec.nsteps}")

    capturer = ActivationCapturer(model.nn)
    _log(f"hooked {capturer.nlayers} denoiser transformer layers")

    N = spec.n_states
    batch_tiled = tile_batch(batch, spec.n_children)
    mask_tiled = init_mask.repeat(spec.n_children, 1)
    # layout after tile_tensor_dict: flat = child * n_roots + root
    root_idx = np.tile(np.arange(n_roots), spec.n_children)
    child_idx = np.repeat(np.arange(spec.n_children), n_roots)
    tags = [f"root{r}_child{c}" for c, r in zip(child_idx, root_idx)]

    # --- trunk: noise -> capture step, no selection ------------------------
    t0 = time.time()
    torch.manual_seed(spec.root_seed)
    xt0 = model.fm.sample_noise(n_orig, shape=(n_roots,), device=device, mask=init_mask)
    xt = tile_tensor_dict(xt0, spec.n_children)
    xt, x_1_pred = seeded_partial_simulation(
        model,
        search_ctx,
        batch_tiled,
        xt,
        None,
        mask_tiled,
        start_step=0,
        end_step=spec.capture_step,
        chunk=spec.sim_chunk,
        seed=spec.trunk_seed,
    )
    timings["trunk_s"] = time.time() - t0
    timings["trunk_s_per_state_step"] = timings["trunk_s"] / max(1, N * spec.capture_step)
    _log(f"trunk 0->{spec.capture_step} for {N} states in {timings['trunk_s']:.1f}s")
    _progress(1, 3 + spec.k_completions, "trunk")

    # --- decode the model's CLEAN prediction at the capture step -----------
    # (not the noisy latent xt: x_1_pred is what the model currently believes
    # the finished design is, and it is the baseline the activation probe has
    # to beat.)
    partial = _decode(model, x_1_pred, mask_tiled)
    binder_ca_ang = partial["coors"][:, :n_orig, CA_ATOM_INDEX, :].detach()
    target_ca_ang = (batch["x_target"][:, :, CA_ATOM_INDEX, :] * NM_TO_ANG).detach()

    # --- activations at the capture step -----------------------------------
    t0 = time.time()
    acts = capture_states(
        model,
        search_ctx,
        capturer,
        batch_tiled,
        xt,
        x_1_pred,
        mask_tiled,
        spec.capture_step,
        n_orig,
        n_target,
        hotspot_mask,
        binder_ca_ang,
        target_ca_ang,
        spec,
    )
    timings["capture_s"] = time.time() - t0
    _log(f"activation capture in {timings['capture_s']:.1f}s -> binder_pool {tuple(acts['binder_pool'].shape)}")
    _progress(2, 3 + spec.k_completions, "capture")

    scratch = tempfile.mkdtemp(prefix="commit_capture_")
    partial_dir = os.path.join(scratch, "partial")
    comp_dir = os.path.join(scratch, "completions")

    if spec.save_structures:
        pp = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in partial.items()}
        if batch.get("prepend_target", False) and not hasattr(model, "ligand"):
            pp = prepend_target_to_samples(pp, batch, repeat_mode="tile")
        _write_pdbs(pp, tags, partial_dir)

    # --- k independent completions per state, each AF2-folded --------------
    metric_cols: dict[str, list[np.ndarray]] = {}
    comp_timings = []
    for j in range(spec.k_completions):
        t0 = time.time()
        torch.manual_seed(spec.completion_seed + 1_000_003 * j + spec.capture_step)
        done = generate_samples_to_completion(
            search_ctx, model, xt, spec.capture_step, init_mask, x_1_pred=x_1_pred
        )
        t_gen = time.time() - t0
        t1 = time.time()
        rewards = compute_reward_from_samples(
            model.reward_model, done, hotspot_mask, getattr(model, "ligand", None)
        )
        t_fold = time.time() - t1
        for key, val in rewards.items():
            metric_cols.setdefault(key, []).append(val.detach().cpu().numpy().astype(np.float32))
        if spec.save_structures:
            _write_pdbs(done, [f"{t}_comp{j}" for t in tags], comp_dir)
        comp_timings.append({"j": j, "gen_s": t_gen, "fold_s": t_fold, "n": int(N)})
        _log(
            f"completion {j + 1}/{spec.k_completions}: gen {t_gen:.1f}s fold {t_fold:.1f}s "
            f"({t_fold / max(1, N):.2f}s/fold) mean_total_reward="
            f"{float(np.nanmean(metric_cols[TOTAL_REWARD_KEY][-1])):.4f}"
        )
        _progress(3 + j, 3 + spec.k_completions, "completions")
        del done

    capturer.remove()
    metrics = {k: np.stack(v, axis=1) for k, v in metric_cols.items()}  # [N, k]
    timings["gen_s_total"] = float(sum(c["gen_s"] for c in comp_timings))
    timings["fold_s_total"] = float(sum(c["fold_s"] for c in comp_timings))
    timings["fold_s_per_fold"] = timings["fold_s_total"] / max(1, N * spec.k_completions)
    timings["gen_s_per_state_step"] = timings["gen_s_total"] / max(
        1, N * spec.k_completions * (spec.nsteps - spec.capture_step)
    )

    # --- write the shard ---------------------------------------------------
    out_dir = os.path.join(spec.out_dir, spec.target_name, f"step{spec.capture_step}")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "spec": asdict(spec),
        "target_name": spec.target_name,
        "capture_step": spec.capture_step,
        "root_idx": root_idx,
        "child_idx": child_idx,
        "tags": tags,
        "binder_len": binder_len,
        "n_orig": int(n_orig),
        "n_target": int(n_target),
        "hotspot_mask": hotspot_mask.detach().cpu().numpy().astype(bool),
        "partial_ca": binder_ca_ang.half().cpu(),
        "partial_valid": init_mask.repeat(spec.n_children, 1).detach().cpu(),
        "target_ca": target_ca_ang.half().cpu(),
        "metrics": metrics,
        "timings": timings,
        "comp_timings": comp_timings,
        **{k: v for k, v in acts.items()},
    }
    shard_path = os.path.join(out_dir, "shard.pt")
    if not spec.timing_only:
        torch.save(payload, shard_path)
        _log(f"wrote {shard_path} ({os.path.getsize(shard_path) / 1e6:.1f} MB)")
        if spec.save_structures:
            _tar_dir(partial_dir, os.path.join(out_dir, "partial.tar.gz"))
            _tar_dir(comp_dir, os.path.join(out_dir, "completions.tar.gz"))
            _log("wrote partial.tar.gz + completions.tar.gz")

    timings["total_s"] = time.time() - t_job0
    manifest = {
        "target_name": spec.target_name,
        "capture_step": spec.capture_step,
        "n_states": int(N),
        "n_folds": int(N * spec.k_completions),
        "binder_len": binder_len.tolist(),
        "n_orig": int(n_orig),
        "n_target": int(n_target),
        "timings": timings,
        "comp_timings": comp_timings,
        "spec": asdict(spec),
    }
    if not spec.timing_only:
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    _log(
        f"DONE {spec.target_name} step{spec.capture_step}: {N} states, "
        f"{N * spec.k_completions} folds, total {timings['total_s'] / 60:.1f} min"
    )
    return manifest


@torch.no_grad()
def verify_trunk_determinism(cfg: Any, spec: CaptureSpec, split_at: int = 5) -> dict[str, Any]:
    """Instrument check for the step-nesting claim.

    Two things have to hold for the seven capture steps to be seven views of the
    *same* 128 trajectories rather than seven independent draws:

    ``repeat_max_abs_diff``
        Running the same trunk twice gives bit-identical states (no
        nondeterministic kernel is injecting noise we do not control).
    ``prefix_max_abs_diff``
        The state at ``split_at`` produced by a short ``0 -> split_at`` job equals
        the state a long ``0 -> capture_step`` job passes through at ``split_at``.
        This is checked by emulating the long run per chunk as ``0 -> split_at``
        followed by ``split_at -> capture_step`` on the *same, uninterrupted* RNG
        stream, and comparing (a) its intermediate against the short job and (b)
        its endpoint against the real single-call long job.

    Both should be exactly ``0.0``.
    """
    import hydra
    import lightning as L
    from proteinfoundation.generate import load_ckpt_n_configure_inference
    from proteinfoundation.search.search_utils import tile_batch, tile_tensor_dict
    from proteinfoundation.utils.tensor_utils import concat_dict_tensors

    L.seed_everything(spec.root_seed)
    model = load_ckpt_n_configure_inference(cfg)
    device = torch.device("cuda")
    model = model.to(device).eval()
    torch.set_float32_matmul_precision("high")
    L.seed_everything(spec.root_seed)
    dataloader = hydra.utils.instantiate(cfg.generation.dataloader)
    batch = _to_device(next(iter(dataloader)), device)
    init_mask = batch["mask"]
    n_roots, n_orig = init_mask.shape
    search_ctx = _make_search_context(model, cfg.generation, batch)

    batch_tiled = tile_batch(batch, spec.n_children)
    mask_tiled = init_mask.repeat(spec.n_children, 1)

    def _fresh_xt0():
        torch.manual_seed(spec.root_seed)
        return tile_tensor_dict(
            model.fm.sample_noise(n_orig, shape=(n_roots,), device=device, mask=init_mask),
            spec.n_children,
        )

    def _sim(xt, pred, s, e, seed):
        return seeded_partial_simulation(
            model, search_ctx, batch_tiled, xt, pred, mask_tiled, s, e, spec.sim_chunk, seed
        )

    long_xt, _ = _sim(_fresh_xt0(), None, 0, spec.capture_step, spec.trunk_seed)
    long_xt_again, _ = _sim(_fresh_xt0(), None, 0, spec.capture_step, spec.trunk_seed)
    short_xt, _ = _sim(_fresh_xt0(), None, 0, split_at, spec.trunk_seed)

    # Emulate the long run chunk by chunk with an uninterrupted RNG stream,
    # snapshotting at ``split_at``.
    xt0 = _fresh_xt0()
    total = next(iter(xt0.values())).shape[0]
    mid_parts, end_parts = [], []
    for ci, s0 in enumerate(range(0, total, spec.sim_chunk)):
        s1 = min(s0 + spec.sim_chunk, total)
        torch.manual_seed(spec.trunk_seed + _CHUNK_SEED_STRIDE * ci)
        kw = dict(
            predict_for_sampling=search_ctx.predict_fn,
            ts=search_ctx.ts,
            gt=search_ctx.gt,
            self_cond=search_ctx.self_cond,
            simulation_step_params=search_ctx.simulation_step_params,
            device=search_ctx.device,
            guidance_w=search_ctx.guidance_w,
            ag_ratio=search_ctx.ag_ratio,
        )
        c_batch = _slice_batch(batch_tiled, s0, s1)
        m_xt, m_pred = model.fm.partial_simulation(
            batch=c_batch,
            x={k: v[s0:s1] for k, v in xt0.items()},
            x_1_pred=None,
            mask=mask_tiled[s0:s1],
            start_step=0,
            end_step=split_at,
            **kw,
        )
        mid_parts.append({k: v.clone() for k, v in m_xt.items()})
        e_xt, _ = model.fm.partial_simulation(
            batch=c_batch,
            x=m_xt,
            x_1_pred=m_pred,
            mask=mask_tiled[s0:s1],
            start_step=split_at,
            end_step=spec.capture_step,
            **kw,
        )
        end_parts.append(e_xt)
    mid_xt = concat_dict_tensors(mid_parts, dim=0)
    end_xt = concat_dict_tensors(end_parts, dim=0)

    def _maxdiff(a, b):
        return max(float((a[k] - b[k]).abs().max().item()) for k in a)

    return {
        "repeat_max_abs_diff": _maxdiff(long_xt, long_xt_again),
        "prefix_max_abs_diff": _maxdiff(short_xt, mid_xt),
        "split_endpoint_max_abs_diff": _maxdiff(long_xt, end_xt),
        "split_at": split_at,
        "capture_step": spec.capture_step,
        "sim_chunk": spec.sim_chunk,
        "n_states": int(total),
    }
