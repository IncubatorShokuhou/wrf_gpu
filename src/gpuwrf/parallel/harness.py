"""Shared harness for decomposition checks and benchmarks (CPU virtual ranks and real GPUs).

load_case -> (carry, namelist) of a real single-domain case (product loader), optionally cropped to a window or
mirror-tiled into a large synthetic-from-real domain; run_single -> the single-device reference; run_decomposed ->
the TileRuntime result; compare -> per-leaf bitwise/max-diff report. Timings use block_until_ready only at the
end of a timed block (no per-step host sync)."""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.parallel import decomposition as dc


def load_case(input_dir: Path, scratch: Path, *, crop: str | None = None, mirror: tuple[int, int] | None = None,
              device=None) -> tuple[Any, Any, dict]:
    """Load the d01 domain of a wrfinput/wrfbdy case through the product loader (native single-domain path).
    ``crop`` 'x0:x1,y0:y1' keeps a window (window state + re-sliced boundary strips: a self-consistent smaller
    domain); ``mirror`` (mx, my) mirror-tiles the (cropped) domain (parallel/synthetic.py)."""
    import jax

    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains

    config = NestedPipelineConfig(Path(input_dir), scratch / "stream", scratch / "proof", hours=1, max_dom=1)
    t0 = time.perf_counter()
    # Full boundary stacks: the harness crops/steps the release loop directly.
    _h, bundles, _m, run_start, dts, carries = _load_domains(config, ("d01",), record_windows=False)
    meta = {"load_s": round(time.perf_counter() - t0, 1), "dt_s": dts["d01"], "run_start": str(run_start)}
    nl, carry = bundles["d01"].namelist, carries["d01"]
    if crop:
        (x0, x1), (y0, y1) = (tuple(int(v) for v in part.split(":")) for part in crop.split(","))
        full = dc.DecompositionSpec.build(nl.grid.nx, nl.grid.ny, 1, 1, 0)
        window = dc.Tile(0, 0, 0, (x0, x1), (y0, y1), (x0, x1), (y0, y1))
        nl = dc._slice_obj(nl, window, full, {})
        carry = dc.slice_tree(carry, window, full)
        meta["crop"] = {"x": [x0, x1], "y": [y0, y1]}
    if mirror:
        from gpuwrf.parallel.synthetic import mirror_tile_case

        carry, nl = mirror_tile_case(carry, nl, int(mirror[0]), int(mirror[1]))
        meta["mirror"] = list(mirror)
    if device is not None:
        carry = jax.device_put(carry, device)
        nl = jax.device_put(nl, device)
    meta.update(nx=int(nl.grid.nx), ny=int(nl.grid.ny), nz=int(nl.grid.nz), acoustic_substeps=int(nl.acoustic_substeps),
                radiation_cadence_steps=int(nl.radiation_cadence_steps))
    return carry, nl, meta


def fresh_copy(tree):
    """Distinct device copies of every array leaf: the release loop DONATES its carry (GPUWRF_CARRY_DONATE=1)
    and the product carry holds some arrays at two positions, so the caller's carry must never enter it."""
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda x: jnp.copy(x) if isinstance(x, jax.Array) else x, tree)


def run_single(carry, nl, *, start_step: int, steps: int, k: int, clock, cadence: int, warmup: bool = False) -> tuple[Any, dict]:
    """Single-device run in chunks of k steps (the compiled loop the tiles use; trip count traced). ``warmup``
    first compiles on a copy (one chunk) so the timed block excludes compile. ``carry`` is left intact."""
    import jax

    from gpuwrf.runtime import operational_mode as om

    if k < 1 or steps < 1 or steps % k:
        raise ValueError("steps must be positive and a multiple of k")
    device = next(iter(jax.tree_util.tree_leaves(carry)[0].devices()))
    start = jax.device_put(np.int32(start_step), device)
    trips = jax.device_put(np.int32(k), device)
    cad = jax.device_put(np.int32(cadence), device)

    # Donation on the nested release JIT does not propagate through this entry boundary.
    @partial(jax.jit, donate_argnums=(0, 2))
    def chunk(c, nl, si, clock, n, cadence):
        return om._advance_chunk_fori(c, nl, si, clock, n_steps=n, cadence=cadence), si + n

    timing: dict = {}
    if warmup:
        t0 = time.perf_counter()
        w = chunk(fresh_copy(carry), nl, fresh_copy(start), clock, trips, cad)
        jax.block_until_ready(w)
        del w
        timing["compile_and_first_chunk_s"] = round(time.perf_counter() - t0, 3)
    c = fresh_copy(carry)
    jax.block_until_ready(c)
    t0 = time.perf_counter()
    for i in range(0, steps, k):
        c, start = chunk(c, nl, start, clock, trips, cad)
    jax.block_until_ready(c)
    timing["run_s"] = round(time.perf_counter() - t0, 3)
    timing["s_per_step"] = timing["run_s"] / max(steps, 1)
    timing.update(start_step=start_step, steps=steps, stop_step_exclusive=start_step + steps, cadence=cadence)
    return c, timing


def finite_summary(tree) -> dict:
    """Post-timing host check, including runs without a single-device comparison."""
    import jax

    leaves = jax.tree_util.tree_leaves(tree)
    bad = []
    for i, x in enumerate(leaves):
        a = np.asarray(x)
        n = int((~np.isfinite(a)).sum()) if np.issubdtype(a.dtype, np.number) else 0
        if n:
            bad.append({"leaf": i, "non_finite": n})
    return {"leaves": len(leaves), "non_finite": sum(r["non_finite"] for r in bad),
            "verdict": "FINITE" if leaves and not bad else "INVALID", "bad_leaves": bad}


def compare(got, ref, kinds: list, carry0: list) -> dict:
    """Strict structure/schema/finite gate. BITWISE means identical literal bytes, including signed zero.

    Lists of leaves retain the original API; callers with pytrees also get structure validation.
    A mismatch never becomes a pass because values are NaN, missing or numerically equal but byte-distinct.
    """
    import jax

    got, got_def = jax.tree_util.tree_flatten(got)
    ref, ref_def = jax.tree_util.tree_flatten(ref)
    schema = []
    if got_def != ref_def:
        schema.append("candidate/reference pytree structures differ")
    if len(got) != len(ref) or len(ref) != len(kinds) or len(ref) != len(carry0) or not ref:
        schema.append(f"leaf counts candidate/reference/kinds/initial: {len(got)}/{len(ref)}/{len(kinds)}/{len(carry0)}")
    got_finite, ref_finite = finite_summary(got), finite_summary(ref)
    rows, n_bit = [], 0
    for i, (a, b, kind) in enumerate(zip(got, ref, kinds)):
        a, b = np.asarray(a), np.asarray(b)
        matched_schema = a.shape == b.shape and a.dtype == b.dtype
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        same = matched_schema and finite and a.tobytes() == b.tobytes()
        if not matched_schema:
            schema.append(f"leaf {i}: shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        row = {"leaf": i, "kind": kind[0], "shape": list(a.shape), "dtype": str(a.dtype), "bitwise": bool(same),
               "finite": finite, "changed_by_step": i >= len(carry0) or b.tobytes() != np.asarray(carry0[i]).tobytes()}
        if not same and matched_schema and finite and np.issubdtype(a.dtype, np.number):
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            differs = ~(d == 0)  # NaN differences count (a NaN where the reference is finite)
            finite_d = np.where(np.isfinite(d), d, 0.0)
            row["max_abs_diff"] = float(finite_d.max()) if finite_d.size else 0.0
            row["n_nonfinite_diff"] = int((differs & ~np.isfinite(d)).sum())
            scale = float(np.max(np.abs(b.astype(np.float64)))) if b.size else 0.0
            row["max_rel_to_field_max"] = row["max_abs_diff"] / (scale or 1.0)
            row["n_diff"] = int(differs.sum())
            if kind[0] == "h" and a.ndim >= 2 and differs.any():
                cols = np.argwhere(differs.reshape(-1, *d.shape[-2:]).any(axis=0))
                row["diff_x_range"] = [int(cols[:, 1].min()), int(cols[:, 1].max())]
                row["diff_y_range"] = [int(cols[:, 0].min()), int(cols[:, 0].max())]
        n_bit += same
        rows.append(row)
    invalid = bool(schema or got_finite["verdict"] != "FINITE" or ref_finite["verdict"] != "FINITE")
    return {"leaves": len(rows), "bitwise_leaves": n_bit, "changed_leaves": sum(r["changed_by_step"] for r in rows),
            "non_finite": got_finite["non_finite"], "reference_non_finite": ref_finite["non_finite"],
            "schema_errors": schema, "verdict": "INVALID" if invalid else ("BITWISE" if n_bit == len(ref) else "MISMATCH"),
            "mismatch_rows": [r for r in rows if not r["bitwise"]]}


__all__ = ["compare", "finite_summary", "fresh_copy", "load_case", "run_single"]
