"""Restart-seam analysis injection (A1 R5, lane w41-da-cycle): keep each member's model state across the analysis.

The file seam (``cycle_case``) restarts every member from an analysis wrfinput, so the port re-initialises
its held physics at every cycle (radiation tendency RTHRATEN until the next radiation call, KF trigger and
NCA countdown, h_diabatic, accumulators, Noah-MP internals read back through the land marker). Here the
member's end-of-window restart generation (``runtime.restart_store``) is kept and only the analysed
prognostics move by the analysis increment, ``analysis wrfinput - background frame`` at ``t_a``:

    u += dU, v += dV, qv += dQVAPOR,
    theta_m += (T_a+300)(1+rv/rd qv_a) - (T_b+300)(1+rv/rd qv_b)        (use_theta_m=1),
    p' and p_total += dP, ph' and ph_total += dPH;  MU must be unchanged (the analysis freezes it).

Every other carry leaf (held physics, land, accumulators, step scratch) is kept as the model left it; a
WRF restart likewise re-reads only state, and rk_step_prep recomputes ww/alt/php before the dynamics use
them. Cells with a zero increment keep their stored bits, so a zero-update analysis continues the forecast
bit for bit. The new generation keeps the stream identity; the forecast resumes with
``gpuwrf run --resume-checkpoint <generation> --extend-run``. Host-only (JAX on CPU); never a GPU.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

RVOVRD = 461.6 / 287.0
ANALYSED = ("U", "V", "T", "QVAPOR", "P", "PH")
_LEAVES = {"U": ("u",), "V": ("v",), "QVAPOR": ("qv",), "P": ("p_perturbation", "p_total"),
           "PH": ("ph_perturbation", "ph_total"), "THM": ("theta",)}
WRF_TIME = "%Y-%m-%d_%H:%M:%S"


def analysis_increments(analysis_wrfinput: Path, background_frame: Path, t_a: str) -> dict[str, np.ndarray]:
    """float64 increments in carry terms (keys of ``_LEAVES``); refuses a moved MU or non-finite fields."""
    from .state_io import read_frame

    an = read_frame(analysis_wrfinput, t_a, ANALYSED + ("MU",))
    bg = read_frame(background_frame, t_a, ANALYSED + ("MU",))
    for name in ANALYSED + ("MU",):
        a, b = np.asarray(an[name]), np.asarray(bg[name])
        if a.shape != b.shape:
            raise ValueError(f"{name}: analysis shape {a.shape} != background {b.shape}")
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError(f"{name}: non-finite analysis/background values")
    if not np.array_equal(np.asarray(an["MU"]), np.asarray(bg["MU"])):
        raise ValueError("analysis changed MU; the restart seam injects at fixed dry mass only")
    inc = {name: np.asarray(an[name], np.float64) - np.asarray(bg[name], np.float64)
           for name in ("U", "V", "QVAPOR", "P", "PH")}

    def theta_m(frame):
        return (np.asarray(frame["T"], np.float64) + 300.0) * (
            1.0 + RVOVRD * np.maximum(np.asarray(frame["QVAPOR"], np.float64), 0.0))

    same = (np.asarray(an["T"]) == np.asarray(bg["T"])) & (np.asarray(an["QVAPOR"]) == np.asarray(bg["QVAPOR"]))
    inc["THM"] = np.where(same, 0.0, theta_m(an) - theta_m(bg))
    return inc


def _apply(old, increment: np.ndarray) -> np.ndarray:
    old = np.asarray(old)
    if old.shape != increment.shape:
        raise ValueError(f"carry leaf shape {old.shape} != increment {increment.shape}")
    new = (old.astype(np.float64) + increment).astype(old.dtype)
    return np.where(increment == 0.0, old, new)  # zero increment: the stored bits, including -0.0


def inject(store_root: Path, analysis_wrfinput: Path, background_frame: Path, t_a: str, *,
           domain: str = "d01", max_bytes: int = 8 * 1024**3, max_generations: int = 2,
           reserve_bytes: int = 10 * 1024**3) -> dict[str, Any]:
    """Publish a new restart generation = latest generation + analysis increment (receipt returned)."""
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("restart injection is host-only: set JAX_PLATFORMS=cpu")
    from gpuwrf.runtime.restart_store import RestartStore

    store = RestartStore(Path(store_root), max_bytes, max_generations, reserve_bytes)
    source = store.latest()
    snapshot, manifest = store.read(source)
    identity = manifest["identity"]
    if tuple(manifest["domains"]) != (domain,):
        raise ValueError(f"restart injection supports one root domain, got {manifest['domains']}")
    start = datetime.fromisoformat(identity["run_start"]).replace(tzinfo=None)
    lead = datetime.strptime(t_a, WRF_TIME) - start
    steps, dt = int(manifest["own_steps"][domain]), float(manifest["dt_s"][domain])
    if abs(steps * dt - lead.total_seconds()) > 1e-6:
        raise ValueError(f"latest generation is at {start + timedelta(seconds=steps * dt)}, not the analysis time {t_a}")
    inc = analysis_increments(analysis_wrfinput, background_frame, t_a)
    carry = snapshot["carries"][domain]
    state = carry.state
    updates, stats = {}, {}
    for field, leaves in _LEAVES.items():
        for leaf in leaves:
            value = getattr(state, leaf)
            if value is None:
                raise ValueError(f"carry has no {leaf} leaf")
            new = _apply(np.asarray(value), inc[field])
            updates[leaf] = new
            stats[leaf] = {"dtype": str(new.dtype), "changed_cells": int((new != np.asarray(value)).sum()),
                           "increment_absmax": float(np.abs(inc[field]).max())}
    import jax.numpy as jnp

    new_state = state.replace(**{k: jnp.asarray(v) for k, v in updates.items()})
    for leaf, value in updates.items():  # replace must not have re-cast anything
        if not np.array_equal(np.asarray(getattr(new_state, leaf)), value):
            raise RuntimeError(f"{leaf}: injected value changed on State.replace")
    carries = dict(snapshot["carries"], **{domain: carry.replace(state=new_state)})
    generation, storage = store.save(carries, manifest["own_steps"], manifest["dt_s"], identity,
                                     driver_state=snapshot["driver_state"])
    new_manifest = generation / "manifest.json"
    return {"schema": 1, "kind": "restart-seam analysis injection", "t_a": t_a, "domain": domain,
            "source_generation": str(source), "source_payload_sha256": manifest["sha256"],
            "generation": str(generation),
            "generation_manifest_sha256": hashlib.sha256(new_manifest.read_bytes()).hexdigest(),
            "analysis_wrfinput": str(analysis_wrfinput), "background_frame": str(background_frame),
            "own_steps": manifest["own_steps"], "leaves": stats, "storage": storage}


def main(argv=None) -> int:
    """CPU child: ``python -m gpuwrf.assimilation.letkf.restart_inject SPEC.json`` (members -> paths)."""
    import sys

    spec = json.loads(Path((argv or sys.argv[1:])[0]).read_text())
    receipts = {}
    for label, entry in spec["members"].items():
        receipts[label] = inject(Path(entry["store"]), Path(entry["analysis_wrfinput"]),
                                 Path(entry["background_frame"]), spec["t_a"])
        print(label, receipts[label]["generation"], flush=True)
    Path(spec["receipt"]).write_text(json.dumps(receipts, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
