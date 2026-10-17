"""Localize c02 NF with the unchanged traced-count executable, one own step/call."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stop-step", type=int, default=2600)
    args = parser.parse_args()
    assert os.environ.get("GPUWRF_GPU_LOCK_HELD") == "1"
    assert int(os.environ["GPUWRF_GPU_LOCK_FD"]) >= 3
    import gpuwrf  # noqa: F401
    import jax
    import numpy as np
    from gpuwrf.runtime.operational_mode import _advance_chunk
    from gpuwrf.runtime.finite_state_guard import first_nonfinite_state_location
    assert jax.devices()[0].platform == "gpu", jax.devices()
    source = Path(_advance_chunk.__code__.co_filename).resolve().parents[3]
    assert subprocess.run(["git", "-C", str(source), "symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode != 0
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.snapshot.open("rb") as stream:
        saved = pickle.load(stream)
    carry = jax.device_put(saved["carry"])
    namelist = jax.device_put(saved["namelist"])
    clock = jax.device_put(saved["clock_base"])
    rows = []
    first_bad = None
    for step in range(int(saved["step"])+1, args.stop_step+1):
        # Diagnostic transfer at the own-step boundary, outside the unmodified
        # compiled timestep. Retain one predecessor; no extra in-step outputs.
        before = jax.device_get(carry)
        started = time.monotonic()
        carry = _advance_chunk(carry, namelist, step, clock, n_steps=1,
            cadence=int(namelist.radiation_cadence_steps))
        location = first_nonfinite_state_location(carry.state, domain="d01", step=step,
            sim_time_s=float(step)*float(namelist.dt_s))
        rows.append(dict(step=step, elapsed_s=time.monotonic()-started,
            first_bad=None if location is None else vars(location)))
        if location is not None:
            first_bad = vars(location)
            for label, value, at in (("last_finite", before, step-1), ("first_nonfinite", jax.device_get(carry), step)):
                with (args.output / f"{label}.pkl").open("xb") as stream:
                    pickle.dump(dict(carry=value, namelist=saved["namelist"],
                        clock_base=saved["clock_base"], step=at), stream, protocol=5)
            break
        if step % 10 == 0:
            print(f"step={step} finite", flush=True)
    receipt = dict(platform="gpu", source_commit=subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        snapshot=str(args.snapshot), cache=os.environ.get("GPUWRF_JAX_CACHE_DIR"),
        first_bad=first_bad, rows=rows, scope="own-step diagnostic; not timing or WRF fidelity evidence")
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(first_bad), flush=True)


if __name__ == "__main__":
    main()
