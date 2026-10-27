"""Prepare real Swiss domain columns through current production SF/MYNN wiring.

This is a workload fixture, not a WRF numerical oracle. It recomputes the port's
surface-layer handles from the original CPU-WRF history before building the
MYNN column view. No diagnostic values from a failed observer are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts/v025")]
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise RuntimeError("QUIET")
    from real_state import load_real_snapshot
    import jax
    import numpy as np
    from gpuwrf.contracts import state as S
    from gpuwrf.coupling import physics_couplers as C
    assert jax.devices()[0].platform == "cpu"
    S._gpu_device = lambda: jax.devices("cpu")[0]
    snapshot = load_real_snapshot(args.case, domain="d01", wrfout_name=args.history)
    grid = SimpleNamespace(metrics=snapshot.metrics,
                           projection=SimpleNamespace(dx_m=snapshot.config["dx"]))
    state = C.surface_adapter(snapshot.state, snapshot.config["dt"], grid)
    column = C._mynn_column_from_state(state, grid)
    ny, nx, nz = column.theta.shape
    column = C._flatten_columns_to_batch(column, ny, nx)
    flux = C._flatten_columns_to_batch(C._surface_fluxes_from_state(state, grid), ny, nx)
    arrays = {f"state_{n}": np.asarray(getattr(column, n), np.float32)
              for n in column.__slots__ if getattr(column, n) is not None}
    arrays.update({f"flux_{n}": np.asarray(getattr(flux, n), np.float32)
                   for n in flux._fields if getattr(flux, n) is not None})
    assert all(np.isfinite(a).all() for a in arrays.values()), "non-finite workload input"
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "columns.npz"
    np.savez(path, **arrays)
    record = dict(scope=__doc__, provenance=snapshot.provenance,
                  shape=[ny, nx, nz], dt=snapshot.config["dt"], dx=snapshot.config["dx"],
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_coupler_sha256=hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),
                  state_fields=list(column.__slots__),
                  absent_state_fields=[n for n in column.__slots__ if getattr(column, n) is None],
                  absent_flux_fields=[n for n in flux._fields if getattr(flux, n) is None],
                  dtype="float32", nonfinite_inputs=0)
    (args.out / "inputs.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("shape", "dt", "dx", "input_sha256")}), flush=True)


if __name__ == "__main__":
    main()
