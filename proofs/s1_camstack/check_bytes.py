"""CPU regression byte gate of final CAM against the original source helpers.

This is regression evidence only. The separate pristine-WRF oracle tests remain
the fidelity authority, with their existing tolerances and active-regime census.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-ref", required=True)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    assert os.environ.get("JAX_PLATFORMS") == "cpu"
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tests"))
    import gpuwrf  # noqa: F401
    import jax
    import numpy as np
    import test_ra_cam_lw_oracle as test
    from gpuwrf.physics import ra_cam_lw_abs as absmod, ra_cam_lw as lwmod

    fx = dict(np.load(test.FIXTURE))
    tables = test.C.load_cam_abs_tables()
    candidate_fresh = test._camrad_lw(fx, tables, "l", True)
    candidate_held = test._camrad_lw(fx, tables, "h", False, candidate_fresh.held)
    candidates = (candidate_fresh, candidate_held)
    restored = []
    replacements = [(absmod, "src/gpuwrf/physics/ra_cam_lw_abs.py",
                     ("_interp32", "_h2o_overlap_tw", "_h2o_weights", "_trcabn")),
                    (lwmod, "src/gpuwrf/physics/ra_cam_lw.py", ("radclwmx_fluxes", "_cumsum_levels", "radems", "interp5", "trcems", "lookup_index"))]
    for mod, file, names in replacements:
        source = subprocess.check_output(["git", "-C", str(root), "show", f"{a.base_ref}:{file}"], text=True)
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                restored.append((mod, node.name, getattr(mod, node.name)))
                exec(compile(ast.Module(body=[node], type_ignores=[]), mod.__file__, "exec"), mod.__dict__)
    try:
        jax.clear_caches()
        baseline_fresh = test._camrad_lw(fx, tables, "l", True)
        baseline_held = test._camrad_lw(fx, tables, "h", False, baseline_fresh.held)
    finally:
        for mod, name, fn in restored:
            setattr(mod, name, fn)
        jax.clear_caches()
    rows = []
    dumps = {}
    for arm, candidate, baseline in zip(("fresh", "held"), candidates, (baseline_fresh, baseline_held)):
        for (path, got), ref in zip(jax.tree_util.tree_leaves_with_path(candidate), jax.tree.leaves(baseline)):
            got, ref = np.asarray(got), np.asarray(ref)
            name = arm + jax.tree_util.keystr(path)
            finite = bool(np.isfinite(got).all() and np.isfinite(ref).all())
            same = finite and got.dtype == ref.dtype and got.shape == ref.shape and got.tobytes() == ref.tobytes()
            rows.append({"name": name, "byte_equal": same, "finite": finite,
                         "max_abs": float(np.max(np.abs(got.astype(np.float64) - ref))),
                         "candidate_sha256": hashlib.sha256(got.tobytes()).hexdigest(),
                         "base_sha256": hashlib.sha256(ref.tobytes()).hexdigest()})
            if not same:
                dumps[name + "_candidate"] = got
                dumps[name + "_baseline"] = ref
    result = {"scope": "CPU source regression only, not WRF fidelity or GPU bit identity",
              "base_ref": a.base_ref, "columns": int(fx["in_t"].shape[0]),
              "pass": all(row["byte_equal"] for row in rows), "rows": rows}
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    if dumps:
        np.savez_compressed(a.out.with_suffix(".npz"), **dumps)
    print(json.dumps({key: result[key] for key in ("scope", "columns", "pass")}))
    assert result["pass"], "CAM CPU source regression changed output bytes; see receipt"


if __name__ == "__main__":
    main()
