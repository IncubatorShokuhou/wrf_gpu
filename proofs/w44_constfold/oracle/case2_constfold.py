"""w44-constfold CASE2 column oracle: port mym_condensation CASE(2) vs PRISTINE WRF 4.7.1 mym_condensation.

Harness, driver and build script copied unchanged from lane o5-n333 (proofs/o5_n333/oracle, CPDF01-ORACLE,
2026-10-24): the oracle links the UNEDITED pristine module_bl_mynnedmf.o.  This script scores CASE 2 (the
release bl_mynn_cloudpdf) native REAL32 with GPUWRF_WRF_REAL_CONSTFOLD OFF (main) and ON (this lane),
bitwise gate (no tolerance), per input set.

usage: JAX_PLATFORMS=cpu python case2_constfold.py ORACLE_EXE OUT_JSON LABEL RESTART [RESTART ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import cloudpdf_oracle as O  # noqa: E402
from gpuwrf.contracts import real_constfold as rc  # noqa: E402
from gpuwrf.physics import mynn_sgs_cloud as S  # noqa: E402

exe, out_json, label, *paths = sys.argv[1:]
c, meta = O.load_restart_columns(paths)
ref = O.run_oracle(exe, c, 2)
dz = c["dz"]
zw = np.concatenate((np.zeros_like(dz[:, :1]), np.cumsum(dz, 1)[:, :-1]), 1)
aloft = (zw + 0.5 * dz) > np.maximum(10.0, c["pblh"])[:, None]
icy = (c["qi"] + c["qs"]) > 1e-9
result = {"label": label, "inputs": meta, "assert_src": S.__file__, "columns": int(dz.shape[0]),
          "icy_cells_aloft": int((icy & aloft).sum()), "arms": {}}
for arm, flag in (("OFF_main", False), ("ON_constfold", True)):
    rc.ENABLED = flag
    import jax
    jax.clear_caches()  # E161: no reuse of a trace across arms
    port = O.run_port(c, 2, True)
    arm_out = {}
    for f in ("qc_bl", "qi_bl", "cldfra_bl"):
        r, x = ref[f].astype(np.float64), port[f]
        finite = bool(np.isfinite(r).all() and np.isfinite(x).all())
        miss = x != r
        arm_out[f] = {"finite": finite, "mismatch_cells": int(miss.sum()), "cells": int(r.size),
                      "max_abs": float(np.abs(x - r).max()), "ref_nonzero": int((r != 0).sum()),
                      "mismatch_ice_aloft": int((miss & icy & aloft).sum())}
    result["arms"][arm] = arm_out
    print(label, arm, {f: (v["mismatch_cells"], v["max_abs"]) for f, v in arm_out.items()}, flush=True)
Path(out_json).write_text(json.dumps(result, indent=1) + "\n")
