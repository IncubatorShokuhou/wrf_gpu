"""Structural check of a twin setup: twin tau0 history variables vs the CPU reference tau0 frames (domain map n18k5),
bitwise equality per variable only (no metric values; guarded reads A-O5-9NEST-148). Global attributes compared by name.
usage: python t0_identity.py <twin_root> <label>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gread148 as g
REF = "<DATA_ROOT>/alisios/scratch/n148-cpuref-20261018_12z/cpu"
MAP = {"d01": "d01", "d02": "d02", "d03": "d04", "d04": "d05", "d05": "d06"}
root = Path(sys.argv[1])
import os
pairs = {tw: (root / "cpu" / f"wrfout_{tw}_2026-10-18_18:00:00", os.path.realpath(f"{REF}/wrfout_{rf}_2026-10-18_18:00:00")) for tw, rf in MAP.items()}
twin_arrays = {}
import netCDF4
for tw, (tp, _) in pairs.items():   # own files read BEFORE the guarded session
    with netCDF4.Dataset(tp) as nc:
        nc.set_auto_mask(False)
        twin_arrays[tw] = ({k: nc.variables[k][:] for k in nc.variables}, {k: nc.getncattr(k) for k in nc.ncattrs()})
sess, rdir, listed = g.session([rp for _, rp in pairs.values()], sys.argv[2])
out = {}
with sess:
    for tw, (_, rp) in pairs.items():
        tv, ta = twin_arrays[tw]
        with netCDF4.Dataset(rp) as nc:
            nc.set_auto_mask(False)
            diff = [k for k in nc.variables if k in tv and not np.array_equal(np.asarray(nc.variables[k][:]), np.asarray(tv[k]))]
            only_ref = [k for k in nc.variables if k not in tv]
            only_twin = [k for k in tv if k not in nc.variables]
            adiff = sorted(k for k in nc.ncattrs() if k in ta and str(nc.getncattr(k)) != str(ta[k]))
            out[tw] = {"ref": MAP[tw], "n_vars": len(tv), "vars_equal": len(tv) - len(diff) - len(only_twin), "vars_differ": diff,
                       "only_ref": only_ref, "only_twin": only_twin, "global_attrs_differ": adiff}
g.close_receipt(sess, rdir, {"result": out})
print(json.dumps({k: {kk: v[kk] for kk in ("ref", "n_vars", "vars_equal", "vars_differ", "global_attrs_differ")} for k, v in out.items()}, indent=0))
