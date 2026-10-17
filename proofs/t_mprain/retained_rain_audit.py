"""Read-only attribution of total-versus-liquid precipitation bookkeeping.

The diagnostic projection is NOT a new forecast or a replacement D6 receipt.
WRF WDM/WSM/Morrison RAINNCV already includes frozen fallout. State.rain_acc
and the history writer use disjoint channels (liquid, snow, graupel, ice, hail).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from gpuwrf.io.netcdf_lock import Dataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    result = {"scope": "retained-field diagnostic; no model execution or gate substitution", "cases": {}}
    for option in (16, 24, 26, 40):
        case = root / f"mp{option}"
        rows = []
        for cpu in sorted((case / "cpu").glob("wrfout_d01_*")):
            gpu = case / "gpu" / cpu.name
            with Dataset(cpu) as ref, Dataset(gpu) as actual:
                names = [n for n in ("SNOWNC", "GRAUPELNC", "HAILNC") if n in actual.variables]
                target = np.asarray(ref["RAINNC"][:], dtype=np.float64)
                old = np.asarray(actual["RAINNC"][:], dtype=np.float64)
                frozen = sum((np.asarray(actual[n][:], dtype=np.float64) for n in names), np.zeros_like(old))
                projected = old - frozen
                assert all(np.isfinite(a).all() for a in (target, old, frozen, projected))
                rmse = lambda a: float(np.sqrt(np.mean((a - target) ** 2)))
                rows.append(dict(frame=cpu.name, cpu_sha256=digest(cpu), gpu_sha256=digest(gpu),
                                 frozen_fields=names, raw_rmse_mm=rmse(old),
                                 projected_rmse_mm=rmse(projected),
                                 raw_bias_mm=float(np.mean(old-target)),
                                 projected_bias_mm=float(np.mean(projected-target)),
                                 frozen_max_mm=float(frozen.max()), projected_min_mm=float(projected.min())))
        assert len(rows) == 25, (option, len(rows))
        result["cases"][str(option)] = dict(frames=rows, max_raw_rmse_mm=max(r["raw_rmse_mm"] for r in rows),
                                             max_projected_rmse_mm=max(r["projected_rmse_mm"] for r in rows),
                                             projected_above_frozen_1mm=sum(r["projected_rmse_mm"] > 1 for r in rows))
    return result


if __name__ == "__main__":
    out = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: {n: v for n, v in r.items() if n != "frames"} for k, r in out["cases"].items()}, indent=2))
