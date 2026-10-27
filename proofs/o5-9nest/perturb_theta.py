"""Round-off seed for a realization twin: copy wrfinput_d0N and change ONLY variable T (perturbation theta, theta - 300 K) by
+-1 ulp of the REAL total theta (fl32(300 + T) -> next float32 up/down, random sign per cell, numpy seed 148 + N).

1 ulp of total theta (~3.05e-5 K at 300 K) survives in both a perturbation-theta store (CPU-WRF t_2) and a total-theta
REAL store; 1 ulp of T alone (~1e-6 K) would round away in a total-theta REAL carry. Every other variable/attribute stays
byte-for-byte as in the source (the copy is modified in place with netCDF4; T keeps dtype/shape/attributes).

usage: JAX_PLATFORMS=cpu python perturb_theta.py <src_dir> <dst_dir> [d01 ... d09]   (dst_dir must not exist)
Writes <dst_dir>/PERTURB.json (per domain: src/dst sha256, cells, |dT| max, sign balance).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def perturb(src, dst, n):
    import netCDF4
    shutil.copyfile(src, dst)
    with netCDF4.Dataset(dst, "r+") as nc:
        nc.set_auto_maskandscale(False)
        v = nc.variables["T"]
        t = v[:]
        if t.dtype != np.float32 or not np.isfinite(t).all():
            raise ValueError(f"{src}: T must be finite float32, got {t.dtype}")
        total = np.float32(300.0) + t
        sign = np.random.default_rng(148 + n).integers(0, 2, size=t.shape).astype(bool)
        bumped = np.where(sign, np.nextafter(total, np.float32(np.inf)), np.nextafter(total, np.float32(-np.inf)))
        new = (bumped - np.float32(300.0)).astype(np.float32)
        d = new.astype(np.float64) - t.astype(np.float64)
        if not (np.abs(d) > 0).all() or np.abs(d).max() > 1e-4:
            raise ValueError(f"{src}: perturbation not in (0, 1e-4] K everywhere")
        v[:] = new
    return {"src": str(src), "src_sha256": sha(src), "dst": str(dst), "dst_sha256": sha(dst), "cells": int(t.size),
            "abs_dT_max": float(np.abs(d).max()), "abs_dT_min": float(np.abs(d).min()), "frac_positive": float(sign.mean())}


def main(src_dir, dst_dir, doms):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=False)
    rec = {"rule": __doc__.split("\n\n")[0], "domains": {}}
    for d in doms:
        n = int(d[1:])
        rec["domains"][d] = perturb(src_dir / f"wrfinput_{d}", dst_dir / f"wrfinput_{d}", n)
    (dst_dir / "PERTURB.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({d: (r["abs_dT_max"], r["frac_positive"]) for d, r in rec["domains"].items()}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or [f"d0{i}" for i in range(1, 10)])
