"""Make a perturbed copy of a WRF case: T (perturbation potential temperature) in wrfinput_d0* moved by exactly ONE float32
ulp per cell, sign drawn from a seeded RNG (round-off-scale IC perturbation, the size of an fp32 operation-order difference).

usage: JAX_PLATFORMS=cpu python3 perturb_t.py SRC_CASE_DIR OUT_CASE_DIR SEED
- wrfinput_d0* of SRC are copied byte-for-byte (seed > 0) and T of each copy is rewritten in place; every other file (and,
  for seed 0 = unperturbed control, every file) is HARD-LINKED (read-only use by the GPU CLI; sha256 recorded).
- writes OUT.PERTURB.json (beside OUT, never inside the case dir): per file sha256 before/after, per domain cells changed, max |dT| (K), max |dT|/ulp == 1.
No other variable, attribute or file is touched (checked after the write: every other variable byte-identical).
"""
import hashlib, json, os, shutil, sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


src, out, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
assert not out.exists(), f"refusing existing {out}"
out.mkdir(parents=True)
rec = {"src": str(src), "seed": seed, "rule": "T += sign*ulp(float32) per cell (seed 0: none)", "files": {}, "domains": {}}
for p in sorted(src.iterdir()):
    if p.is_file():
        if seed > 0 and p.name.startswith("wrfinput_d0"):
            shutil.copyfile(p, out / p.name)
        else:
            os.link(p, out / p.name)
        rec["files"][p.name] = {"src_sha256": sha(p)}
rng = np.random.default_rng(seed)
for p in sorted(out.glob("wrfinput_d0*")):
    if seed == 0:
        break
    with Dataset(p) as ds:
        before = {v: np.array(ds[v][:]) for v in ds.variables if v != "T"}
    with Dataset(p, "r+") as ds:
        t = np.asarray(ds["T"][:], dtype=np.float32)
        assert np.isfinite(t).all()
        sign = rng.choice(np.array([-1, 1], dtype=np.int8), size=t.shape)
        tp = np.where(sign > 0, np.nextafter(t, np.float32(np.inf)), np.nextafter(t, np.float32(-np.inf))).astype(np.float32)
        ds["T"][:] = tp
    with Dataset(p) as ds:
        t2 = np.asarray(ds["T"][:], dtype=np.float32)
        same = all(np.array(ds[v][:]).tobytes() == before[v].tobytes() for v in before)
    ulp = np.abs(np.spacing(t))
    assert same and (t2 != t).all() and np.array_equal(t2, tp), p
    rec["domains"][p.name] = {"cells": int(t.size), "changed": int((t2 != t).sum()),
                              "max_abs_dT_K": float(np.abs(t2.astype(np.float64) - t).max()),
                              "max_dT_over_ulp": float((np.abs(t2.astype(np.float64) - t) / ulp).max()),
                              "other_variables_identical": same}
for n in rec["files"]:
    rec["files"][n]["out_sha256"] = sha(out / n)
(out.parent / f"{out.name}.PERTURB.json").write_text(json.dumps(rec, indent=1) + "\n")
print(json.dumps(rec["domains"]))
