"""GPUWRF_NOAH_URBAN_SOIL_PARAMS gate fixture (V033-W2, b-thompson/b-noahmp). CPU only, no JAX.

Columns: real CPU-WRF 0227 d03 land columns (wg_20260227_18z_a1, state at 2026-02-28_00, lowest-level forcing held from
02-28_01, dt 6 s): 36 urban (IVGTYP 13) + 24 non-urban of the 531-cell low-land mask (LANDMASK > .5, HGT < 300 m,
ring >= 5), built by build_urban_cols.py (lane SOIL/). Oracle: the wired pristine NOAHMP_SFLX offline driver
(proofs/noahmp/noahmp_offline_driver.F90, NOAHMP_WIRE_SOIL=1 NOAHMP_WIRE_FICEOLD=1 = WRF driver wiring incl.
TRANSFER_MP_PARAMETERS' urban soil block, module_sf_noahmpdrv.F:1694-1700), REAL4. Stores the first NSTEPS steps of
TSLB(1:4), SH2O(1:4) and SMC(1:4) per column. Columns also carry 8 wet and 8 frozen-wet urban
variants (real columns, moist / partly frozen profile + rain; FRZX-order coverage, review-s2small 2026-10-07T04:49Z).
usage: make_fixture.py <wired_dir> <nsteps> <out_json> <out_npz>"""
import json, sys
import numpy as np
w, nsteps, oj, on = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cols = json.load(open(f"{w}/columns.json"))
n = len(cols["columns"])
T = np.zeros((nsteps, n, 4), np.float32); S = np.zeros((nsteps, n, 4), np.float32); M = np.zeros((nsteps, n, 4), np.float32)
for line in open(f"{w}/noahmp_steps.out"):
    p = line.split(); ic, it = int(p[1]) - 1, int(p[2])
    if it <= nsteps:
        v = [float(x) for x in p[4:]]
        T[it - 1, ic] = v[5:9]; S[it - 1, ic] = v[15:19]; M[it - 1, ic] = v[19:23]
assert np.isfinite(T).all() and (T > 200).all()
json.dump({**cols, "oracle": "wired pristine NOAHMP_SFLX (WIRE_SOIL=1, WIRE_FICEOLD=1), REAL4", "nsteps": nsteps}, open(oj, "w"))
np.savez_compressed(on, tslb=T, sh2o=S, smc=M)
print(n, "columns,", nsteps, "steps ->", oj, on)
