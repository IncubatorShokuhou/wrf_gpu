"""V3b pristine oracle (lane a1-io): real.exe hydrostatic block (real_hydro_column.F90, REAL*4, all 6 moist species)
vs state_io.rebalance_increment (float64, qv-only Δ recursion, α at the model p) on real CPU-WRF 0503 d01 columns.

Increments compared: F(an) − F(bg) of the verbatim block vs the adapter, for θ-only, qv-only and combined tapered
blobs (lowest 12 levels). Also reports the port's balance() (qv-only loading) vs the pristine block on the
background state. Writes PRISTINE_ORACLE.json next to this file. Build/scratch dir: <USER_HOME>/wrf_gpu2_lanes/a1-io/.
  PYTHONPATH=src JAX_PLATFORMS=cpu nice -n 19 taskset -c 8,9,24,25 python3 tests/a1_da/oracles/pristine_hydro/run_oracle.py
(moved from proofs/a1_da/io/pristine_hydro on lane/a1-da 1131707318; the receipt there remains the historical record)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from gpuwrf.assimilation.letkf import state_io as sio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = Path(os.environ.get("A1_PRISTINE_ORACLE_WORK", "<USER_HOME>/wrf_gpu2_lanes/a1-io/pristine_hydro"))  # tests set WORK
FC = "<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran"
RUN = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260503_18z_a1/run/run")
T_A = "2026-05-04_12:00:00"
MOIST = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
BOX = (slice(25, 46), slice(55, 86))  # rows, cols around the blob centre (35, 70)


def build(real8=False):
    WORK.mkdir(parents=True, exist_ok=True)
    exe = WORK / ("real_hydro_column_r8" if real8 else "real_hydro_column")
    flags = ["-fdefault-real-8"] if real8 else []
    subprocess.run([FC, "-O0", *flags, "-o", str(exe), str(HERE / "real_hydro_column.F90")], check=True)
    return exe


def write_input(path, geo, cols, real8=False):
    nz = geo["nz"]
    rt = np.float64 if real8 else np.float32
    f32 = lambda a: np.asarray(a, rt).ravel()  # noqa: E731
    with open(path, "wb") as fh:
        fh.write(np.asarray([nz, len(cols), len(MOIST)], np.int32).tobytes())
        for n in ("C1F", "C2F", "C3F", "C4F", "C1H", "C2H", "C3H", "C4H", "RDNW", "RDN", "DNW"):
            fh.write(f32(geo[n]).tobytes())
        fh.write(f32([geo["P_TOP"]]).tobytes())
        for c in cols:  # Fortran column-major moist(nz, nmoist): write species-major
            fh.write(f32([c["MUB"], c["MU"]]).tobytes())
            for n in ("PB", "ALB", "T"):
                fh.write(f32(c[n]).tobytes())
            fh.write(np.asarray(np.stack([c[m] for m in MOIST], axis=1), rt).ravel(order="F").tobytes())
            fh.write(f32(c["PHB"]).tobytes())


def run_fortran(exe, geo, state, tag, real8=False):
    j0, i0 = BOX
    cols = []
    for j in range(j0.start, j0.stop):
        for i in range(i0.start, i0.stop):
            cols.append({"MUB": geo["MUB"][j, i], "MU": state["MU"][j, i], "PB": geo["PB"][:, j, i],
                         "ALB": geo["ALB"][:, j, i], "T": state["T"][:, j, i], "PHB": geo["PHB"][:, j, i],
                         **{m: state[m][:, j, i] for m in MOIST}})
    fin, fout = WORK / f"in_{tag}.bin", WORK / f"out_{tag}.bin"
    write_input(fin, geo, cols, real8)
    subprocess.run([str(exe), str(fin), str(fout)], check=True)
    nz = geo["nz"]
    raw = np.fromfile(fout, np.float64 if real8 else np.float32).reshape(len(cols), 4 * nz + 1)
    fin.unlink()
    fout.unlink()
    shape = (j0.stop - j0.start, i0.stop - i0.start)
    out = {"P": raw[:, :nz], "PH": raw[:, nz:2 * nz + 1], "AL": raw[:, 2 * nz + 1:3 * nz + 1]}
    return {k: np.moveaxis(v.reshape(shape + (v.shape[-1],)), -1, 0).astype(np.float64) for k, v in out.items()}


def evaluate(adapter=sio.rebalance_increment, t=None):
    """Return the comparison record; ``adapter(bg, an, geo, return_increments=True)['increments']``."""
    if t is None:
        spec = importlib.util.spec_from_file_location("a1io_tests", ROOT / "tests/a1_da/test_state_io.py")
        t = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(t)
    exes = {"real4": build(False), "real8": build(True)}
    wrfout = RUN / f"wrfout_d01_{T_A}"
    geo = sio.read_geometry(wrfout, base_wrfinput=RUN / "wrfinput_d01")
    bg = sio.read_frame(wrfout, T_A, ("U", "V", "T", "QVAPOR", "MU", "P", "PH") + MOIST[1:])
    # Box + increment centred on the cloudiest interior column, so the frozen-hydrometeor loading is exercised
    # (E154: an oracle without hydrometeors would be vacuous for the qv-only Δp claim).
    global BOX
    hyd = sum(bg[m].astype(np.float64) for m in MOIST[1:]).sum(axis=0)
    taper = sio.boundary_taper(*hyd.shape, 5, 5)
    jc, ic = np.unravel_index(np.argmax(np.where(taper == 1, hyd, -1.0)), hyd.shape)
    jc, ic = int(np.clip(jc, 20, hyd.shape[0] - 21)), int(np.clip(ic, 25, hyd.shape[1] - 26))
    BOX = (slice(jc - 10, jc + 11), slice(ic - 15, ic + 16))
    bump = t.blob(hyd.shape, centre=(jc, ic), width=6.0) * taper
    low = np.arange(geo["nz"])[:, None, None] < 12
    full = {"T": (bg["T"].astype(np.float64) + 0.5 * bump * low).astype(np.float32),
            "QVAPOR": (bg["QVAPOR"].astype(np.float64) * (1.0 + 0.05 * bump * low)).astype(np.float32)}
    cases = {"theta_only": {**bg, "T": full["T"]}, "qv_only": {**bg, "QVAPOR": full["QVAPOR"]},
             "combined": {**bg, "T": full["T"], "QVAPOR": full["QVAPOR"]}}
    res = {"oracle": "verbatim real.exe :3923-4036 (hypsometric_opt 2), 6 moist species; compiled REAL*4 "
                     "(WRF precision) and -fdefault-real-8 (formulation check without REAL*4 noise)",
           "compiler": FC, "source_sha256": hashlib.sha256((HERE / "real_hydro_column.F90").read_bytes()).hexdigest(),
           "frame": str(wrfout), "box_rows_cols": [[BOX[0].start, BOX[0].stop], [BOX[1].start, BOX[1].stop]],
           "centre_ji": [jc, ic], "hydrometeor_max_in_box": float(sum(bg[m][(slice(None),) + BOX].astype(np.float64)
                                                                       for m in MOIST[1:]).max()),
           "hydrometeor_columns_in_box": int((hyd[BOX] > 0).sum()),
           "cases": {}}
    f_bg = {}
    for prec, exe in exes.items():
        f_bg[prec] = run_fortran(exe, geo, bg, f"bg_{prec}", prec == "real8")
    for name, an in cases.items():
        mine = adapter(bg, an, geo, return_increments=True)["increments"]
        res["cases"][name] = {}
        for prec, exe in exes.items():
            f_an = run_fortran(exe, geo, an, f"{name}_{prec}", prec == "real8")
            row = {}
            for field, key in (("P", "P"), ("PH", "PH"), ("AL", "ALPHA")):
                ref = f_an[field] - f_bg[prec][field]
                got = mine[key][(slice(None),) + BOX]
                scale = float(np.abs(ref).max())
                err = float(np.abs(got - ref).max())
                row[field] = {"oracle_max_abs": scale, "adapter_max_abs": float(np.abs(got).max()),
                              "max_abs_diff": err, "rel": err / scale if scale else 0.0}
            res["cases"][name][prec] = row
    f_bg = f_bg["real8"]
    port = sio.balance_state(bg["T"], bg["QVAPOR"], bg["MU"], geo)
    res["port_balance_qv_only_vs_pristine_6_species_background"] = {
        f: {"max_abs_diff": float(np.abs(port[f][(slice(None),) + BOX] - f_bg[f]).max()),
            "max_abs": float(np.abs(f_bg[f]).max())} for f in ("P", "PH", "AL")}
    return res


def main():
    from gpuwrf.assimilation.letkf import alisios_guard
    from gpuwrf.assimilation.letkf.ensemble import run_days_from_name
    days = run_days_from_name(RUN)
    reads = {RUN / f"wrfout_d01_{T_A}": [T_A[:10].replace("-", "")], RUN / "wrfinput_d01": days[:1],
             RUN / "namelist.input": days}
    with alisios_guard.guarded_session({str(k): v for k, v in reads.items()}, ["20260503"], WORK / "guard") as session:
        res = evaluate()
    res["alisios_guard"] = session.receipt
    (HERE / "PRISTINE_ORACLE.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
