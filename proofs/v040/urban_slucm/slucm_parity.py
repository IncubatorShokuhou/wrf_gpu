"""SLUCM JAX kernel vs pristine-WRF oracle savepoints (step-by-step parity).

Replays every oracle regime through :func:`gpuwrf.physics.urban_slucm.noahmp_urban`
(+ ``urban_var_init`` and the surface-driver 2-m/10-m overwrite) with the exact
REAL inputs the oracle dumped, for all 30 steps, and reports per-field max
absolute / relative error over every step.

Usage: JAX_PLATFORMS=cpu python slucm_parity.py [fp64|fp32] [default|alt1|alt2] [--json out]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics import urban_slucm as _us  # noqa: E402

HERE = Path(__file__).resolve().parent
SAVE = HERE / "savepoints"

STATE_FIELDS = {"TR": "tr", "TB": "tb", "TG": "tg", "TC": "tc", "QC": "qc", "UC": "uc",
                "CMR": "cmr", "CHR": "chr", "CMC": "cmc", "CHC": "chc", "DRELR": "drelr",
                "DRELB": "drelb", "DRELG": "drelg", "FLXHUMR": "flxhumr", "FLXHUMB": "flxhumb",
                "FLXHUMG": "flxhumg", "XXXR": "xxxr", "XXXC": "xxxc"}
LAYER_FIELDS = {"TRL": "trl", "TBL": "tbl", "TGL": "tgl"}
LSM_FIELDS = {"TSK": "tsk", "HFX": "hfx", "QFX": "qfx", "LH": "lh", "GRDFLX": "grdflx",
              "ALBEDO": "albedo", "QSFC": "qsfc", "UST": "ust"}
DIAG_FIELDS = {"TS_URB": "ts_urb", "SH_URB": "sh_urb", "LH_URB": "lh_urb", "G_URB": "g_urb",
               "RN_URB": "rn_urb", "PSIM_URB": "psim_urb", "PSIH_URB": "psih_urb",
               "GZ1OZ0_URB": "gz1oz0_urb", "U10_URB": "u10_urb", "V10_URB": "v10_urb",
               "TH2_URB": "th2_urb", "Q2_URB": "q2_urb", "UST_URB": "ust_urb", "AKMS_URB": "akms_urb",
               "CHS": "chs", "CHS2": "chs2", "CQS2": "cqs2"}
SFC_FIELDS = ("T2", "Q2", "U10", "V10")


def cal_mon_day(julday: int, julyr: int) -> int:
    month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if julyr % 4 == 0:
        month[1] = 29
    d = julday
    for i, n in enumerate(month):
        if d > n:
            d -= n
        else:
            return i + 1
    return 12


def run(prec: str = "fp64", tbl: str = "default", savepoint_dir: Path = SAVE, mutate=None, regimes: str = "",
        kernel=None):
    """``regimes`` = "" (25 base regimes, 30 steps) or "_branch" (12 branch-coverage regimes, 60 steps);
    ``kernel`` = an alternative urban_slucm module (in-process source mutants)."""
    us = _us if kernel is None else kernel
    z = np.load(savepoint_dir / f"slucm_{prec}_{tbl}{regimes}.npz")
    dt_np = np.float64 if prec == "fp64" else np.float32
    table = us.parse_urbparm(savepoint_dir / f"URBPARM.TBL.{tbl}", dtype=dt_np)
    us.check_supported_options(table)
    if mutate is not None:
        table = mutate(table)

    def inp(k):
        return jnp.asarray(z["in_" + k].astype(dt_np))

    n = len(z["regime_names"])
    mask = jnp.ones(n, bool)
    tslb = jnp.stack([inp(f"tslb{k}") for k in range(1, 5)], axis=-1)
    state, frc, utype = us.urban_var_init(table, urban_mask=mask, tsk=inp("tsk"), tslb=tslb,
                                          frc_urb_in=inp("frc_in"))
    dts = z["in_dt"].astype(dt_np)
    assert np.all(dts == dts[0])
    dt = float(dts[0])
    jmonth = jnp.asarray([cal_mon_day(int(j), int(y)) for j, y in zip(z["in_julian"], z["in_julyr"])],
                         jnp.int32)
    lsm0 = us.UrbanLSMFields(*[inp(v) for v in LSM_FIELDS.values()])
    c = np.dtype(dt_np).type
    rcp = c(287.0) / (c(7.0) * c(287.0) / c(2.0))

    @jax.jit
    def step(state):
        lsm, st, diag = us.noahmp_urban(
            table, urban_mask=mask, frc_urb=frc, utype=utype, t3d1=inp("t3d1"), qv1=inp("qv1"),
            u1=inp("u1"), v1=inp("v1"), swdown=inp("swdown"), glw=inp("glw"), p8w1=inp("p8w1"),
            p8w2=inp("p8w2"), rainbl=inp("rainbl"), dz8w1=inp("dz8w1"), dt=dt, chs=inp("chs"),
            chs2=inp("chs2"), cqs2=inp("cqs2"), lsm=lsm0, state=state, omg=inp("omg"), jmonth=jmonth)
        t2, q2, u10, v10 = us.urban_surface_diagnostics(
            urban_mask=mask, frc_urb=frc, fveg=inp("fveg"), t2mv=inp("t2mv"), t2mb=inp("t2mb"),
            q2mv=inp("q2mv"), q2mb=inp("q2mb"), psfc=inp("psfc"), diag=diag, t2=inp("t2mv"),
            q2=inp("q2mv"), u10=inp("u1"), v10=inp("v1"), rcp=rcp)
        return lsm, st, diag, (t2, q2, u10, v10)

    nsteps = z["TR"].shape[1] - 1
    errs: dict[str, dict] = {}

    def acc(name, got, ref, s):
        got = np.asarray(got, dtype=np.float64)
        ref = np.asarray(ref, dtype=np.float64)
        ae = np.abs(got - ref)
        scale = np.maximum(np.abs(ref), 1e-30)
        rel = ae / scale
        e = errs.setdefault(name, {"max_abs": 0.0, "max_rel": 0.0, "worst_step": 0,
                                   "nonfinite": 0, "ref_absmax": 0.0})
        e["nonfinite"] += int((~np.isfinite(got)).sum())
        if ae.max() > e["max_abs"]:
            e["max_abs"] = float(ae.max())
            e["worst_step"] = s
        e["max_rel"] = max(e["max_rel"], float(rel.max()))
        e["ref_absmax"] = max(e["ref_absmax"], float(np.abs(ref).max()))

    # init state (step 0)
    for k, a in STATE_FIELDS.items():
        if k in z.files:
            acc(k, getattr(state, a), z[k][:, 0], 0)
    for k, a in LAYER_FIELDS.items():
        for l in range(4):
            acc(f"{k}{l + 1}", getattr(state, a)[:, l], z[f"{k}{l + 1}"][:, 0], 0)
    acc("FRC_URB", frc, z["in_frc_urb"], 0)

    for s in range(1, nsteps + 1):
        lsm, state, diag, sfc = step(state)
        for k, a in STATE_FIELDS.items():
            if k in z.files:
                acc(k, getattr(state, a), z[k][:, s], s)
        for k, a in LAYER_FIELDS.items():
            for l in range(4):
                acc(f"{k}{l + 1}", getattr(state, a)[:, l], z[f"{k}{l + 1}"][:, s], s)
        for k, a in LSM_FIELDS.items():
            acc(k, getattr(lsm, a), z[k][:, s], s)
        for k, a in DIAG_FIELDS.items():
            acc(k, getattr(diag, a), z[k][:, s], s)
        for k, v in zip(SFC_FIELDS, sfc):
            acc(k, v, z[k][:, s], s)
    return errs


def main(argv):
    prec = argv[1] if len(argv) > 1 else "fp64"
    tbl = argv[2] if len(argv) > 2 else "default"
    regimes = "_branch" if "--branch" in argv else ""
    errs = run(prec, tbl, regimes=regimes)
    worst_rel = max(e["max_rel"] for e in errs.values())
    print(f"{prec}/{tbl}: fields={len(errs)} worst max_rel={worst_rel:.3e}")
    for k, e in sorted(errs.items(), key=lambda kv: -kv[1]["max_rel"])[:12]:
        print(f"  {k:12s} max_abs={e['max_abs']:.3e} max_rel={e['max_rel']:.3e} "
              f"step={e['worst_step']} |ref|max={e['ref_absmax']:.3e} nonfinite={e['nonfinite']}")
    if "--json" in argv:
        out = argv[argv.index("--json") + 1]
        Path(out).write_text(json.dumps({"prec": prec, "table": tbl, "errors": errs}, indent=1))
    return errs


if __name__ == "__main__":
    main(sys.argv)
