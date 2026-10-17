"""F20 fixture: real CPU-WRF columns as MYNN-driver operands (PROD d01/d02 + WN3 0227 d03, day + night).

Each set is one original CPU-WRF history frame. The operands are the history state at that time arranged
as module_pbl_driver hands them to mynnedmf_driver: ``w`` keeps its w-stagger (grid%w_2, nz+1 faces, the
surface face included), u/v are A-grid means, th is dry theta (T+300; THM is the moist one), p/exner from
P+PB, dz8w from PH+PHB with g=9.81. Both sides of every gate receive these identical arrays, so the exact
phy_prep pressure choice does not matter for the comparisons. Column choice (disclosed, not a happy path):
the 16 columns where a face-w vs mass-averaged-w taper estimate (history PBLH as cut) differs most, plus 16
evenly spaced columns of the whole domain.

Run (CPU only): JAX_PLATFORMS=cpu python proofs/t_cfix_mynn/extract_columns.py <out.npz>
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

PROD = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/"
            "alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case")
WN3 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
SETS = {
    "prod_d01_day": (PROD / "wrfout_d01_2026-07-26_13:00:18", 54.0, 9000.0),
    "prod_d01_night": (PROD / "wrfout_d01_2026-07-27_02:00:36", 54.0, 9000.0),
    "prod_d02_day": (PROD / "wrfout_d02_2026-07-26_13:00:00", 18.0, 3000.0),
    "prod_d02_night": (PROD / "wrfout_d02_2026-07-27_02:00:00", 18.0, 3000.0),
    "wn3_d03_day": (WN3 / "wrfout_d03_2026-02-28_13:00:00", 6.0, 1000.0),
    "wn3_d03_night": (WN3 / "wrfout_d03_2026-03-01_02:00:00", 6.0, 1000.0),
}
G, RD, CP = np.float32(9.81), np.float32(287.0), np.float32(7.0 * 287.0 / 2.0)
N_TAPER, N_GRID = 16, 16


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _psig(w, dz, pblh):
    """Numpy taper estimate (DMP_mf :5855-5878) for column SELECTION only."""
    nz = dz.shape[-1]
    zagl = np.cumsum(dz, axis=-1) - 0.5 * dz
    below = (zagl <= pblh[:, None] + 500.0) & (np.arange(nz) < nz - 1)
    wpbl = np.where(w < 0.0, 2.0 * w, w)
    maxw = np.max(np.where(below, np.abs(wpbl), 0.0), axis=-1)
    return np.maximum(0.0, 1.0 - np.maximum(0.0, maxw - 1.0))


def extract(path, dt, dx):
    with Dataset(path) as nc:
        v = lambda name: np.asarray(nc.variables[name][0], dtype=np.float32)
        u, vv, w = v("U"), v("V"), v("W")
        nz, ny, nx = v("T").shape
        col = lambda a: a.reshape(a.shape[0], ny * nx).T  # (ncol, nlev)
        sfc = lambda a: a.reshape(ny * nx)
        fields = dict(
            u=col(np.float32(0.5) * (u[:, :, :-1] + u[:, :, 1:])),
            v=col(np.float32(0.5) * (vv[:, :-1, :] + vv[:, 1:, :])),
            w=col(w),
            th=col(v("T") + np.float32(300.0)),
            p=col(v("P") + v("PB")),
            qv=col(v("QVAPOR")), qc=col(v("QCLOUD")), qi=col(v("QICE")), qs=col(v("QSNOW")),
            qni=col(v("QNICE")), qke=col(v("QKE")), el_pbl=col(v("EL_PBL")[:nz]),
            xland=sfc(v("XLAND")), tsk=sfc(v("TSK")), psfc=sfc(v("PSFC")), ust=sfc(v("UST")),
            ch=sfc(v("CH")), hfx=sfc(v("HFX")), qfx=sfc(v("QFX")), pblh_hist=sfc(v("PBLH")),
            hgt=sfc(v("HGT")),
        )
        phi = v("PH") + v("PHB")
        fields["dz"] = col((phi[1:] - phi[:-1]) / G)
    fields["exner"] = (fields["p"] / np.float32(100000.0)) ** (RD / CP)
    fields["t3d"] = fields["th"] * fields["exner"]
    qv = fields["qv"]
    fields["rho"] = (fields["p"] / (RD * fields["t3d"] * (np.float32(1.0) + np.float32(461.6 / 287.0) * qv))
                     * (np.float32(1.0) + qv)).astype(np.float32)
    fields["wspd"] = np.maximum(np.hypot(fields["u"][:, 0], fields["v"][:, 0]), np.float32(0.1))
    face = _psig(fields["w"][:, :-1], fields["dz"], fields["pblh_hist"])
    mass = _psig(np.float32(0.5) * (fields["w"][:, :-1] + fields["w"][:, 1:]), fields["dz"], fields["pblh_hist"])
    gap = np.abs(face - mass)
    taper = [int(i) for i in np.argsort(-gap, kind="stable")[:N_TAPER] if gap[i] > 0.0]
    grid = [int(i) for i in np.linspace(0, ny * nx - 1, N_GRID).round().astype(int) if i not in taper]
    pick = np.asarray(taper + grid, dtype=np.int64)
    out = {k: np.ascontiguousarray(a[pick]) for k, a in fields.items()}
    out["columns"] = pick
    meta = dict(source=str(path), sha256=_sha(path), shape=[int(nz), int(ny), int(nx)], dt=dt, dx=dx,
                n_taper_selected=len(taper), n_grid=len(grid),
                n_domain_taper_estimate=int(np.count_nonzero(gap > 0.0)))
    return out, meta


def main(out_path):
    arrays, manifest = {}, {}
    for name, (path, dt, dx) in SETS.items():
        out, meta = extract(path, dt, dx)
        for key, value in out.items():
            arrays[f"{name}__{key}"] = value
        manifest[name] = meta
        print(name, meta["n_taper_selected"], meta["n_domain_taper_estimate"], flush=True)
    out_path = Path(out_path)
    np.savez_compressed(out_path, **arrays)
    manifest["extractor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out_path.with_suffix(".json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
