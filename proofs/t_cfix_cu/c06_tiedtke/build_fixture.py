"""Real-c06 cu6 Tiedtke oracle fixture (t-cfix-cu, c06 RAINNC blocker 2026-10-16).

Inputs: ``phyprep.npz`` written by ``phyprep_from_restart.py`` = WRF phy_prep (REAL4 transcription) of the
pristine serial c06 restart at 2023-01-15_15:00 + WRF's own held RQVFTEN / this step's RQVBLTEN / QFX
(auxhist2).  Reference: the UNMODIFIED module_cu_tiedtke.F through two whole-patch drivers built by
``build.sh``: ``tied_grid`` (CU_TIEDTKE over the full tile, i.e. TIECNV once per j-row = serial WRF) and
``tied_grid1`` (one CU_TIEDTKE call per column = a 1-column tile).

Patch = 9 real rows + 1 ASSEMBLED row of 42 real columns whose own first ascent finds no convection
(icum = 0: their single-column call differs from their row call, module_cu_tiedtke.F:959).  WRF returns
early for that row, although every one of its columns convects in its real row.

usage: python build_fixture.py PHYPREP_NPZ DRIVER_BUILD_DIR WORK_DIR OUT_NPZ
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PHY, BUILD, WORK, OUT = (Path(a) for a in sys.argv[1:5])
WORK.mkdir(parents=True, exist_ok=True)
f4 = np.float32
ROWS = (3, 21, 29, 36, 37, 38, 39, 40, 41)
FIELDS3 = ("u_phy", "v_phy", "w", "t_phy", "qv", "qc", "qi", "pi_phy", "rho", "qvften", "qvpblten", "dz8w",
           "p_hyd", "p_hyd_w")
NAMES = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN")

src = dict(np.load(PHY))
nz = src["t_phy"].shape[0]


def fort(a):
    a = np.asarray(a, f4)
    if a.shape[0] == nz:
        a = np.concatenate([a, np.zeros((1,) + a.shape[1:], f4)], axis=0)
    return np.ascontiguousarray(np.transpose(a, (1, 0, 2)))


def run(fields, exe, tag):
    ny, nx = fields["qfx"].shape
    work = WORK / tag
    work.mkdir(exist_ok=True)
    with open(work / "in.bin", "wb") as fh:
        np.array([nx, ny, nz, 2, 1], np.int32).tofile(fh)
        np.array([18.0], f4).tofile(fh)
        for name in FIELDS3:
            fort(fields[name]).tofile(fh)
        np.ascontiguousarray(fields["qfx"], f4).tofile(fh)
        np.ascontiguousarray(fields["xland"], f4).tofile(fh)
        np.concatenate([fields["znu"], [0.0]]).astype(f4).tofile(fh)
    subprocess.run([str(BUILD / exe)], cwd=work, check=True)
    raw = np.fromfile(work / ("out.bin" if exe == "tied_grid" else "out1.bin"), f4)
    n3 = nx * (nz + 1) * ny
    out = {n: raw[i * n3:(i + 1) * n3].reshape(ny, nz + 1, nx).transpose(1, 0, 2)[:nz] for i, n in enumerate(NAMES)}
    out["RAINCV"] = raw[6 * n3:6 * n3 + nx * ny].reshape(ny, nx)
    return out


def sub(fields, jj, ii):
    """Patch of columns (jj[r], ii[r, c]) -> fields with shape (.., len(jj), ii.shape[1])."""
    out = {}
    for name, a in fields.items():
        if name == "znu":
            out[name] = a
        elif a.ndim == 3:
            out[name] = a[:, jj[:, None], ii]
        else:
            out[name] = a[jj[:, None], ii]
    return out


fields = {n: src[n] for n in FIELDS3 + ("qfx", "xland", "znu")}
full_row = run(fields, "tied_grid", "full_row")
full_one = run(fields, "tied_grid1", "full_one")
assert np.array_equal(full_row["RAINCV"], src["ref_RAINCV"].astype(f4)), "row driver must reproduce the oracle run"
# icum = 0 columns of convective rows: the 1-column tile returns before any tendency (exact zeros in
# RTH/RU/RV; RQV keeps the q<->specific-humidity round trip), the real row convects there.  (Columns
# that differ only at round-off between the vectorised row call and the scalar 1-column call are NOT
# row-coupled and are excluded.)
silent_alone = np.all([np.all(full_one[n] == 0, axis=0) for n in ("RTHCUTEN", "RUCUTEN", "RVCUTEN")], axis=0)
active_in_row = np.max(np.abs(full_row["RTHCUTEN"]), axis=0) > 1.0e-6
coupled = np.argwhere(silent_alone & active_in_row)
assert len(coupled) >= 42, len(coupled)
null_cols = coupled[:42]

ny_full, nx_full = fields["qfx"].shape
jj = np.array(list(ROWS) + [0])
ii = np.tile(np.arange(nx_full), (len(jj), 1))
patch = sub(fields, jj, ii)
for name, a in patch.items():                       # last row = the assembled icum=0 row
    if name == "znu":
        continue
    if a.ndim == 3:
        a[:, -1, :] = fields[name][:, null_cols[:, 0], null_cols[:, 1]]
    else:
        a[-1, :] = fields[name][null_cols[:, 0], null_cols[:, 1]]
row = run(patch, "tied_grid", "patch_row")
one = run(patch, "tied_grid1", "patch_one")
# Row independence: the real rows of the patch reproduce the full-domain run exactly.
for n in NAMES + ("RAINCV",):
    ref = full_row[n][..., list(ROWS), :]
    got = row[n][..., :-1, :]
    assert np.array_equal(got, ref), n
# The assembled row: WRF returns before any tendency (all zero), the same columns convect in their real rows.
assert all(not np.any(row[n][..., -1, :]) for n in ("RTHCUTEN", "RUCUTEN", "RVCUTEN", "RAINCV"))
assert np.max(np.abs(row["RQVCUTEN"][:, -1, :])) < 1.0e-9   # REAL4 q/(1+q) round trip only
assert sum(np.any(full_row[n][:, null_cols[:, 0], null_cols[:, 1]] != 0) for n in NAMES) >= 1

np.savez_compressed(OUT, rows=np.array(ROWS), null_cols=null_cols, **{"in_" + k: np.asarray(v, f4) for k, v in patch.items()},
                    **{"row_" + k: v for k, v in row.items()}, **{"one_" + k: v for k, v in one.items()})
digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
summary = {"fixture": str(OUT), "sha256": digest, "rows": list(ROWS), "null_row_columns": null_cols.tolist(),
           "n_icum0_coupled_columns_full_domain": int(len(coupled)),
           "patch_row_raincv_pos": int((row["RAINCV"] > 0).sum()), "patch_one_raincv_pos": int((one["RAINCV"] > 0).sum()),
           "full_row_raincv_pos": int((full_row["RAINCV"] > 0).sum()), "full_one_raincv_pos": int((full_one["RAINCV"] > 0).sum())}
print(json.dumps(summary, indent=1))
