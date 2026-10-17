"""Independent c06 Tiedtke reference: WRF phy_prep (REAL4 numpy transcription of
module_big_step_utilities_em.F:4790-4970) on the 15:00 restart + UNMODIFIED module_cu_tiedtke.F
(whole-patch driver fort/tied_grid) -> compare with WRF's own R*CUTEN/RAINCV (aux 15:00:18).

Writes phyprep.npz (the inputs) for the port-adapter input comparison.
usage: python phyprep_from_restart.py RUN_DIR OUT_DIR [DRIVER_BUILD_DIR]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
FORT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parent / "fort"
f4 = np.float32
G, R_D, R_V, P1000 = f4(9.81), f4(287.0), f4(461.6), f4(1.0e5)
CP = f4(7.0) * R_D / f4(2.0)
RCP = R_D / CP
T0 = f4(300.0)

with Dataset(RUN / "wrfrst_d01_2023-01-15_15:00:00") as ds:
    r = {n: np.asarray(ds[n][0]).astype(f4) for n in (
        "THM_2", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP", "P", "PB", "ALT", "U_2", "V_2", "W_2",
        "PH_2", "PHB", "MU_2", "MUB", "C1H", "C2H", "DNW", "ZNU", "XLAND", "RQVFTEN", "P_HYD", "P_HYD_W",
        "T_PHY", "U_PHY", "V_PHY", "RHO")}
    p_top = f4(np.asarray(ds["P_TOP"][:]).ravel()[0])
with Dataset(RUN / "aux2_d01_2023-01-15_15:00:00") as ds:
    aux = {n: np.asarray(ds[n][:]).astype(f4) for n in (
        "RQVFTEN", "RQVBLTEN", "QFX", "RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN",
        "RAINCV", "PRATEC")}

qv = r["QVAPOR"]
nz, ny, nx = qv.shape
th_phy = (r["THM_2"] + T0) / (f4(1.0) + R_V / R_D * qv)
p_phy = r["P"] + r["PB"]
pi_phy = (p_phy / P1000) ** RCP
t_phy = th_phy * pi_phy
rho = f4(1.0) / r["ALT"] * (f4(1.0) + qv)
u_phy = f4(0.5) * (r["U_2"][:, :, :-1] + r["U_2"][:, :, 1:])
v_phy = f4(0.5) * (r["V_2"][:, :-1, :] + r["V_2"][:, 1:, :])
z_at_w = (r["PHB"] + r["PH_2"]) / G                      # (nz+1)
dz8w = np.zeros_like(z_at_w)
dz8w[:-1] = z_at_w[1:] - z_at_w[:-1]                     # dz8w(kte=kde) = 0
mut = r["MU_2"] + r["MUB"]
qtot = r["QVAPOR"] + r["QCLOUD"] + r["QRAIN"] + r["QICE"] + r["QSNOW"] + r["QGRAUP"]
p_hyd_w = np.zeros_like(z_at_w)
p_hyd_w[nz] = p_top
for k in range(nz - 1, -1, -1):
    p_hyd_w[k] = p_hyd_w[k + 1] - (f4(1.0) + qtot[k]) * (r["C1H"][k] * mut + r["C2H"][k]) * r["DNW"][k]
p_hyd = f4(0.5) * (p_hyd_w[:-1] + p_hyd_w[1:])

checks = {}
for name, mine in (("P_HYD", p_hyd), ("P_HYD_W", p_hyd_w), ("T_PHY", t_phy), ("U_PHY", u_phy),
                   ("V_PHY", v_phy), ("RHO", rho)):
    ref = r[name]
    checks[name + "_restart_vs_phyprep_maxabs"] = float(np.max(np.abs(ref.astype(np.float64) - mine)))


def fort(a):  # (k, j, i) numpy -> Fortran (i, k, j) memory, padded to nz+1 levels
    a = np.asarray(a, f4)
    if a.ndim == 3 and a.shape[0] == nz:
        a = np.concatenate([a, np.zeros((1,) + a.shape[1:], f4)], axis=0)
    return np.ascontiguousarray(np.transpose(a, (1, 0, 2)))  # (j, k, i) C order == (i, k, j) Fortran


def run_wrf_tiedtke(qvften, qvpblten, qfx, tag):
    work = OUT / f"fort_{tag}"
    work.mkdir(exist_ok=True)
    with open(work / "in.bin", "wb") as fh:
        np.array([nx, ny, nz, 2, 1], np.int32).tofile(fh)
        np.array([18.0], f4).tofile(fh)
        for a in (u_phy, v_phy, r["W_2"], t_phy, qv, r["QCLOUD"], r["QICE"], pi_phy, rho, qvften, qvpblten,
                  dz8w, p_hyd, p_hyd_w):
            fort(a).tofile(fh)
        np.ascontiguousarray(np.asarray(qfx, f4)).tofile(fh)          # (j, i) C == (i, j) Fortran
        np.ascontiguousarray(r["XLAND"]).tofile(fh)
        np.concatenate([r["ZNU"], [f4(0.0)]]).astype(f4).tofile(fh)
    subprocess.run(["taskset", "-c", "8,9", str(FORT / "tied_grid")], cwd=work, check=True)
    raw = np.fromfile(work / "out.bin", f4)
    n3 = nx * (nz + 1) * ny
    out = {}
    for i, name in enumerate(("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN")):
        out[name] = raw[i * n3:(i + 1) * n3].reshape(ny, nz + 1, nx).transpose(1, 0, 2)[:nz]
    off = 6 * n3
    out["RAINCV"] = raw[off:off + nx * ny].reshape(ny, nx)
    out["PRATEC"] = raw[off + nx * ny:off + 2 * nx * ny].reshape(ny, nx)
    return out


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    s = np.sqrt(np.mean(b * b))
    return float(np.sqrt(np.mean((a - b) ** 2)) / s) if s > 0 else None


qvften = aux["RQVFTEN"][0]
assert np.array_equal(qvften, r["RQVFTEN"])
ref = run_wrf_tiedtke(qvften, aux["RQVBLTEN"][1], aux["QFX"][1], "wrfinputs")
res = {"phyprep_checks": checks,
       "pristine_fortran_vs_wrf_aux": {n: rel(ref[n], aux[n][1]) for n in
                                       ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN", "RAINCV")},
       "raincv_n_pos": {"wrf_aux": int((aux["RAINCV"][1] > 0).sum()), "pristine_fortran": int((ref["RAINCV"] > 0).sum())},
       "raincv_sum": {"wrf_aux": float(aux["RAINCV"][1].sum()), "pristine_fortran": float(ref["RAINCV"].sum())}}
np.savez(OUT / "phyprep.npz", t_phy=t_phy, pi_phy=pi_phy, rho=rho, u_phy=u_phy, v_phy=v_phy, w=r["W_2"],
         dz8w=dz8w, p_hyd=p_hyd, p_hyd_w=p_hyd_w, znu=r["ZNU"], qfx=aux["QFX"][1], xland=r["XLAND"],
         qvften=qvften, qvpblten=aux["RQVBLTEN"][1], qv=qv, qc=r["QCLOUD"], qi=r["QICE"],
         **{"ref_" + k: v for k, v in ref.items()}, **{"aux_" + k: aux[k][1] for k in ("RTHCUTEN", "RQVCUTEN", "RAINCV")})
(OUT / "pristine.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
