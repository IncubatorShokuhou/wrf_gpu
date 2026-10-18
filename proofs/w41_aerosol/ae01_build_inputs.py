"""AE01 inputs: the RE01 operands (tests/v025/fid_q2 fixture) + aerosol arms, and the synthetic unit-oracle set.

usage: python ae01_build_inputs.py OUTDIR      (writes ae01_input.bin, ae01_units_input.bin, ae01_arms.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz"
# (julian 0-based REAL; < 0 = the column's own radiation-call julian, aer_opt, aerod scale)
ARMS = [(-1.0, 0, 1.0), (-1.0, 1, 1.0), (0.3, 0, 1.0), (0.3, 1, 1.0), (135.0, 0, 1.0), (135.0, 1, 1.0),
        (196.6, 0, 1.0), (196.6, 1, 1.0), (352.2, 0, 1.0), (352.2, 1, 1.0), (-1.0, 1, 10.0)]
F4 = np.float32


def f(a):
    """Fortran column-major bytes of a C-order array."""
    return np.asfortranarray(np.asarray(a)).tobytes(order="F")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(FIXTURE)
    ncol, nz = d["in_t"].shape
    with (out / "ae01_input.bin").open("wb") as h:
        h.write(np.asarray([ncol, nz], "<i4").tobytes())
        h.write(np.asarray([d["in_p_top"], d["in_gmt"]], "<f4").tobytes())
        for key in ("in_julian", "in_julday", "in_tsk", "in_emiss", "in_albedo", "in_coszen_hist", "in_xland", "in_xlat",
                    "in_xlong", "in_snow", "in_xice", "in_lat_d01"):
            h.write(np.asarray(d[key], "<i4" if key == "in_julday" else "<f4").tobytes())
        for key in ("in_t", "in_p", "in_pi", "in_th", "in_rho", "in_qv", "in_qc", "in_qi", "in_ni", "in_qs", "in_qr", "in_qg",
                    "in_qc_rad", "in_qi_rad", "in_cldfra", "in_p_hyd", "in_p_d01"):
            h.write(f(np.asarray(d[key], "<f4")))
        for key in ("in_dz8w", "in_p_hyd_w", "in_t8w"):
            h.write(f(np.asarray(d[key], "<f4")))
        h.write(np.asarray(d["in_tau"], "<f4").tobytes())
        h.write(np.asarray([len(ARMS)], "<i4").tobytes())
        h.write(np.asarray([a[0] for a in ARMS], "<f4").tobytes())
        h.write(np.asarray([a[1] for a in ARMS], "<i4").tobytes())
        h.write(np.asarray([a[2] for a in ARMS], "<f4").tobytes())

    # Unit set: latitudes/longitudes on table points, cell interiors and the periodic edges.
    lat = np.unique(np.concatenate([np.linspace(-89.9, 89.9, 37), [-90.0, -89.5, -89.0, -86.0, -1e-3, 0.0, 2.0, 27.03, 28.4,
                                                                   29.85, 86.0, 88.99, 89.0, 89.5, 90.0]])).astype(F4)
    lon = np.unique(np.concatenate([np.linspace(-180.0, 179.9, 41), [-180.0, -179.99, -19.54, -15.0, -11.9, -2.5, 0.0, 2.5,
                                                                     174.99, 175.0, 177.5, 179.99, 180.0]])).astype(F4)
    jul = np.unique(np.concatenate([np.arange(0.0, 366.0, 0.731), [0.0, 0.3, 14.0, 15.0, 15.5, 16.0, 43.99, 44.0, 104.0,
                                                                   135.0, 196.6, 288.0, 348.5, 349.0, 349.5, 352.2, 363.99,
                                                                   364.0, 364.5, 364.999]])).astype(F4)
    # pressure columns: real p_hyd/p_hyd_w + synthetic ones reaching above 20 hPa and hitting table levels exactly
    rng = np.random.default_rng(7)
    real_p, real_pf = d["in_p_hyd"][::6], d["in_p_hyd_w"][::6]
    synth_pf = []
    for top, sfc in ((500.0, 101300.0), (1000.0, 99000.0), (2000.0, 96500.0), (5000.0, 70000.0), (300.0, 104000.0)):
        eta = np.linspace(1.0, 0.0, nz + 1) ** 1.6
        synth_pf.append(top + (sfc - top) * eta)
    synth_pf = np.asarray(synth_pf, F4)
    synth_p = (0.5 * (synth_pf[:, :-1] + synth_pf[:, 1:])).astype(F4)
    # one monotone column whose mass levels hit all 12 table pressures exactly (pmid == pin(k) branches)
    pina_pa = np.asarray([20., 45., 80., 125., 180., 247.5, 337.5, 470., 635., 787., 884., 959.], F4) * F4(100.0)
    hits = np.sort(np.concatenate([pina_pa, np.linspace(600.0, 104000.0, nz - 12).astype(F4)]))[::-1].astype(F4)
    edges = np.concatenate([[hits[0] + 500.0], 0.5 * (hits[:-1] + hits[1:]), [max(hits[-1] - 100.0, 1.0)]]).astype(F4)
    synth_p = np.concatenate([synth_p, hits[None]]).astype(F4)
    synth_pf = np.concatenate([synth_pf, edges[None]]).astype(F4)
    p = np.concatenate([real_p, synth_p]).astype(F4)
    pf = np.concatenate([real_pf, synth_pf]).astype(F4)
    # every column needs p(k) between pf(k) and pf(k+1) only for physical meaning; aer_p_int uses them pointwise
    aerodt = rng.uniform(0.0, 6.0e-5, (p.shape[0], 12, 6)).astype(F4)
    aerodt[0] = 0.0
    with (out / "ae01_units_input.bin").open("wb") as h:
        h.write(np.asarray([lat.size, lon.size], "<i4").tobytes())
        h.write(lat.tobytes()); h.write(lon.tobytes())
        h.write(np.asarray([jul.size], "<i4").tobytes()); h.write(jul.tobytes())
        h.write(np.asarray(p.shape, "<i4").tobytes())
        h.write(f(p)); h.write(f(pf)); h.write(f(aerodt))
    np.savez(out / "ae01_units_operands.npz", lat=lat, lon=lon, julian=jul, p=p, pf=pf, aerodt=aerodt)
    (out / "ae01_arms.json").write_text(json.dumps({"arms": ARMS, "fixture": str(FIXTURE)}, indent=1) + "\n")
    print("ncol", ncol, "nz", nz, "arms", len(ARMS), "units lat/lon/jul/pcols", lat.size, lon.size, jul.size, p.shape[0])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
