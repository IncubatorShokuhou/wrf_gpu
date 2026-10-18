"""AE09: 1-ulp ensemble of the pristine WRF aer_opt=3 oracle (AE07) -> per-cell WRF reproducibility envelope.

usage: python ae09_ensemble.py AE01_DIR OUTDIR N      (AE01_DIR holds the built ae07_driver.exe and its run/ dir)

Every RE01 column is replicated N times; member 0 is the unperturbed AE07 operand (asserted bitwise equal to the
registered AE07 outputs), members 1..N-1 move ~half of the column's T and QNWFA values by one REAL ulp (seeded).
QNIFA, pressure (McICA seed digits) and clouds are untouched.  The same unchanged WRF V4.7.1 chain (gt_aod ->
calc_aerosol_rrtmg_sw -> RRTMG_SWRAD) runs on all members and arms.  Output: per (column, arm, level) min/max of
the gated SW outputs over the ensemble and the WRF self-spread ratio max|member - member0| / frozen bound;
the fixture keeps only the cells whose self-spread ratio exceeds 1 (index, ensemble min/max).
A cell whose WRF self-spread ratio exceeds 1 is one where WRF's own REAL result is not reproducible at the frozen
tier-1 bound (two-stream conservative/resonance branch in thin top layers, AE09 notes); the port is gated there
against the WRF envelope widened by the bound instead of member 0 alone.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ae01_build_inputs as A1  # noqa: E402
import ae07_build_inputs as A7  # noqa: E402
from ae01_parse import SCALARS as AE01_SCALARS  # noqa: E402

ROOT = HERE.parents[1]
AE07_FIXTURE = ROOT / "tests/v025/w41_aerosol/fixtures/thompson_aer_sw_0227_v1.npz"
FLUX_BOUND, HEATING_BOUND = (1.0, 0.05), (1.0e-4, 0.05)   # frozen tier-1 (tests/v025/w41_aerosol/tegen_fixture.py)
SCALARS = ("swdnb", "swupt", "swddir", "swdnbc", "swuptc")
CHUNK = 8
PROFILES = ("swdn", "swup", "hsw")


def write_input(path: Path, n: int, cols, seed: int = 909):
    d = dict(np.load(A1.FIXTURE))
    ncol, nz = d["in_t"].shape
    nwfa, nifa = A7.profiles(d["in_dz8w"])
    idx = np.repeat(np.asarray(cols), n)
    sub = {k: (v[idx] if v.ndim >= 1 and v.shape[0] == ncol else v) for k, v in d.items()}
    t, nw, ni = np.asarray(sub["in_t"], np.float32).copy(), nwfa[idx].copy(), nifa[idx].copy()
    rng = np.random.default_rng([seed, int(cols[0])])
    pert = (np.arange(idx.size) % n) != 0
    for a in (t, nw):
        m = (rng.random(a.shape) < 0.5) & pert[:, None]
        up = rng.random(a.shape) < 0.5
        a[:] = np.where(m, np.where(up, np.nextafter(a, np.float32(np.inf)), np.nextafter(a, np.float32(-np.inf))), a)
    sub["in_t"] = t
    f = A1.f
    with path.open("wb") as h:  # ae07_input.bin layout (byte-identical to AE07's for n=1, no perturbation)
        h.write(np.asarray([idx.size, nz], "<i4").tobytes())
        h.write(np.asarray([sub["in_p_top"], sub["in_gmt"]], "<f4").tobytes())
        for key in ("in_julian", "in_julday", "in_tsk", "in_emiss", "in_albedo", "in_coszen_hist", "in_xland", "in_xlat",
                    "in_xlong", "in_snow", "in_xice", "in_lat_d01"):
            h.write(np.asarray(sub[key], "<i4" if key == "in_julday" else "<f4").tobytes())
        for key in ("in_t", "in_p", "in_pi", "in_th", "in_rho", "in_qv", "in_qc", "in_qi", "in_ni", "in_qs", "in_qr", "in_qg",
                    "in_qc_rad", "in_qi_rad", "in_cldfra", "in_p_hyd", "in_p_d01", "in_dz8w", "in_p_hyd_w", "in_t8w"):
            h.write(f(np.asarray(sub[key], "<f4")))
        h.write(np.asarray(sub["in_tau"], "<f4").tobytes())
        h.write(f(nw)); h.write(f(ni))
        h.write(np.asarray([len(A7.ARMS)], "<i4").tobytes())
        h.write(np.asarray([a[0] for a in A7.ARMS], "<f4").tobytes())
        h.write(np.asarray([a[1] for a in A7.ARMS], "<i4").tobytes())
        h.write(np.asarray([a[2] for a in A7.ARMS], "<f4").tobytes())
    return ncol, nz


def read_output(path: Path, nz: int):
    """ae07_output.bin as one zero-copy structured array (record layout of ae07_parse.parse, which keeps Python lists
    per record and does not fit a 6 GB slot at 35k records)."""

    buf = path.read_bytes()
    ncol, _, narm = np.frombuffer(buf, "<i4", 3)
    rec = np.dtype([("head", "<f4", 4), ("taod5503d", "<f4", nz), ("taod5502d", "<f4"), ("tauaer", "<f4", nz * 14),
                    ("ssaaer", "<f4", nz * 14), ("asyaer", "<f4", nz * 14), ("hsw", "<f4", nz), ("hswc", "<f4", nz),
                    ("swup", "<f4", nz + 2), ("swupc", "<f4", nz + 2), ("swdn", "<f4", nz + 2), ("swdnc", "<f4", nz + 2),
                    ("scalars", "<f4", len(AE01_SCALARS))])
    assert len(buf) == 12 + rec.itemsize * ncol * narm, (len(buf), rec.itemsize, ncol, narm)
    r = np.frombuffer(buf, rec, ncol * narm, 12).reshape(ncol, narm)
    out = {k: r[k] for k in ("hsw", "swup", "swdn")}
    for i, k in enumerate(AE01_SCALARS):
        out[k] = r["scalars"][..., i]
    return out


def main(ae01: Path, out: Path, n: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ncol, nz = np.load(A1.FIXTURE)["in_t"].shape
    # The driver's memory grows with the number of RRTMG_SWRAD calls (> 6 GB at 35k calls, 0.42 GB at 128), so it
    # runs on chunks of CHUNK columns x n members x 6 arms; chunks are independent (seed per first column).
    parts = []
    for c0 in range(0, ncol, CHUNK):
        tag = f"{c0:03d}"
        cols = np.arange(c0, min(c0 + CHUNK, ncol))
        write_input(out / f"ae07_input_{tag}.bin", n, cols)
        if "--reuse" not in sys.argv or not (out / f"ae07_output_{tag}.bin").is_file():
            subprocess.run(["taskset", "-c", "8", "nice", "-n", "19", str(ae01 / "ae07_driver.exe"),
                            str(out / f"ae07_input_{tag}.bin"), str(out / f"ae07_output_{tag}.bin")], cwd=ae01 / "run",
                           check=True, capture_output=True)
        parts.append(read_output(out / f"ae07_output_{tag}.bin", nz))
    w = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    ref = np.load(AE07_FIXTURE)
    shape = lambda a: a.reshape((ncol, n) + a.shape[1:])  # noqa: E731  (col*n, arm, ...) -> (col, n, arm, ...)
    env, checks = {}, {}
    for k in SCALARS + PROFILES:
        a = shape(np.asarray(w[k]))
        if k in ("swdn", "swup"):
            a = a[..., : nz + 1]
        m0 = a[:, 0]
        checks[f"member0_{k}_bitwise_AE07"] = bool(np.array_equal(
            m0.view(np.uint32), ref[k][..., : m0.shape[-1]].view(np.uint32)))
        bound = HEATING_BOUND if k == "hsw" else FLUX_BOUND
        spread = np.abs(a.astype(np.float64) - m0[:, None].astype(np.float64)).max(1) / (bound[0] + bound[1] * np.abs(m0))
        cells = np.argwhere(spread > 1)   # (column, arm[, level]) where WRF's own 1-ulp spread leaves the bound
        env[f"{k}_cells"] = cells.astype(np.int32)
        env[f"{k}_lo"], env[f"{k}_hi"] = a.min(1)[tuple(cells.T)], a.max(1)[tuple(cells.T)]
        env[f"{k}_self_max"] = np.float32(spread.max())
        checks[f"cells_wrf_self_spread_over_bound_{k}"] = int((spread > 1).sum())
        checks[f"max_wrf_self_spread_ratio_{k}"] = float(spread.max())
    assert all(v for k, v in checks.items() if k.startswith("member0_")), checks
    path = out / "thompson_aer_sw_0227_ens_v1.npz"
    np.savez_compressed(path, members=np.int32(n), arm_julian=ref["arm_julian"], arm_aer_opt=ref["arm_aer_opt"],
                        arm_scale=ref["arm_scale"], **env)
    summary = {"members": n, "perturbed": "T and QNWFA, ~50 % of values per member by one REAL ulp (seed 909)",
               "checks": checks, "fixture": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
               "raw": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("ae07_*.bin"))}}
    (HERE / "ae09_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
