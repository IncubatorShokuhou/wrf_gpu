"""AE07 inputs: the RE01 operands + QNWFA/QNIFA columns + aer_opt=3 arms.

usage: python ae07_build_inputs.py OUTDIR      (writes ae07_input.bin, ae07_arms.json, ae07_aerosol.npz)

QNWFA/QNIFA (number per kg): WRF thompson_init's default decaying profiles (module_mp_thompson.F:493-549, no
input aerosol: naCCN0/1 = 300e6/50e6, naIN0/1 = 1.5e6/0.5e6 per kg, h_01 = 0.8 km below 1000 m terrain) on each
column's dz8w heights, times a seeded per-column log-uniform factor in [0.2, 5] so the RH/T lookups and the
1e-3 per-layer floor of gt_aod are all exercised.  Arm scales multiply both species.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ae01_build_inputs import FIXTURE, f  # noqa: E402

# (julian 0-based REAL; < 0 = the column's own radiation-call julian, aer_opt, QNWFA/QNIFA scale)
ARMS = [(-1.0, 0, 1.0), (-1.0, 3, 1.0), (-1.0, 3, 10.0), (135.0, 3, 1.0), (196.6, 3, 0.1), (196.6, 0, 1.0)]


def profiles(dz8w):
    z = np.cumsum(np.asarray(dz8w, np.float64)[:, :-1], axis=1) - 0.5 * np.asarray(dz8w, np.float64)[:, :1]
    rel_km = (z - z[:, :1]) / 1000.0
    ccn3 = -np.log(50.0e6 / 300.0e6) / 0.8
    in3 = -np.log(0.5e6 / 1.5e6) / 0.8
    rng = np.random.default_rng(2207)
    fac = np.exp(rng.uniform(np.log(0.2), np.log(5.0), (z.shape[0], 1)))
    nwfa = (50.0e6 + 300.0e6 * np.exp(-rel_km * ccn3)) * fac
    nifa = (0.5e6 + 1.5e6 * np.exp(-rel_km * in3)) * fac
    return nwfa.astype(np.float32), nifa.astype(np.float32)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    src = out / "ae01_input.bin"
    if not src.is_file():
        raise SystemExit(f"{src} missing: run ae01_build_inputs.py {out} first")
    d = np.load(FIXTURE)
    nwfa, nifa = profiles(d["in_dz8w"])
    raw = src.read_bytes()
    ncol, nz = np.frombuffer(raw, "<i4", 2)
    narm_old = 11
    # AE01 layout ends with narm (i4), julian(f4 x narm), aer_opt (i4 x narm), scale (f4 x narm)
    head = raw[: len(raw) - 4 - 12 * narm_old]
    assert np.frombuffer(raw, "<i4", 1, len(head))[0] == narm_old
    with (out / "ae07_input.bin").open("wb") as h:
        h.write(head)
        h.write(f(nwfa)); h.write(f(nifa))
        h.write(np.asarray([len(ARMS)], "<i4").tobytes())
        h.write(np.asarray([a[0] for a in ARMS], "<f4").tobytes())
        h.write(np.asarray([a[1] for a in ARMS], "<i4").tobytes())
        h.write(np.asarray([a[2] for a in ARMS], "<f4").tobytes())
    np.savez(out / "ae07_aerosol.npz", qnwfa=nwfa, qnifa=nifa)
    (out / "ae07_arms.json").write_text(json.dumps({"arms": ARMS, "ncol": int(ncol), "nz": int(nz)}, indent=1))
    print("ae07 inputs", ncol, nz, len(ARMS))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
