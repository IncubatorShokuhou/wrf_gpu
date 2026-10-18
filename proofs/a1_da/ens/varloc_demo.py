"""Variable-localisation demonstration of the R3 (4) 0228 mechanism (lane w41-da-ens), product analyse path.

Truth and ensemble: temperature and wind perturbations are drawn INDEPENDENTLY (true T–U cross covariance 0), smooth
random fields on a 40x44 grid, N members (10 / 20). Co-located surface stations observe T2 (= lowest-level T) and
U10 (= lowest-level U) with sigma_o 1.0 K / 1.5 m/s. Analyses: (a) T2 only, (b) T2 + U10 without variable
localisation, (c) T2 + U10 with ``varloc.as_var_loc('surface_split')``. Metric: interior lowest-level T analysis
RMSE vs truth, mean over trials. Expected: (b) worse than (a) through sampling cross covariance (R3: 0228 T2 OmA
0.520 all obs vs 0.442 T2-only at N = 10); (c) equals (a) for T by construction and keeps the U10 benefit for U.
Run: JAX_PLATFORMS=cpu python varloc_demo.py out.json
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from gpuwrf.assimilation.letkf import analysis as an  # noqa: E402
from gpuwrf.assimilation.letkf import varloc  # noqa: E402

NZ, NY, NX = 3, 40, 44


def smooth(a, k=4):
    for _ in range(k):
        a = (a + np.roll(a, 1, -1) + np.roll(a, -1, -1) + np.roll(a, 1, -2) + np.roll(a, -1, -2)) / 5.0
    return a / a.std()


def fields(rng, n):
    return {"T": np.stack([smooth(rng.normal(size=(NZ, NY, NX))) for _ in range(n)]),
            "U": 1.5 * np.stack([smooth(rng.normal(size=(NZ, NY, NX + 1))) for _ in range(n)])}


def trial(seed, n):
    rng = np.random.default_rng(seed)
    lat, lon = np.meshgrid(27.0 + 0.08 * np.arange(NY), -18.0 + 0.09 * np.arange(NX), indexing="ij")
    geo = {"XLAT": lat, "XLONG": lon, "HGT": np.zeros((NY, NX))}
    truth = {k: v[0] for k, v in fields(rng, 1).items()}
    ens = fields(rng, n)
    pb = (1.0e5 * np.exp(-0.12 * np.arange(NZ)))[:, None, None] * np.ones((NZ, NY, NX))
    frames = {"T": 280.0 + ens["T"], "U": ens["U"], "V": np.zeros((n, NZ, NY + 1, NX)),
              "QVAPOR": np.full((n, NZ, NY, NX), 0.008), "P": np.zeros((n, NZ, NY, NX)),
              "PB": np.broadcast_to(pb, (n, NZ, NY, NX)).copy(), "PSFC": np.full((n, NY, NX), 1.0e5)}
    st = [(j, i) for j in range(10, NY - 10, 5) for i in range(10, NX - 10, 5)]
    jj = np.array([s[0] for s in st] * 2)
    ii = np.array([s[1] for s in st] * 2)
    var = np.array([11] * len(st) + [13] * len(st))
    sig = np.where(var == 11, 1.0, 1.5)
    tval = 280.0 + truth["T"][0, jj, ii]
    uval = truth["U"][0, jj, ii]
    value = np.where(var == 11, tval, uval) + sig * rng.standard_normal(len(var))
    obs = types.SimpleNamespace(kind=np.full(len(var), 2), var=var, value=value, error=sig, qc=np.zeros(len(var), int),
                                lat=lat[jj, ii], lon=lon[jj, ii], p_pa=np.full(len(var), np.nan), j=jj, i=ii)

    def hx(o, f, g):
        t = np.asarray(f["T"])[:, 0, o.j, o.i]
        u = np.asarray(f["U"])[:, 0, o.j, o.i]
        return np.where(np.asarray(o.var)[None] == 11, t, u)

    params = an.LetkfParams(stride_h=1, stride_v=1, rtps_alpha=0.0, rebalance=False, variables=("T", "U"),
                            diagnostic_variables=(), loc_h_km=60.0, pmax=200)
    t_only = np.asarray(var) == 11
    sub = types.SimpleNamespace(**{k: (v[t_only] if isinstance(v, np.ndarray) and v.shape == var.shape else v)
                                   for k, v in vars(obs).items()})
    a_t, _ = an.analyse(frames, geo, sub, params, model_equivalents=hx)
    a_all, _ = an.analyse(frames, geo, obs, params, model_equivalents=hx)
    a_vl, _ = an.analyse(frames, geo, obs, params, model_equivalents=hx, var_loc=varloc.as_var_loc("surface_split",
                                                                                                    ("T", "U")))
    inner = (slice(8, NY - 8), slice(8, NX - 8))

    def rmse(a, name):
        tr = truth[name][0] + (280.0 if name == "T" else 0.0)
        return float(np.sqrt(np.mean((a[name].mean(0)[0][inner] - tr[inner]) ** 2)))

    return {"T_b": rmse(frames, "T"), "T_t2only": rmse(a_t, "T"), "T_all": rmse(a_all, "T"), "T_varloc": rmse(a_vl, "T"),
            "U_b": rmse(frames, "U"), "U_all": rmse(a_all, "U"), "U_varloc": rmse(a_vl, "U")}


if __name__ == "__main__":
    out = {}
    for n in (10, 20):
        rows = [trial(1000 + s, n) for s in range(40)]
        out[f"N{n}"] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[f"N{n}"]["T_all_worse_than_t2only_fraction"] = float(np.mean([r["T_all"] > r["T_t2only"] for r in rows]))
        out[f"N{n}"]["T_varloc_equals_t2only_max_abs"] = float(max(abs(r["T_varloc"] - r["T_t2only"]) for r in rows))
        print(n, json.dumps(out[f"N{n}"]), flush=True)
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
