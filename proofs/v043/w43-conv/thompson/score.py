"""CT03: score CT02 with the FROZEN increment gate of tests/v025/b_thompson/test_mixed_phase_wrf_oracle.py
(excess = (max|d_port-d_WRF| - 2ulp(max|WRF|) - R1 - 1e-4*fixture max|d_WRF|)/column max|d_WRF| <= 1e-2, Ni 5e-2; precip
alike), plus signed aggregate diagnostics (sum of surface precip, column-integrated qr/qg/theta increments) per kind.
E200 / frozen _validate_fixture policy (critic cr-w43-conv RC03): EVERY input (in_*, dt) and reference (wrf_*) value must be
finite BEFORE any field-wide scale is formed, else the whole fixture is INVALID (verdict FAIL, rc 2); non-finite candidate
values (port_*, incl. every precipitation component) make that column/field excess = inf (FAIL).
usage: score.py <columns.npz> <oracle.npz> <out.json>   |   score.py --controls <columns.npz> <oracle.npz> <out.json>"""
import json, sys
import numpy as np
FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th")
TOL = {**{v: 1e-2 for v in FIELDS}, "ni": 5e-2, "precip": 1e-2}
R1 = {v: 1e-12 for v in ("qc", "qr", "qi", "qs", "qg")}
f64 = lambda a: np.asarray(a, np.float64)  # noqa: E731
PPT = ("port_rainncv", "port_snowncv", "port_graupelncv", "port_iceppt")


def validate(z, o):
    """frozen _validate_fixture: inputs + references finite before any reduction."""
    bad = [k for k in sorted(z) if (k.startswith("in_") or k == "dt") and not np.isfinite(f64(z[k])).all()]
    bad += [k for k in sorted(o) if k.startswith("wrf_") and not np.isfinite(f64(o[k])).all()]
    return bad


def excesses(z, o):
    names = z["names"]
    scales = {v: float(np.abs(f64(o[f"wrf_{v}"]) - z[f"in_{v}"]).max()) for v in FIELDS}
    scales["precip"] = float(f64(o["wrf_rainncv"]).max())
    port_pr = sum(f64(o[k]) for k in PPT)
    ppt_ok = np.all([np.isfinite(f64(o[k])) for k in PPT], axis=0)
    ex = {}
    for i, n in enumerate(names):
        for v in FIELDS:
            x, w, p = f64(z[f"in_{v}"][i]), f64(o[f"wrf_{v}"][i]), f64(o[f"port_{v}"][i])
            if not (np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(p).all()):
                ex[(n, v)] = np.inf; continue
            dw, dp = w - x, p - x
            floor = 2.0 * float(np.spacing(np.float32(np.abs(w).max()))) + R1.get(v, 0.0) + 1e-4 * scales[v]
            ex[(n, v)] = max(0.0, np.abs(dp - dw).max() - floor) / max(np.abs(dw).max(), 1e-30)
        w, p = float(o["wrf_rainncv"][i]), float(port_pr[i])
        if not (np.isfinite(w) and np.isfinite(p) and bool(ppt_ok[i])):
            ex[(n, "precip")] = np.inf; continue
        floor = 2.0 * float(np.spacing(np.float32(w))) + 1e-4 * scales["precip"]
        ex[(n, "precip")] = max(0.0, abs(p - w) - floor) / max(w, 1e-30)
    return ex, port_pr


def verdict(z, o):
    bad = validate(z, o)
    if bad:
        return {"valid": False, "nonfinite_inputs_or_references": bad, "n_fail": None, "passed": False}, None, None
    ex, port_pr = excesses(z, o)
    fails = sorted(((n, v, e) for (n, v), e in ex.items() if e > TOL[v]), key=lambda t: -t[2])
    return {"valid": True, "n_fail": len(fails), "passed": len(fails) == 0}, (ex, fails), port_pr


def controls(z, o):
    """single NaN/+Inf/-Inf in reference precip, input qr and candidate qr/precip must FAIL (E200)."""
    out = {}
    for kind, key, idx in (("ref", "wrf_rainncv", (0,)), ("in", "in_qr", (0, 0)), ("cand", "port_qr", (0, 0)),
                           ("cand", "port_rainncv", (0,))):
        for val in (np.nan, np.inf, -np.inf):
            zz, oo = dict(z), dict(o)
            tgt = zz if key.startswith("in_") else oo
            a = np.array(tgt[key], np.float32, copy=True); a[idx] = val; tgt[key] = a
            v, *_ = verdict(zz, oo)
            out[f"{key}={val}"] = {"passed": v["passed"], "valid": v["valid"], "n_fail": v["n_fail"]}
    out["all_controls_fail"] = all(not r["passed"] for r in out.values() if isinstance(r, dict))
    return out


if sys.argv[1] == "--controls":
    z, o = dict(np.load(sys.argv[2])), dict(np.load(sys.argv[3]))
    res = controls(z, o); json.dump(res, open(sys.argv[4], "w"), indent=1); print(json.dumps(res)); sys.exit(0 if res["all_controls_fail"] else 1)
z, o = dict(np.load(sys.argv[1])), dict(np.load(sys.argv[2]))
names, kinds = z["names"], z["kind"]
head, detail, port_pr = verdict(z, o)
if not head["valid"]:
    json.dump(head, open(sys.argv[3], "w"), indent=1); print("INVALID", head); sys.exit(2)
ex, fails = detail
byfield = {}
for n, v, e in fails:
    byfield.setdefault(v, []).append(n)
# signed aggregates (rho*dz weighting not needed for a same-input sign check: use dz-weighted mixing-ratio increments)
dz = f64(z["in_dz8w"])
agg = {}
for kind in sorted(set(kinds.tolist())) + ["ALL"]:
    m = np.ones(len(names), bool) if kind == "ALL" else kinds == kind
    a = {"n": int(m.sum()), "precip_wrf_sum": float(f64(o["wrf_rainncv"])[m].sum()), "precip_port_sum": float(port_pr[m].sum())}
    for v in ("qr", "qg", "qs", "qc", "qv", "th", "nr"):
        dwi = ((f64(o[f"wrf_{v}"]) - z[f"in_{v}"]) * dz)[m].sum()
        dpi = ((f64(o[f"port_{v}"]) - z[f"in_{v}"]) * dz)[m].sum()
        a[f"d{v}_dz_wrf"], a[f"d{v}_dz_port"] = float(dwi), float(dpi)
    agg[kind] = a
census = {"n": len(names), "max_qr": float(z["in_qr"].max()), "max_qg": float(z["in_qg"].max()), "max_w": float(z["in_w"].max()),
          "min_w": float(z["in_w"].min()), "cols_qr_ge_3gkg": int((z["in_qr"].max(1) >= 3e-3).sum()),
          "cols_qg_ge_2gkg": int((z["in_qg"].max(1) >= 2e-3).sum()),
          "cols_wrf_precip_ge_0p1mm": int((f64(o["wrf_rainncv"]) >= 0.1).sum()),
          "max_wrf_precip_step_mm": float(f64(o["wrf_rainncv"]).max()),
          "cols_with_dqg_wrf": int((np.abs(f64(o["wrf_qg"]) - z["in_qg"]) > 0).any(1).sum())}
worst = {v: max((e for (n, vv), e in ex.items() if vv == v), default=0.0) for v in TOL}
res = {**head, "census": census, "fails_by_field": {k: len(v) for k, v in byfield.items()},
       "worst_excess": worst, "top_fails": [(n, v, float(e)) for n, v, e in fails[:40]], "aggregates": agg,
       "env": [str(s) for s in o.get("env_applied", [])]}
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: res[k] for k in ("valid", "passed", "census", "n_fail", "fails_by_field", "worst_excess")}, indent=0))
print(json.dumps(res["top_fails"][:15]))
for k, a in agg.items():
    print(k, a["n"], "precip wrf/port %.4g/%.4g" % (a["precip_wrf_sum"], a["precip_port_sum"]),
          "dqr %.4g/%.4g dqg %.4g/%.4g dth %.4g/%.4g" % (a["dqr_dz_wrf"], a["dqr_dz_port"], a["dqg_dz_wrf"], a["dqg_dz_port"], a["dth_dz_wrf"], a["dth_dz_port"]))
