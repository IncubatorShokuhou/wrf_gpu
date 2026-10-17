"""Evaluate the CPU coupled smoke (probe.py with O1_EXEC_STEPS>0): control vs grid_fdda=1 arm.

usage: python smoke_check.py <control probe.json> <control arm> <arm probe.json> <fdda arm> <wrffdda_d01>
                              <wrfinput_d01> <coef at the last step> <out.json>

Nudging pulls each field toward the time-interpolated analysis; in the first interval WRF's OLD analysis is
the model state at step 1 (fddagd_driver), so the target is wrfinput*(1-coef) + NEW(record 0)*coef and the
FDDA-minus-control difference after a few steps must be positively aligned with (target - control state) in
u, v, dry theta and qv (each lane separately: ru/rv_tendf, t_tendf, moist_tend(qv)).  Reports the Pearson
correlation and the regression slope (expected ~ sum over steps of dt*g*weights).
"""

from __future__ import annotations

import json
import sys

import netCDF4
import numpy as np

RVRD = np.float32(np.float32(461.6) / np.float32(287.0))


def main() -> None:
    probe_c, control, probe_a, arm, wrffdda, wrfinput, coef, out = sys.argv[1:9]
    coef = float(coef)
    rec = {**json.load(open(probe_c))["arms"], **{arm: json.load(open(probe_a))["arms"][arm]}}
    rec[control] = json.load(open(probe_c))["arms"][control]
    c, f = np.load(rec[control]["final_npz"]), np.load(rec[arm]["final_npz"])
    with netCDF4.Dataset(wrffdda) as ds:
        new = {k: np.asarray(ds[f"{k}_NDG_NEW"][0], np.float64) for k in ("U", "V", "T", "Q")}
    with netCDF4.Dataset(wrfinput) as ds:
        qv0 = np.asarray(ds["QVAPOR"][0], np.float64)
        s0 = {"U": np.asarray(ds["U"][0], np.float64), "V": np.asarray(ds["V"][0], np.float64), "Q": qv0,
              "T": (np.asarray(ds["THM"][0], np.float64) + 300.0) / (1.0 + float(RVRD) * qv0) - 300.0}
    new = {k: s0[k] * (1.0 - coef) + new[k] * coef for k in new}  # WRF first-interval target
    dry = lambda s: np.asarray(s["theta"], np.float64) / (1.0 + float(RVRD) * np.asarray(s["qv"], np.float64))  # noqa
    pairs = {
        "u": (np.asarray(f["u"], np.float64) - c["u"], new["U"] - c["u"]),
        "v": (np.asarray(f["v"], np.float64) - c["v"], new["V"] - c["v"]),
        "theta_dry": (dry(f) - dry(c), new["T"] + 300.0 - dry(c)),
        "qv": (np.asarray(f["qv"], np.float64) - c["qv"], new["Q"] - c["qv"]),
    }
    result = {"control": control, "arm": arm, "fields": {}}
    for name, (delta, push) in pairs.items():
        inner = (slice(None), slice(2, -2), slice(2, -2))  # away from the specified/relaxation rows
        d, p = delta[inner].ravel(), push[inner].ravel()
        corr = float(np.corrcoef(d, p)[0, 1]) if d.std() > 0 and p.std() > 0 else 0.0
        result["fields"][name] = dict(max_abs_delta=float(np.abs(delta).max()), corr_with_push=corr,
                                      slope=float((d @ p) / (p @ p)) if p @ p > 0 else 0.0)
    for key in ("nonfinite_values", "guard_total", "exec_s", "fdda_held_absmax"):
        result[f"fdda_{key}"] = rec[arm].get(key)
        result[f"control_{key}"] = rec[control].get(key)
    result["pass"] = bool(all(v["corr_with_push"] > 0.5 for v in result["fields"].values())
                          and rec[arm].get("nonfinite_values") == 0)
    json.dump(result, open(out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
