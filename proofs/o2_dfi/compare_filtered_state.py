"""Compare a port DFI-filtered wrfinput with the CPU-WRF wrfinput_initialized_d01 (complete state).

For every field: RMSE/max |port - WRF|, the WRF DFI increment scale RMSE(WRF - input), the port
increment RMSE(port - input), the increment correlation and the residual fraction
RMSE(port - WRF) / RMSE(WRF - input) (0 = identical filtered state, 1 = no better than the
unfiltered input).  Pure NumPy/netCDF4 (no JAX).

usage: python compare_filtered_state.py <port_wrfinput> <wrf_initialized> <raw_wrfinput> <out.json>
"""
import json
import sys

import netCDF4
import numpy as np

PROGNOSTIC = ("U", "V", "W", "PH", "MU", "THM", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
              "QNICE", "QNRAIN", "P", "AL", "P_HYD")
LAND = ("TSK", "TSLB", "SMOIS", "SNOW", "SNOWH", "SNOWC", "LAI", "CANWAT")
DIAG = ("T2", "Q2", "U10", "V10", "PSFC", "QKE", "T")


def stats(port, wrf, raw):
    port, wrf, raw = (np.asarray(x, np.float64) for x in (port, wrf, raw))
    out = {
        "rmse_port_wrf": float(np.sqrt(np.mean((port - wrf) ** 2))),
        "max_port_wrf": float(np.max(np.abs(port - wrf))),
        "rmse_wrf_increment": float(np.sqrt(np.mean((wrf - raw) ** 2))),
        "rmse_port_increment": float(np.sqrt(np.mean((port - raw) ** 2))),
        "nonfinite_port": int((~np.isfinite(port)).sum()),
    }
    a, b = (wrf - raw).ravel(), (port - raw).ravel()
    if np.std(a) > 0 and np.std(b) > 0:
        out["increment_corr"] = float(np.corrcoef(a, b)[0, 1])
    if out["rmse_wrf_increment"] > 0:
        out["residual_fraction"] = out["rmse_port_wrf"] / out["rmse_wrf_increment"]
    return out


def main():
    port_p, wrf_p, raw_p, out_p = sys.argv[1:5]
    res = {"port": port_p, "wrf": wrf_p, "raw": raw_p, "fields": {}}
    with netCDF4.Dataset(port_p) as p, netCDF4.Dataset(wrf_p) as w, netCDF4.Dataset(raw_p) as r:
        for group, names in (("prognostic", PROGNOSTIC), ("land", LAND), ("diagnostic", DIAG)):
            for name in names:
                if name in p.variables and name in w.variables and name in r.variables:
                    res["fields"][name] = {"group": group, **stats(p.variables[name][0], w.variables[name][0],
                                                                   r.variables[name][0])}
    with open(out_p, "w") as fh:
        json.dump(res, fh, indent=1)
    for name, s in res["fields"].items():
        print(f"{name:8s} {s['group'][:4]} rmse(port-wrf)={s['rmse_port_wrf']:.4g} max={s['max_port_wrf']:.4g} "
              f"wrf_inc={s['rmse_wrf_increment']:.4g} port_inc={s['rmse_port_increment']:.4g} "
              f"corr={s.get('increment_corr', float('nan')):.3f} frac={s.get('residual_fraction', float('nan')):.3f}")


if __name__ == "__main__":
    main()
