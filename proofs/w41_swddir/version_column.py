"""w41-swddir (a): does a given source tree's RELEASE SW path emit WRF's SWDDIR?  Version-neutral (v0.3.3 .. now).

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 PYTHONPATH=<tree>/src python version_column.py OUT_JSON [NCOL]
Solves the release RRTMG SW column path of the gpuwrf on PYTHONPATH (REAL entry + fused band kernel; Pallas forced to
CPU interpret) on the first NCOL (default 16) cloudy day RE01 columns with the arm-A caller operands (Thompson radii,
maximum-random overlap, own radiation call; GHG as tests/v025/fid_q2/test_rrtmg_mp_re_port_today.py) and reports
SWDDIR / SWDOWN vs pristine RE01 arm A at the frozen tier-1 bound (1 W/m2 + 5 %).  Only APIs present since v0.3.3.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import jax.experimental.pallas as _pallas

_orig_pallas_call = _pallas.pallas_call


def _interpret_pallas_call(*args, **kwargs):  # v0.3.3 kernels call jax.experimental.pallas directly (no CPU backend)
    kwargs["interpret"] = True
    return _orig_pallas_call(*args, **kwargs)


_pallas.pallas_call = _interpret_pallas_call

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import gpuwrf  # noqa: E402
from gpuwrf.physics import rrtmg_sw as R  # noqa: E402
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time  # noqa: E402

FIXTURE = Path(__file__).resolve().parents[2] / "tests/v025/fid_q2/fixtures/rrtmg_mp_re_0227_v1.npz"
RRSW_SCON = 1368.22


def main(out, ncol=16):
    assert jax.devices()[0].platform == "cpu" and R._real_entry() and R._sw_band_sums_mode(), "release SW path"
    fx = dict(np.load(FIXTURE))
    pick = np.flatnonzero((fx["category"] != "clear") & (fx["col_coszen"] > 0.05))[:ncol]
    hours = fx["in_tau"][pick].astype(int)
    rows = {}
    for h in np.unique(hours):
        idx = pick[hours == h]
        a = lambda k, cols=idx: jnp.asarray(np.asarray(fx[k][cols], np.float64))  # noqa: E731
        ghg = clwrf_ssp245_gases_for_time(datetime(2026, 2, 28) + np.timedelta64(int(h), "h").astype(object))
        nz = fx["in_t"].shape[1]
        st = R.RRTMGSWColumnState(
            T=a("in_t"), p=a("in_p_hyd"), qv=a("in_qv"), qc=a("in_qc_rad"), qi=a("in_qi_rad"), qs=a("in_qs"),
            qg=a("in_qg"), cloud_fraction=a("in_cldfra"), dz=a("in_dz8w")[:, :nz], rho=a("in_rho"),
            pressure_interfaces=a("in_p_hyd_w"), temperature_interfaces=a("in_t8w"), ozone_vmr=a("col_o3"),
            co2_vmr=ghg.co2_vmr, n2o_vmr=ghg.n2o_vmr, ch4_vmr=ghg.ch4_vmr, surface_albedo=a("in_albedo"),
            coszen=a("col_coszen"), solar_source_scale=jnp.asarray(np.asarray(fx["col_solcon"][idx], np.float64) / RRSW_SCON),
            re_cloud=a("col_re_cloud"), re_ice=a("col_re_ice"), re_snow=a("col_re_snow"), xland=a("in_xland"))
        sw = R.solve_rrtmg_sw_column(st, with_clear_sky=True)
        for k, c in enumerate(idx):
            rows[int(c)] = {"category": str(fx["category"][c]), "swddir": float(sw.surface_direct[k]),
                            "swdnb": float(sw.surface_down[k]), "wrf_swddir": float(fx["A_swddir"][c]),
                            "wrf_swdnb": float(fx["A_swdnb"][c])}
    ratio = lambda k: max(abs(r[k] - r["wrf_" + k]) / (1.0 + 0.05 * abs(r["wrf_" + k])) for r in rows.values())  # noqa: E731
    result = {"gpuwrf": gpuwrf.__file__, "columns": rows, "swddir_worst_ratio": ratio("swddir"),
              "swdnb_worst_ratio": ratio("swdnb"),
              "swddir_max_abs_err": max(abs(r["swddir"] - r["wrf_swddir"]) for r in rows.values())}
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "columns"}))


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3]))
