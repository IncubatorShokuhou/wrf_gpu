"""w41-swddir LG02: legacy entry (GPUWRF_FAST_DEFAULTS=0, f64 XLA, constant radii, random overlap) SWDDIR vs pristine
RE01 arm D, fixed vs the pre-fix product (the XLA tile solver without the unscaled direct depths).  CPU.

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=0 python legacy_arm_d.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/v025/fid_q2"))
from gpuwrf.physics import rrtmg_sw as R  # noqa: E402
from rrtmg_mp_re_fixture import FIXTURE, columns_outside, load  # noqa: E402
from test_rrtmg_mp_re_port_today import _port_today  # noqa: E402

assert not R._real_entry() and not R._sw_band_sums_mode(), "legacy entry expected (GPUWRF_FAST_DEFAULTS=0)"


def run(mutant):
    original = R._sw_band_tile_fluxes
    if mutant:
        R._sw_band_tile_fluxes = lambda *a, **k: original(*a, **{**k, "direct_taus": None})
    _port_today.__globals__["solve_rrtmg_sw_column"].clear_cache()
    try:
        return _port_today(fx)
    finally:
        R._sw_band_tile_fluxes = original
        _port_today.__globals__["solve_rrtmg_sw_column"].clear_cache()


fx = load(FIXTURE)
cloudy_day = (fx["category"] != "clear") & (fx["col_coszen"] > 0.05)
out = {}
for tag, mutant in (("fixed", False), ("prefix_mutant", True)):
    port = run(mutant)
    err = np.abs(port["swddir"] - fx["D_swddir"])
    out[tag] = {"swddir_cols_outside_D": int(columns_outside(fx, {"swddir": port["swddir"]}, "D").sum()),
                "all_outputs_cols_outside_D": int(columns_outside(fx, port, "D").sum()),
                "swddir_max_abs_err_cloudy_day": float(err[cloudy_day].max()),
                "swddir_p90_abs_err_cloudy_day": float(np.percentile(err[cloudy_day], 90)),
                "swddir_max_abs_err_clear": float(err[fx["category"] == "clear"].max()),
                "swdnb_max_abs_err": float(np.abs(port["swdnb"] - fx["D_swdnb"]).max())}
    print(tag, out[tag], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
