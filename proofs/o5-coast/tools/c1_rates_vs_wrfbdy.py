"""C1: root walk rates vs the wrfbdy file's own real.exe tendency arrays (o5-coast, CPU only).

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=0 GPUWRF_JAX_CACHE=0 python c1_rates_vs_wrfbdy.py CASE_DIR OUT.json
Loads the case through the production root loader (d02_replay.build_replay_case(standalone=True)), builds
coupling.root_spec_walk.root_walk_rates at several step-end leads, and compares ring 0 with MU/T/PH/U/V _BTXS/_BTXE/_BTYS/_BTYE
of the record bracket WRF uses (dtbc after solve_em.F:378: l = ceil(lead/interval)-1) with WRF spec_bdytend
ownership (S/N rows full, W/E rows 1..len-2).  Reports max |rate - BT| / max |BT| per field and side."""
import json
import math
import re as _re
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

case, out = Path(sys.argv[1]), Path(sys.argv[2])
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from gpuwrf.coupling.boundary_apply import BoundaryConfig  # noqa: E402
from gpuwrf.coupling.root_spec_walk import root_walk_rates  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402

# Only the production wrfbdy -> State loader is needed (no physics init): build_replay_case(standalone=True)
# is the call nested_pipeline._load_domains makes for the root.
rc = build_replay_case(case, domain="d01", standalone=True)
state, metrics = rc.state, rc.metrics
nz, ny, nx = (int(n) for n in state.theta.shape)
interval = float(_re.search(r"(?mi)^\s*interval_seconds\s*=\s*(\d+)", (case / "namelist.input").read_text()).group(1))
dt = float(_re.search(r"(?mi)^\s*time_step\s*=\s*(\d+)", (case / "namelist.input").read_text()).group(1))
cfg = BoundaryConfig(spec_bdy_width=5, spec_zone=1, relax_zone=4, update_cadence_s=interval)
f = nc.Dataset(case / "wrfbdy_d01")
res = {"case": str(case), "nz_ny_nx": [nz, ny, nx], "interval": interval, "dt": dt, "leads": {}}
worst = 0.0
for step in (1, 2, int(interval / dt), int(interval / dt) + 1, int(1.5 * interval / dt)):
    lead = step * dt
    rec = math.ceil(lead / interval) - 1
    r = root_walk_rates(state, lead, metrics, cfg, theta_offset=300.0, nz=nz, ny=ny, nx=nx,
                        dtype=np.float64)
    rows = {}
    for name, arr, var, zl in (("mu", np.asarray(r.mu)[None], "MU", 1), ("theta", np.asarray(r.theta), "T", nz),
                               ("ph", np.asarray(r.ph), "PH", nz + 1), ("u", np.asarray(r.u), "U", nz),
                               ("v", np.asarray(r.v), "V", nz)):
        def bt(side):
            a = np.asarray(f[f"{var}_BT{side}"][rec], np.float64)  # (bdy_width, [z,] tan)
            return a[0][None] if a.ndim == 2 else a[0]
        ref = {"XS": bt("XS"), "XE": bt("XE"), "YS": bt("YS"), "YE": bt("YE")}
        got = {"XS": arr[:, :, 0], "XE": arr[:, :, -1], "YS": arr[:, 0, :], "YE": arr[:, -1, :]}
        side_err = {}
        for side in ("XS", "XE", "YS", "YE"):
            g, t = got[side], ref[side][:, : got[side].shape[-1]]
            if side in ("XS", "XE"):
                g, t = g[:, 1:-1], t[:, 1:g.shape[-1] - 1]
            scale = max(float(np.abs(t).max()), 1e-30)
            side_err[side] = float(np.abs(g - t).max() / scale)
        rows[name] = side_err
        worst = max(worst, max(side_err.values()))
    res["leads"][str(lead)] = dict(record=rec, rel_err=rows)
    print(lead, rec, {k: max(v.values()) for k, v in rows.items()})
res["worst_rel_err"] = worst
out.write_text(json.dumps(res, indent=1))
print("worst", worst)
