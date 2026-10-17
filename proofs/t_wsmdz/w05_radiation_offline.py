"""t-wsmdz W05/W10: one RRTMG call of the PORT on an ORIGINAL CPU-WRF WSM3/5/6 h1 state, radii on/off.

Loads the case through the product (_load_domains, release defaults), overwrites the
thermodynamic/hydrometeor/geometry leaves with the CPU-WRF hour-1 history (THM, QV..QS,
P+PB, PH+PHB, MU+MUB, TSK), sets the held WSM5 radii from the port's WSM effectRad on that
state (what the adapter holds after the previous MP call), and calls the runtime's
rrtmg_radiation_diagnostics exactly as operational_mode does, with use_mp_re=1 (WRF
physics_init has_reqc/i/s=1) and use_mp_re=0 (the pre-fix port). Scores TOA OLR and
surface LW down against the CPU-WRF h1 OLR/GLW (WRF's last radiation call is 2 steps
earlier: operand approximation, disclosed). CPU evidence of the radiative mechanism only.
Usage: python w05_radiation_offline.py <src_root> <case_dir> <out.json>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root, case, output = (Path(p).resolve() for p in sys.argv[1:4])
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path.insert(0, str(root / "src"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from gpuwrf.coupling import physics_couplers as C  # noqa: E402
from gpuwrf.physics.microphysics_wsm6 import _effective_radii  # noqa: E402
from gpuwrf.io.netcdf_lock import Dataset  # noqa: E402

assert jax.devices()[0].platform == "cpu" and om.__file__.startswith(str(root))
source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
work = output.parent / (output.stem + "_work")
t0 = time.perf_counter()
cfg = NestedPipelineConfig(case / "input", work / "out", work / "proof", hours=1, max_dom=1)
_, bundles, _, _, _, carries = _load_domains(cfg, ("d01",))
ns, carry = bundles["d01"].namelist, carries["d01"]
assert int(ns.mp_physics) in (3, 4, 6) and int(ns.ra_lw_physics) == 4  # OLR/GLW are RRTMG LW
st = carry.state
ref_path = sorted((case / "cpu").glob("wrfout_d01_*"))[1]
with Dataset(ref_path) as d:
    v = lambda n: (np.asarray(d.variables[n][0], np.float32) if n in d.variables  # noqa: E731
                   else np.zeros_like(np.asarray(d.variables["QVAPOR"][0], np.float32)))  # WSM3: no QICE/QSNOW
    h = dict(thm=v("THM") + np.float32(300.0), qv=v("QVAPOR"), qc=v("QCLOUD"), qr=v("QRAIN"), qi=v("QICE"),
             qs=v("QSNOW"), p=v("P") + v("PB"), ph=v("PH") + v("PHB"), mu=v("MU") + v("MUB"), tsk=v("TSK"),
             olr=v("OLR"), glw=v("GLW"))
    times = d.variables["Times"][0].tobytes().decode()
like = lambda name, arr: jnp.asarray(arr, getattr(st, name).dtype)  # noqa: E731
st = st.replace(theta=like("theta", h["thm"]), qv=like("qv", h["qv"]), qc=like("qc", h["qc"]),
                qr=like("qr", h["qr"]), qi=like("qi", h["qi"]), qs=like("qs", h["qs"]),
                p=like("p_total", h["p"]), ph_total=like("ph_total", h["ph"]),
                mu_total=like("mu_total", h["mu"]), t_skin=like("t_skin", h["tsk"]))
# Held WSM radii = port WSM effectRad on this state (mp_wsm6_effectRad / effectRad_wsm5).
rvovrd = np.float32(np.float32(461.6) / np.float32(287.0))
pii = (h["p"] / np.float32(1.0e5)) ** np.float32(2.0 / 7.0)
t_dry = h["thm"] / (np.float32(1.0) + rvovrd * h["qv"]) * pii
rho = h["p"] / (np.float32(287.0) * h["thm"] * pii)
qc_re, qi_re, qs_re = h["qc"], h["qi"], h["qs"]
if int(ns.mp_physics) == 3:  # microphysics_wsm3: combined qc/qr split at T0C before effectRad
    warm = t_dry >= np.float32(273.15)
    qc_re, qi_re, qs_re = np.where(warm, h["qc"], 0), np.where(warm, 0, h["qc"]), np.where(warm, 0, h["qr"])
radii = _effective_radii(jnp.asarray(t_dry), jnp.asarray(qc_re, jnp.float32), jnp.asarray(qi_re, jnp.float32),
                         jnp.asarray(qs_re, jnp.float32), jnp.asarray(rho))
res = {"schema": "t-wsmdz-w05-v1", "source": source, "reference": str(ref_path), "valid": times,
       "operand_note": "CPU-WRF h1 history state; WRF's last radiation call was itimestep 199 (2 steps earlier)",
       "radii_range_um": {n: [float(np.asarray(r).min() * 1e6), float(np.asarray(r).max() * 1e6)]
                          for n, r in zip(("re_cloud", "re_ice", "re_snow"), radii)}, "arms": {}}
for arm, use in [a for a in (("use_mp_re1_wrf", 1), ("use_mp_re0_prefix", 0)) if str(a[1]) in os.environ.get("W05_ARMS", "10")]:
    s = st
    if s.re_cloud is not None:
        s = s.replace(re_cloud=jnp.asarray(radii[0], jnp.float32), re_ice=jnp.asarray(radii[1], jnp.float32),
                      re_snow=jnp.asarray(radii[2], jnp.float32))
    assert use == 0 or s.re_cloud is not None, "candidate must seed WSM radius leaves"
    rad = C.rrtmg_radiation_diagnostics(
        s, ns.grid, time_utc=ns.time_utc, lead_seconds=3564.0, radiation_static=ns.radiation_static,
        topo_shading=int(ns.topo_shading), slope_rad=int(ns.slope_rad),
        shadow_length_m=float(ns.topo_shadow_length_m), land_state=getattr(carry, "noahmp_land", None),
        use_mp_re=use, mp_physics=int(ns.mp_physics), **om._rrtmg_pbl_cloud_kwargs(ns))
    olr = np.asarray(rad.lw_toa_up, np.float64)
    glw = np.asarray(rad.glw, np.float64)
    rec = {}
    for name, a, r in (("OLR", olr, h["olr"]), ("GLW", glw, h["glw"])):
        dd = a - np.asarray(r, np.float64)
        inner = dd[5:-5, 5:-5]
        rec[name] = {"bias": float(dd.mean()), "rmse": float(np.sqrt((dd ** 2).mean())),
                     "interior_bias": float(inner.mean()), "interior_rmse": float(np.sqrt((inner ** 2).mean())),
                     "max_abs": float(np.abs(dd).max())}
    res["arms"][arm] = rec
    print(arm, json.dumps(rec), flush=True)
res["wall_s"] = time.perf_counter() - t0
output.write_text(json.dumps(res, indent=1))
print("DONE", output, flush=True)
