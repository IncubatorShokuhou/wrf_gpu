"""WN3 0227 3-nest topo_wind composite: CPU load preflight (lane w41-topowind). No GPU.

Loads the composite (YSU+sfclay1, topo_wind=2,1,1, lane-copy wrfinputs with geogrid
VAR_SSO) through nested_pipeline._load_domains exactly like the CLI (CPU loader of
proofs/s1_port/release_identity.py), then per domain:
  * admission + bound PBL/sfclay, the TopoWindStatic bound to the namelist;
  * the port's resolved terrain vs WRF's grid%ht = CPU-WRF history HGT at t=0 of the
    ALISIOS run of the SAME geometry/inputs (nest blending included);
  * CTOPO/CTOPO2 recomputed from WRF's HGT with the byte-exact start_em port == bound static.
usage: JAX_PLATFORMS=cpu python wn3_preflight.py <repo> <case_input_dir> <out_dir>
"""
import json, sys, tempfile, time
from pathlib import Path

REPO, CASE, OUT = (Path(a).resolve() for a in sys.argv[1:4])
sys.path[:0] = [str(REPO / "src"), str(REPO / "proofs/o1_nlbind")]
import jax  # noqa: E402
import numpy as np  # noqa: E402
import netCDF4  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.backends import pallas as facade  # noqa: E402
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.io.topo_wind_static import start_em_ctopo  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
facade.cuda_native = lambda: False
_real = pl.pallas_call
pl.pallas_call = lambda *a, **k: _real(*a, **{**{x: y for x, y in k.items() if x != "compiler_params"}, "interpret": True})
WRF_HIST = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with tempfile.TemporaryDirectory(dir=OUT) as tmp:
    cfg = pipeline.NestedPipelineConfig(CASE, Path(tmp) / "out", Path(tmp) / "proof", hours=3, max_dom=3)
    _h, bundles, meta, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01", "d02", "d03"))
report = {"case": str(CASE), "load_s": round(time.time() - t0, 1), "domains": {}}
# WRF grid%ht = CPU-WRF history HGT at t=0: ALISIOS CPU-WRF output -> only through alisios_guard (wave V041 rule).
def _load_alisios_guard():
    """alisios_guard lives on the DA lanes (w41-da-core) until merged; stdlib-only module -> load by file path."""
    import importlib.util, os
    path = Path(os.environ.get("GPUWRF_ALISIOS_GUARD_SRC", "<USER_HOME>/src/wrf_gpu2_wt/w41-da-core/src")) / \
        "gpuwrf/assimilation/letkf/alisios_guard.py"
    spec = importlib.util.spec_from_file_location("w41_alisios_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
guarded_session = _load_alisios_guard().guarded_session
hist = {d: WRF_HIST / f"wrfout_{d}_2026-02-28_00:00:00" for d in ("d01", "d02", "d03")}
wrf_hts = {}
with guarded_session({p: "20260228" for p in hist.values()}, ["20260227"], OUT / "guard",
                     purpose="w41-topowind WN3 preflight: WRF history HGT t=0 (terrain/CTOPO identity)") as session:
    for d, p in hist.items():
        with netCDF4.Dataset(p) as h:
            wrf_hts[d] = np.asarray(h["HGT"][0], np.float32)
report["guard"] = dict(receipt=str(OUT / "guard/guarded_session.json"), opened=len(session.observed),
                       refused=len(session.refused))
ok = True
for d, tw in (("d01", 2), ("d02", 1), ("d03", 1)):
    nl, st = bundles[d].namelist, carries[d].state
    static = nl.topo_wind_static
    wrf_ht = wrf_hts[d]
    with netCDF4.Dataset(CASE / f"wrfinput_{d}") as w:
        var_sso, var2d = (np.asarray(w[k][0], np.float32) for k in ("VAR_SSO", "VAR"))
    port_ht = np.asarray(bundles[d].grid.terrain_height)
    ct, ct2 = start_em_ctopo(tw, wrf_ht, np.asarray(st.xland), var_sso=var_sso, var2d=var2d)
    row = dict(bl_pbl_physics=int(nl.bl_pbl_physics), sf_sfclay_physics=int(nl.sf_sfclay_physics),
               static_bound=static is not None,
               ht_max_abs_vs_wrf_history=float(np.abs(port_ht.astype(np.float64) - wrf_ht).max()),
               ht_dtype=str(port_ht.dtype))
    if static is not None:
        row["ctopo_byte_equal_from_wrf_ht"] = bool(np.array_equal(np.asarray(static.ctopo).view(np.uint32), ct.view(np.uint32)))
        row["ctopo2"] = None if static.ctopo2 is None else dict(
            byte_equal=bool(np.array_equal(np.asarray(static.ctopo2).view(np.uint32), ct2.view(np.uint32))),
            lt1=int((np.asarray(static.ctopo2) < 1).sum()))
        row["ctopo_range"] = [float(np.min(static.ctopo)), float(np.max(static.ctopo))]
        ok &= row["ctopo_byte_equal_from_wrf_ht"] and (static.ctopo2 is None or row["ctopo2"]["byte_equal"])
        ok &= (static.ctopo2 is None) == bool(np.all(ct2 == 1))
    ok &= row["static_bound"] and row["bl_pbl_physics"] == 1 and row["sf_sfclay_physics"] == 1
    report["domains"][d] = row
report["PASS"] = bool(ok)
(OUT / "WN3_PREFLIGHT.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
