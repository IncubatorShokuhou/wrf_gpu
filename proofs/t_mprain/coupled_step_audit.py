"""One-hour CPU-port trajectory vs original CPU-WRF, paired at every step.

Uses the product's portable CPU dispatch; no GPU solver emulation/patching.
CPU compilation/fusion can differ from CUDA. This is CPU causal/finite
evidence, not GPU qualification or a timing benchmark.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

root, case, reference, output = (Path(p).resolve() for p in sys.argv[1:5])
assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
sys.path.insert(0,str(root/"src"))
import jax
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import _build_subset_output_fields, _dimension_sizes

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(str(root))
source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root,text=True).strip()
cfg=NestedPipelineConfig(case,output.parent/"unused_out",output.parent/"unused_proof",hours=1,max_dom=1)
t0=time.perf_counter()
handles,bundles,_,_,_,carries=_load_domains(cfg,("d01",))
ns,carry=bundles["d01"].namelist,carries["d01"]
assert ns.mp_physics == 16 and ns.dt_s == 18
clock=om.build_clock_base(ns)
print("LOADED",time.perf_counter()-t0,flush=True)
fields=("T","THM","QVAPOR","QCLOUD","QRAIN","QICE","QSNOW","QGRAUP","RAINNC","SNOWNC","GRAUPELNC")
dimensions=_dimension_sizes(nx=42,ny=42,nz=44,namelist={})
result=dict(source=source,platform="cpu",gpu_executed=False,scope=__doc__,steps=[],cache="NONE GPUWRF_JAX_CACHE=0")
loaded=time.perf_counter()
with Dataset(reference) as ds:
    assert ds["Times"].shape[0] == 201, ds["Times"].shape
    for step in range(201):
        if Path("/tmp/wrf_gpu2_quiet").exists():
            # Do not run CPU/disk-heavy work across a newly opened quiet window.
            print("QUIET_WAIT",step,flush=True)
            while Path("/tmp/wrf_gpu2_quiet").exists():
                time.sleep(5)
        started=time.perf_counter()
        if step:
            carry=om._advance_chunk(carry,ns,step,clock,n_steps=1,cadence=int(ns.radiation_cadence_steps))
            jax.block_until_ready(carry)
        if step in (0,1,2,200):
            leaves=[np.asarray(a) for a in jax.tree.leaves(carry)]
            assert all(np.isfinite(a).all() for a in leaves if np.issubdtype(a.dtype,np.floating))
        actual=_build_subset_output_fields(carry.state,None,{"use_theta_m":1},dimensions,requested_names=frozenset(fields))
        # The subset writer supports RAINNC; its FULL-history extension writes
        # the frozen subsets from these same disjoint State channels.
        actual["SNOWNC"] = np.asarray(carry.state.snow_acc) + np.asarray(carry.state.ice_acc)
        actual["GRAUPELNC"] = np.asarray(carry.state.graupel_acc)
        row=dict(step=step,lead_s=step*18,wall_s=time.perf_counter()-started,fields={})
        for name in fields:
            a=np.asarray(actual[name],np.float64)
            if a.shape[0] == 1:
                a=a[0]
            b=np.asarray(ds[name][step],np.float64)
            assert a.shape == b.shape,(name,a.shape,b.shape)
            assert np.isfinite(a).all() and np.isfinite(b).all(),(step,name)
            diff=a-b
            where=np.unravel_index(np.abs(diff).argmax(),diff.shape)
            row["fields"][name]=dict(rmse=float(np.sqrt(np.mean(diff*diff))), max_abs=float(np.max(np.abs(diff))),
                                     max_index=list(map(int,where)),actual_at_max=float(a[where]),reference_at_max=float(b[where]))
        result["steps"].append(row)
        if step < 3 or step%20 == 0:
            result["elapsed_s"]=time.perf_counter()-t0
            output.write_text(json.dumps(result,indent=2)+"\n")
            print("STEP",step,"RAINNC",row["fields"]["RAINNC"],"step_s",row["wall_s"],flush=True)
result.update(elapsed_s=time.perf_counter()-t0,load_s=loaded-t0,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
if carry.census is not None:
    from gpuwrf.diagnostics.census import f2_negative_summary
    result["guards"] = np.asarray(carry.census.guards).tolist()
    result["f2_negative_summary"] = f2_negative_summary(carry.census, carry.state.theta.shape)
output.write_text(json.dumps(result,indent=2)+"\n")
