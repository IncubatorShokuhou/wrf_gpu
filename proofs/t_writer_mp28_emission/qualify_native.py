"""Measure actual standalone initialization against original CPU-WRF frame0."""
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset
from gpuwrf.integration.d02_replay import build_replay_case
from gpuwrf.coupling import physics_couplers as pc

assert jax.devices()[0].platform == 'cpu'
out = Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
started = time.monotonic()
case = build_replay_case('<USER_HOME>/wrf_gpu2_lanes/v-run/round3/mp28_followup/mp28/input',
                         domain='d01',standalone=True)
loaded = time.monotonic()
state = pc.thompson_aero_coldstart_init(case.state,case.grid)
state.nwfa2d.block_until_ready()
initialized = time.monotonic()
refpath = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c19/cpu/wrfout_d01_2023-01-15_00:00:00')
with Dataset(refpath) as ds:
    ds.set_auto_mask(False)
    gold = np.asarray(ds['QNWFA2D'][0]);dust=np.asarray(ds['QNIFA2D'][0])
def stats(a):
    a=np.asarray(a,dtype='f4')
    ulp=np.abs(a.view('i4').astype('i8')-gold.view('i4'))
    return dict(max_abs=float(np.max(np.abs(a.astype('f8')-gold))),max_ulp=int(ulp.max()),
                max_relative=float(np.max(np.abs(a.astype('f8')-gold)/gold)),
                exact_cells=int(np.count_nonzero(a==gold)),cells=a.size,
                mean=float(a.mean()),j21_i18=float(a[21,18]))
eager=stats(state.nwfa2d)
compiled=jax.jit(lambda s:pc._aerosol_surface_emission_columns(s)[0])(case.state)
compiled.block_until_ready();jit=stats(compiled)
again=pc.thompson_aero_coldstart_init(state.replace(ph_total=state.ph_total*1.1),case.grid)
receipt=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    platform='cpu', original_cpu=str(refpath), cpu_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
    native_standalone=case.metadata.get('standalone_native_init'),
    input_directory='<USER_HOME>/wrf_gpu2_lanes/v-run/round3/mp28_followup/mp28/input',
    input_CCNS_present=bool(np.any(np.asarray(case.state.nwfa))) if case.state.nwfa is not None else False,
    eager=eager,compiled=jit,
    eager_bound_ulp=4,compiled_relative_bound=float(32*np.finfo('f4').eps),
    QNIFA2D_byte_exact=np.asarray(state.nifa2d).tobytes()==dust.tobytes(),
    held_after_geometry_change=np.asarray(again.nwfa2d).tobytes()==np.asarray(state.nwfa2d).tobytes(),
    load_s=loaded-started,emission_and_coldstart_s=initialized-loaded,
    wall_s=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    cache='NONE; GPUWRF_JAX_CACHE=0',scope='CPU frame0 only; no GPU/E41/full-forecast PASS')
(out/'NATIVE_FRAME0.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
assert receipt['native_standalone']
assert eager['max_ulp']<=4
assert jit['max_relative']<=receipt['compiled_relative_bound']
assert receipt['QNIFA2D_byte_exact'] and receipt['held_after_geometry_change']
