"""Genuine two-own-step CPU smoke on real inputs with one MP namelist edit.

Usage: coupled_smoke.py SOURCE_ROOT CASE OUTPUT_JSON OPTION
No patching model dispatch; no CPU-WRF trajectory or GPU speed claim.
"""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

root, case, output = (Path(x).resolve() for x in sys.argv[1:4])
option = int(sys.argv[4])
sys.path.insert(0, str(root / 'src'))
import jax
import numpy as np
from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
from gpuwrf.runtime import operational_mode as om
from gpuwrf.diagnostics.census import GUARDS, EVENTS, f2_record

assert jax.devices()[0].platform == 'cpu'
assert om.__file__.startswith(str(root))
print('CPU_SMOKE_IMPORTED', option, flush=True)
# The input view contains only a changed namelist and read-only symlinks.
view = output.parent / ('inputs_mp' + str(option))
view.mkdir(exist_ok=False)
text = (case / 'namelist.input').read_text()
import re
text, count = re.subn(r'(?im)^(\s*mp_physics\s*=)[^\n]*', lambda m: m[1] + ' ' + str(option) + ',', text)
assert count == 1
(view / 'namelist.input').write_text(text)
for path in case.iterdir():
    if path.name != 'namelist.input':
        (view / path.name).symlink_to(path)
cfg = NestedPipelineConfig(view, output.parent / 'smoke_out', output.parent / 'smoke_proof', hours=1, max_dom=1)
t0 = time.monotonic()
handles, bundles, _, _, _, carries = _load_domains(cfg, ('d01',))
ns, carry = bundles['d01'].namelist, carries['d01']
print('CPU_SMOKE_LOADED', option, time.monotonic()-t0, flush=True)
assert ns.mp_physics == option
suite = om._resolve_operational_suite(ns)
assert suite.microphysics.option == option
clock = om.build_clock_base(ns)
loaded = time.monotonic()
result = carry
seconds = []
for step in (1, 2):
    start = time.monotonic()
    result = om._advance_chunk(result, ns, step, clock, n_steps=1, cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(result)
    seconds.append(time.monotonic() - start)
    print('CPU_SMOKE_STEP', option, step, seconds[-1], flush=True)
leaves = [np.asarray(x) for x in jax.tree.leaves(result)]
nonfinite = sum(int((~np.isfinite(x)).sum()) for x in leaves if np.issubdtype(x.dtype, np.floating))
census = getattr(result, 'census', None)
receipt = {'source': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'platform':'cpu','GPU_execution':False,'scope':'CPU execution smoke only; forecast D6 pending',
 'jax_disable_jit':bool(jax.config.jax_disable_jit),
 'case':str(case),'namelist_sha256':hashlib.sha256(text.encode()).hexdigest(),
 'option':option,'selected_microphysics':{'module':suite.microphysics.owner_module,'entrypoint':suite.microphysics.entrypoint},
 'load_s':loaded-t0,'steps_s_including_compile':seconds,
 'nonfinite':nonfinite,'peakRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'guards':None if census is None else {g:dict(zip(EVENTS,map(int,c))) for g,c in zip(GUARDS,np.asarray(census.guards))},
 'f2':None if census is None else f2_record(census,result.state.theta.shape),
 'work':{n:getattr(ns,n) for n in ('dt_s','mp_physics','bl_pbl_physics','sf_surface_physics','ra_lw_physics','ra_sw_physics','cu_physics','radiation_cadence_steps','cumulus_cadence_steps','ferrier_gid')}}
if option == 38:
    from gpuwrf.contracts.precision import THOMPSON_HAIL_SURFACE_LEAVES
    receipt['conditional_state'] = {}
    for name in ('qvolg','nwfa','nifa',*THOMPSON_HAIL_SURFACE_LEAVES):
        value = np.asarray(getattr(result.state,name))
        if name in THOMPSON_HAIL_SURFACE_LEAVES:
            assert value.dtype == np.float32
        receipt['conditional_state'][name] = {'shape':value.shape,'dtype':str(value.dtype),
            'bytes':value.nbytes,'min':float(value.min()),'max':float(value.max())}
output.write_text(json.dumps(receipt,indent=2,default=str)+'\n')
assert nonfinite == 0
print(json.dumps(receipt,default=str))
