"""Carry forward KM02 only after native TKE stage jaxpr + constants identity.

For the Swiss mp8/moist_adv_opt1 configuration, source changes since ec9aaddd9
select the same numerical expressions:
use_flux_advection=True selects the same transport expression, and km2 already
takes the preceding literal branch, so narrowing the scaffold elif is inert.
The new LES fallback is excluded when the release native PD flag is enabled;
the extra-microphysics refusal is false for mp8. CLI defaults do not alter an
explicit moist_adv_opt1. No numerical source changes enter this smoke arm.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from proofs.o2_km2 import stage_oracle as stage
from gpuwrf.runtime import les3d_km3 as current
from gpuwrf.runtime import operational_mode  # Eager import before tracing host-built tables.

# Same module blob as the measured ec9aaddd9 smoke, on the retained O1 branch.
BASE = 'eeb18b81b'
assert jax.devices()[0].platform == 'cpu'
assert os.environ.get('GPUWRF_DYN_PD_FP32') == '1'
old_source = subprocess.check_output(['git','show',BASE+':src/gpuwrf/runtime/les3d_km3.py'],text=True)
old_func = next(n for n in ast.parse(old_source).body if isinstance(n,ast.FunctionDef) and n.name=='les_tke_stage_update')
namespace = vars(current).copy()
exec(compile(ast.Module(body=[old_func],type_ignores=[]), '<old-stage>', 'exec'), namespace)
old = namespace['les_tke_stage_update']

def digest(jx):
    text = re.sub(r'name_and_src_info=[^\]\n]*','',str(jx))
    text = re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',text)
    consts = [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in jx.consts]
    return hashlib.sha256(text.encode()).hexdigest(), consts

with np.load(stage.FIXTURE) as z:
    state, source, vel, nl = stage.inputs(z, False, 0)
report = {'baseline':BASE,'candidate':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'native_pd_flag':os.environ['GPUWRF_DYN_PD_FP32'],'stages':{}}
for rk in (1,2,3):
    def trace(fn):
        return jax.make_jaxpr(lambda q,q0,sc: fn(state.replace(tke=q),state.replace(tke=q0),
            tke_tend=sc,rk_step=rk,dt_rk=nl.dt_s/(4-rk),namelist=nl,transport_velocities=vel).tke)(
                state.tke,state.tke,source)
    a,b=digest(trace(old)),digest(trace(current.les_tke_stage_update))
    report['stages'][str(rk)]={'before':a[0],'after':b[0],'constants_equal':a[1]==b[1]}
    assert a==b, report
Path(sys.argv[1]).write_text(json.dumps(report,indent=1))
print(json.dumps(report,indent=1))
