"""RUC wiring mutant or unchanged non-RUC closed-jaxpr proof; CPU only."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

source,out=map(Path,sys.argv[1:3]);mode=sys.argv[3]
assert os.environ['JAX_PLATFORMS']=='cpu'
sys.path[:0]=[str(source/'scripts/v025'),str(source/'src'),str(source/'tests')]
import cpu_guard
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling import goddard_radiation as goddard
from gpuwrf.runtime import operational_mode as om

result={'platform':'cpu','gpu_executed':False,'cache':'NONE','mode':mode,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()}
if mode=='mutant':
    import pytest
    class Capture:
        def __init__(self):self.assertions=[];self.errors=[]
        def pytest_runtest_makereport(self,item,call):
            if call.excinfo is not None:
                if call.when=='call' and call.excinfo.errisinstance(AssertionError):
                    self.assertions.append(str(call.excinfo.value)[:1000])
                else:self.errors.append(str(call.excinfo.value)[:1000])
    capture=Capture();original=om._explicit_ruc
    # Restore the rejected omission without editing the immutable source.
    om._explicit_ruc=lambda nl:False
    try:
        rc=pytest.main([str(source/'tests/test_goddard_surface_radiation.py')+'::test_real_ruc_surface_operands_reach_refresh',
            '-q','--basetemp','<USER_HOME>/wrf_gpu2_lanes/s2-goddardrad/pytest'],plugins=[capture])
    finally:om._explicit_ruc=original
    result.update(assertions=capture.assertions,setup_errors=capture.errors,pytest_returncode=int(rc))
    result['mutant_killed']=rc==1 and len(capture.assertions)==1 and not capture.errors
    out.write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['mutant_killed'] else 1)

assert mode=='trace'
from dataclasses import replace
from typing import NamedTuple
from gpuwrf.coupling.physics_couplers import RRTMGRadiationStatic
from gpuwrf.validation.tier2 import make_ideal_grid
from test_v0110_wrfrst_netcdf import _state
old_source=subprocess.check_output(['git','show','21fa82da9:src/gpuwrf/coupling/goddard_radiation.py'],cwd=source,text=True)
old_tree=ast.parse(old_source);new_tree=ast.parse((source/'src/gpuwrf/coupling/goddard_radiation.py').read_text())
def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
old_functions,new_functions=functions(old_tree),functions(new_tree)
unchanged={name:ast.dump(old_functions[name],include_attributes=False)==ast.dump(node,include_attributes=False)
           for name,node in new_functions.items() if name!='refresh_driver'}
assert all(unchanged.values())
namespace=dict(goddard.__dict__)
exec(ast.get_source_segment(old_source,old_functions['refresh_driver']),namespace)
old_refresh=namespace['refresh_driver']
grid=make_ideal_grid(nz=5,ny=2,nx=3);z=jnp.zeros((2,3),jnp.float32)
static=RRTMGRadiationStatic(jnp.full_like(z,28),jnp.full_like(z,-16),z,z,z,z,jnp.ones_like(z))
nl=om.OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,use_noahmp=False,
    sf_surface_physics=0,bl_pbl_physics=0,sf_sfclay_physics=0,cu_physics=0,
    ra_lw_physics=5,ra_sw_physics=5,radiation_static=static,time_utc='2026-06-03T12:00:00Z')
carry=om._initial_carry_for_run(_state(grid),nl)
class Surface(NamedTuple):
    albedo:object
    emiss:object
    t_skin:object
surface=Surface(jnp.full_like(z,.41),jnp.full_like(z,.93),carry.state.t_skin)
records=[]
for name,land,urban in [('no_lsm',None,0),('noahmp',surface,0),('slucm_surface',surface,1)]:
    config=replace(nl,use_noahmp=land is not None,sf_surface_physics=4 if land is not None else 0,sf_urban_physics=urban)
    value=carry.replace(noahmp_land=land)
    def trace(fn):return jax.make_jaxpr(lambda c:fn(c,config,600.,True,None))(value)
    before,after=trace(old_refresh),trace(goddard.refresh_driver)
    assert str(before.jaxpr)==str(after.jaxpr)
    assert len(before.consts)==len(after.consts)
    for a,b in zip(before.consts,after.consts,strict=True):
        a,b=np.asarray(a),np.asarray(b)
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    records.append({'surface':name,'jaxpr_sha256':hashlib.sha256(str(after.jaxpr).encode()).hexdigest(),
                    'constant_count':len(after.consts),'jaxpr_and_constants_BYTE':True})
result.update(prior_commit='21fa82da9',unchanged_functions=unchanged,records=records)
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
