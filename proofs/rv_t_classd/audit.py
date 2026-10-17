import os, sys, json, hashlib, pickle
from pathlib import Path
from dataclasses import replace
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/snapshot')
OUT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
import pytest
from test_v040_urban_boundaries import urban_case
from test_online_diagnostics_restart_store import identity, rewrite
from gpuwrf.runtime.restart_store import RestartStore
from gpuwrf.io.restart_schema import validate_urban_bep

mp=pytest.MonkeyPatch()
grid,nl,carry=urban_case.__wrapped__(mp)(3)
records={}
for kind in ('healthy','missing','shape','precision','nan'):
    urban=carry.urban_bep
    leaf=urban.green_roof
    if kind=='missing': leaf=None
    elif kind=='shape': leaf=np.ones((1,1),np.float32)
    elif kind=='precision': leaf=np.asarray(leaf,np.float64)
    elif kind=='nan':
        leaf=np.asarray(leaf).copy();leaf.flat[0]=np.nan
    changed=urban._replace(green_roof=leaf)
    rec={}
    try:
        validate_urban_bep(changed,nl,grid_shape=(grid.ny,grid.nx),nz=grid.nz)
        rec['direct_validator']='ACCEPT'
    except ValueError as e: rec['direct_validator']='REFUSE '+str(e)
    store=RestartStore(OUT/'restart_probe'/kind,10**7,2,0)
    try:
        generation,_=store.save({'d01':carry.replace(urban_bep=changed)}, {'d01':2}, {'d01':nl.dt_s}, identity(nl))
        restored=store.read(generation,expected_identity=identity(nl))[0]['carries']['d01']
        rec['native_save_read']='ACCEPT'
        rec['restored_shape']=None if restored.urban_bep.green_roof is None else list(restored.urban_bep.green_roof.shape)
    except ValueError as e: rec['native_save_read']='REFUSE '+str(e)
    # Read-boundary corruption: retain all identity/hash checks, mutate only QGR.
    store=RestartStore(OUT/'restart_read_probe'/kind,10**7,2,0)
    generation,_=store.save({'d01':carry},{'d01':2},{'d01':nl.dt_s},identity(nl))
    def change(payload,manifest):
        payload['extra_fields']['urban_bep']=payload['extra_fields']['urban_bep']._replace(green_roof=leaf)
    rewrite(generation,change)
    try:
        store.read(generation,expected_identity=identity(nl))
        rec['native_authenticated_read']='ACCEPT'
    except ValueError as e: rec['native_authenticated_read']='REFUSE '+str(e)
    records[kind]=rec
mp.undo()
(OUT/'RV02_restart.json').write_text(json.dumps(records,indent=2)+'\n')
print('RESTART',json.dumps(records),flush=True)

from gpuwrf.runtime.aot_cheap_key import program_key, cheap_key
from gpuwrf.runtime.aot_baked_tables import baked_table_hash
import gpuwrf.diagnostics.morrison_radar as M
cols=np.load(ROOT/'proofs/t_classd/morr_radar_oracle/fixtures/inputs.npz')['columns']
args=tuple(jax.numpy.asarray(cols[:,i,:]) for i in range(9))
radar_nl=replace(nl,mp_physics=10,do_radar_ref=1,urban_bundle=None,sf_urban_physics=0,
                 sf_sfclay_physics=0,bl_pbl_physics=0,cu_physics=0,ra_lw_physics=0,ra_sw_physics=0)
@jax.jit
def radar(*columns):return M.morrison_reflectivity(*columns)
keys=[]
old=M.CONSTANTS
for factor in (1.,1.01):
    c=dict(old);c['c']=dict(c['c']);snow=dict(c['c']['s']);snow['am']=np.float32(snow['am']*factor);c['c']['s']=snow
    M.CONSTANTS=c
    radar.clear_cache()
    hlo=radar.lower(*args).as_text()
    value=np.asarray(radar(*args))
    keys.append(dict(factor=factor,program_key=program_key(radar,args,{},radar_nl),
                     cheap_key=cheap_key(radar,args,{},radar_nl),baked_hash=baked_table_hash(radar,radar_nl),
                     stablehlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                     value_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
M.CONSTANTS=old
(OUT/'RV03_cache.json').write_text(json.dumps(keys,indent=2)+'\n')
print('CACHE',json.dumps(keys),flush=True)
