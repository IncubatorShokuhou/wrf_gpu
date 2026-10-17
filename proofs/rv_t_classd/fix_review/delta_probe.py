import sys,json,hashlib,ctypes,time
from dataclasses import replace
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/fix_review/snapshot')
OUT=ROOT.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'tests/v040')]
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from test_v040_urban_boundaries import urban_case
from test_online_diagnostics_restart_store import identity,rewrite
from gpuwrf.runtime.restart_store import RestartStore,nested_identity,_urban_bep_restart_config
from datetime import datetime
from types import SimpleNamespace
mp=pytest.MonkeyPatch()
grid,nl,carry=urban_case.__wrapped__(mp)(3)
ident=identity(nl)
# Exercise the production identity builder, not only the test fixture's dict.
cfg=SimpleNamespace(input_dir=OUT,output_dir=OUT/'out',feedback=False,emit_initial_history=True)
actual=nested_identity(cfg,{'d01':SimpleNamespace(namelist=nl)},datetime(2023,1,15))
assert actual['urban_bep_config']=={'d01':_urban_bep_restart_config(nl)}
native={}
for kind in ('healthy','missing','shape','precision','group','unbound','domain'):
    leaf=carry.urban_bep.green_roof;group=carry.urban_bep
    this_ident=dict(ident)
    if kind=='missing':group=group._replace(green_roof=None)
    elif kind=='shape':group=group._replace(green_roof=np.ones((1,1),np.float32))
    elif kind=='precision':group=group._replace(green_roof=np.asarray(leaf,np.float64))
    elif kind=='group':group=None
    elif kind=='unbound':this_ident.pop('urban_bep_config')
    elif kind=='domain':this_ident['urban_bep_config']={'d01':{'sf_urban_physics':3,'domain_shape':[8,1,1]}}
    row={};store=RestartStore(OUT/'native'/kind,10**7,2,0)
    try:
        generation,_=store.save({'d01':carry.replace(urban_bep=group)},{'d01':2},{'d01':nl.dt_s},this_ident)
        back=store.read(generation,expected_identity=this_ident)[0]['carries']['d01']
        row['save_read']='ACCEPT'
        assert kind=='healthy'
        a,at=jax.tree.flatten(carry);b,bt=jax.tree.flatten(back)
        assert at==bt
        for x,y in zip(a,b,strict=True):
            x,y=np.asarray(x),np.asarray(y)
            assert (x.dtype,x.shape,x.tobytes())==(y.dtype,y.shape,y.tobytes())
        row['fullcarry_byte_equal']=True
    except ValueError as e:row['save_read']='REFUSE '+str(e)
    if kind!='healthy':assert row['save_read'].startswith('REFUSE'),(kind,row)
    store=RestartStore(OUT/'native_read'/kind,10**7,2,0)
    generation,_=store.save({'d01':carry},{'d01':2},{'d01':nl.dt_s},ident)
    def change(payload,manifest):
        payload['extra_fields']['urban_bep']=group
        if kind in ('unbound','domain'):
            if kind=='unbound':manifest['identity'].pop('urban_bep_config')
            else:manifest['identity']['urban_bep_config']=this_ident['urban_bep_config']
    rewrite(generation,change)
    try:
        store.read(generation)
        row['authenticated_read']='ACCEPT'
    except ValueError as e:row['authenticated_read']='REFUSE '+str(e)
    assert (kind=='healthy')==(row['authenticated_read']=='ACCEPT'),(kind,row)
    native[kind]=row
mp.undo()
(OUT/'RV14_native.json').write_text(json.dumps(native,indent=2)+'\n')
print('NATIVE',native,flush=True)

import gpuwrf.diagnostics.morrison_radar as M
from gpuwrf.runtime.aot_cheap_key import cheap_key
from gpuwrf.runtime.aot_baked_tables import baked_table_hash
cols=np.load(ROOT/'proofs/t_classd/morr_radar_oracle/fixtures/inputs.npz')['columns']
args=tuple(jnp.asarray(cols[:,i,:]) for i in range(9))
old=M.CONSTANTS
keys={}
for code in (10,40):
    selection=replace(nl,mp_physics=code,do_radar_ref=1,urban_bundle=None,sf_urban_physics=0,ra_lw_physics=0,ra_sw_physics=0,sf_sfclay_physics=0,bl_pbl_physics=0,cu_physics=0)
    off=replace(selection,do_radar_ref=0)
    # A fused cascade can close over the active namelist and receive an inactive root namelist.
    def factory(child):
        @jax.jit
        def kernel(*a):
            _=child.mp_physics
            return M.morrison_reflectivity(*a)
        return kernel
    radar=factory(selection);rows=[]
    for scale in (1.,1.01):
        c=dict(old);c['c']=dict(c['c']);s=dict(c['c']['s']);s['am']=np.float32(s['am']*scale);c['c']['s']=s;M.CONSTANTS=c
        radar.clear_cache()
        rows.append(dict(scale=scale,key=cheap_key(radar,args,{},selection),fused_key=cheap_key(radar,args,{},off),
                         stablehlo_sha256=hashlib.sha256(radar.lower(*args).as_text().encode()).hexdigest(),
                         values_sha256=hashlib.sha256(np.asarray(radar(*args)).tobytes()).hexdigest()))
    assert rows[0]['key'] and rows[0]['key']!=rows[1]['key']
    assert rows[0]['fused_key'] and rows[0]['fused_key']!=rows[1]['fused_key']
    assert rows[0]['stablehlo_sha256']!=rows[1]['stablehlo_sha256'] and rows[0]['values_sha256']!=rows[1]['values_sha256']
    keys[code]=rows
M.CONSTANTS=old
(OUT/'RV15_keys.json').write_text(json.dumps(keys,indent=2)+'\n')
print('KEYS direct+fused10/40 PASS',flush=True)

from gpuwrf.physics.nssl2mom import diagnostics as Dm
from gpuwrf.physics.nssl2mom.indices import FP32
rng=np.random.default_rng(87451)
bits=rng.integers(0x3f800000,0x7f000000,size=10000,dtype=np.uint32)
# Include exponents, near-one cancellation, and adjacent powers of ten.
special=np.r_[np.float32(1),np.nextafter(np.float32(1),np.float32(2)),np.logspace(0,38,200,dtype=np.float32)]
x=np.r_[bits.view(np.float32),special].astype(np.float32)
libm=ctypes.CDLL('libm.so.6');libm.log10f.argtypes=[ctypes.c_float];libm.log10f.restype=ctypes.c_float
ref=np.array([libm.log10f(float(v)) for v in x],np.float32)
log=jax.jit(lambda a:Dm._log10_real(a,FP32))
lower=log.lower(x);compiled=lower.compile();got=np.asarray(compiled(x));assert np.isfinite(got).all()
ulp=np.abs(got.view(np.int32).astype(np.int64)-ref.view(np.int32).astype(np.int64))
hlo=lower.compiler_ir('hlo').as_hlo_text()
assert 'f64' not in hlo
memory=compiled.memory_analysis()
result=dict(values=len(x),exact=int(np.count_nonzero(ulp==0)),max_ulp=int(ulp.max()),max_abs=float(np.max(np.abs(got-ref))),
            contains_f64=False,source='glibc log10f independent reference; extra-domain CPU sensitivity, frozen S6 BYTE gate is separate',
            memory={k:int(getattr(memory,k)) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')},
            optimized_hlo_chars=len(compiled.as_text()),cost_analysis=compiled.cost_analysis())
(OUT/'RV16_real_log.json').write_text(json.dumps(result,indent=2)+'\n')
print('REAL LOG',result,flush=True)
