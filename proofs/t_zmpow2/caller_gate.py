"""Two resident calls on sealed Swiss WRF caller inputs; native deletion gates.

This is CPU interpretation regression evidence, not WRF or GPU qualification.
Existing pristine-WRF qualification is reused for byte-identical results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--scheme',type=int,choices=(7,10,11),required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--target',type=Path)
p.add_argument('--mode',choices=('byte','lower','mutant'),default='byte')
p.add_argument('--active',action='store_true',help='supplement real Swiss bank with active original-WRF controlled columns')
a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert subprocess.run(['git','-C',str(a.source),'symbolic-ref','--quiet','HEAD'],capture_output=True).returncode!=0
assert not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain','--untracked-files=no'])
sys.path.insert(0,str(a.source/'src'))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.kernels import phys_cuplus_driver as current
from gpuwrf.kernels import phys_camzm_column as camzm
from gpuwrf.kernels import phys_kfcup_column as cup,phys_kfcup_cps as cps,phys_mskf_column as mskf
from gpuwrf.physics import cumulus_kfcup_support as support

baseline_commit='09f43ebf15dda995820feef872707c19c390a215'
baseline_text=subprocess.check_output(['git','-C',str(a.source),'show',
    baseline_commit+':src/gpuwrf/kernels/phys_cuplus_driver.py'],text=True)
old=types.ModuleType('baseline_cuplus')
exec(compile(baseline_text,'baseline_cuplus.py','exec'),old.__dict__)
baseline_camzm=subprocess.check_output(['git','-C',str(a.source),'show',
    baseline_commit+':src/gpuwrf/kernels/phys_camzm_column.py'],text=True)
old_camzm=types.ModuleType('baseline_camzm')
exec(compile(baseline_camzm,'baseline_camzm.py','exec'),old_camzm.__dict__)
baseline_helpers={}
def old_module(name,folder='kernels'):
    text=subprocess.check_output(['git','-C',str(a.source),'show',
        baseline_commit+':src/gpuwrf/'+folder+'/'+name+'.py'],text=True)
    baseline_helpers[name]=hashlib.sha256(text.encode()).hexdigest()
    result=types.ModuleType('baseline_'+name)
    exec(compile(text,'baseline_'+name+'.py','exec'),result.__dict__)
    return result
old_cup=old_module('phys_kfcup_column')
old_support=old_module('cumulus_kfcup_support',folder='physics')
saved=cup._cup_column
saved_support=support.cup_adjust_mass_flux
try:
    cup._cup_column=old_cup._cup_column
    support.cup_adjust_mass_flux=old_support.cup_adjust_mass_flux
    old_cps=old_module('phys_kfcup_cps')
finally:
    cup._cup_column=saved
    support.cup_adjust_mass_flux=saved_support
old_mskf=old_module('phys_mskf_column')
fixture=a.source/'proofs/v040/cuplus/full_callers'
manifest=json.loads((fixture/'manifest.json').read_text())
assert hashlib.sha256((fixture/'full_callers.npz').read_bytes()).hexdigest()==manifest['fixture_sha256']
group={7:'zm_real',10:'cup_real',11:'mskf_real'}[a.scheme]
if a.active:group={7:'zm_vector',10:'cup',11:'mskf_vector'}[a.scheme]
rows=[r for r in manifest['rows'] if r['group']==group and r['step']==1 and (a.scheme!=7 or r['pbl']==9)]
if a.active:rows=[r for r in rows if r['case'] in (1,3)]
record=dict(source_commit=subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip(),
    baseline_commit=baseline_commit,baseline_sha256=hashlib.sha256(baseline_text.encode()).hexdigest(),
    baseline_camzm_sha256=hashlib.sha256(baseline_camzm.encode()).hexdigest(),
    baseline_helper_sha256=baseline_helpers,
    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixture_sha256=manifest['fixture_sha256'],
    scheme=a.scheme,mode=a.mode,physical_backend='cpu',gpu_executed=False,
    input_scope='active pristine-controlled columns' if a.active else 'sealed Swiss original-WRF forecast caller inputs',
    cache='NONE GPUWRF_JAX_CACHE=0',rows=[],scope=__doc__)
start=time.monotonic()
def save():
    record.update(wall_s=time.monotonic()-start,peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    a.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
save()

with np.load(fixture/'full_callers.npz',allow_pickle=False) as bank:
    cases=[(r,{key:np.array(bank[r['key']+'_'+key]) for key in r['profiles']+r['scalars']}) for r in rows]

def operands(raw):
    if a.scheme==7:
        names=('T_PHY','QV','P','P8W','PI_PHY','U_PHY','V_PHY','Z','Z_AT_W','TKE_PBL','TH','QC','QI','CLDFRA','DZ8W')
        arrays=tuple(jnp.asarray(raw['IN_'+key],jnp.float32) for key in names)
        names=('DT','DX','CUDT','ITIMESTEP','PSFC','HT','PBLH','KPBL','HFX','QFX','UST','MAVAIL','XLAND','TSK','TPERT_CAMUWPBL')
        if a.active:
            ctl=jnp.asarray([54.,9000.,0.,1.,100800. if raw['CASE']==3 else 100000.,0.,1500.,6.,150.,1e-4,.3,1.,1.,arrays[0][0],.3],jnp.float32)
        else:ctl=jnp.asarray([raw['CTRL_'+key] for key in names],jnp.float32)
        w=jnp.zeros(len(arrays[0])+1,jnp.float32);w0=jnp.zeros(len(arrays[0]),jnp.float32)
    else:
        names=('T','QV','P','DZ','RHO','W0AVG','U','V')+(('TH','Z','ZW','PI') if a.scheme==10 else ('PI',))
        arrays=tuple(jnp.asarray(raw[key],jnp.float32) for key in names)
        names=('DT','DX','PBLH','KPBL','HFX','QFX','XLAND','HT') if a.scheme==10 else ('DT','DX','PBLH','HFX','UST','ZOL')
        if a.active:
            hfx=next(r['controls'][0] for r,v in cases if v is raw) if a.scheme==10 else 150.
            ctl=jnp.asarray([raw['DT'],raw['DX'],1500.,6.,float(hfx),1e-4,1.,0.] if a.scheme==10 else
                            [raw['DT'],raw['DX'],1000.,150.,.3,-.1,0.,0.],jnp.float32)
        else:ctl=jnp.asarray([raw['CTRL_'+key] for key in names]+([] if a.scheme==10 else [0.,0.]),jnp.float32)
        # CuP fixtures retain exact Swiss W interfaces; MSKF has the same
        # thermodynamic bank but its historical dump omits W. Reuse the paired
        # real CuP case's exact W, instead of a synthetic zero profile.
        match=raw
        if a.scheme==11:
            index=next(i for i,(r,_) in enumerate(cases) if _ is raw)
            cup=next(r for r in manifest['rows'] if r['group']==('cup' if a.active else 'cup_real') and r['step']==1 and r['case']==cases[index][0]['case'])
            with np.load(fixture/'full_callers.npz',allow_pickle=False) as bank:
                match={'IN_W':np.array(bank[cup['key']+'_IN_W'])}
        w=jnp.asarray(match['IN_W'],jnp.float32)
        w0=jnp.zeros(len(arrays[0]),jnp.float32)
    return arrays,w,w0,old.initial_column(a.scheme,len(arrays[0])),ctl,jnp.int32(1)

def call(module):
    def fn(arrays,w,w0,held,ctl,step):
        if a.scheme==7:ctl=ctl.at[3].set(step.astype(jnp.float32))
        helper=camzm._camzm_cps
        cup_helper,mskf_helper=cps._cps_column,mskf._mskf_column
        try:
            # Bind the complete ORIGINAL CAM-ZM body while tracing the baseline,
            # including original control accesses and literal branch results.
            if module is old:
                camzm._camzm_cps=old_camzm._camzm_cps
                cps._cps_column=old_cps._cps_column
                mskf._mskf_column=old_mskf._mskf_column
            return module.column_step(arrays,w,w0,held,ctl,step,scheme=a.scheme,stepcu=1,cudt=0.,pbl=9,
                num_bins=5,th_bin_size=.1,r_bin_size=1e-4,min_deep_freq=.1,min_shallow_freq=.001,
                interpret=a.mode=='byte')
        finally:
            camzm._camzm_cps=helper
            cps._cps_column,mskf._mskf_column=cup_helper,mskf_helper
    return jax.jit(fn)

if a.mode=='byte':
    fns=[call(old),call(current)]
    for row,raw in cases:
        args=operands(raw)
        before=[args,args]
        for step in (1,2):
            results=[]
            for i,fn in enumerate(fns):
                t=time.monotonic()
                result=jax.block_until_ready(fn(*before[i][:-1],jnp.int32(step)))
                results.append(result)
                # Original post-RK countdown/clear is part of this resident ABI.
                held,_=current.finish_column(result[1],args[4][0],scheme=a.scheme)
                before[i]=(args[0],args[1],result[0],held,args[4],jnp.int32(step+1))
            assert jax.tree.structure(results[0])==jax.tree.structure(results[1])
            count=0;nonzero=0;hashes=[]
            for x,y in zip(jax.tree.leaves(results[0]),jax.tree.leaves(results[1]),strict=True):
                x,y=np.asarray(x),np.asarray(y)
                assert np.isfinite(x).all() and np.isfinite(y).all(),'nonfinite candidate/reference'
                assert (x.shape,x.dtype,x.tobytes())==(y.shape,y.dtype,y.tobytes()),'resident byte regression'
                count+=x.size;nonzero+=np.count_nonzero(y);hashes.append(hashlib.sha256(y.tobytes()).hexdigest())
            record['rows'].append(dict(case=row['case'],step=step,byte_values=count,nonzero_values=int(nonzero),
                active_heat=bool(np.any(np.asarray(results[1][1]['RTHCUTEN'])!=0)),
                active_momentum=bool(np.any(np.asarray(results[1][1].get('DUDT',results[1][1].get('RUCUTEN',0)))!=0)),
                output_sha256=hashes))
            save();print(a.scheme,row['case'],step,'BYTE',count,flush=True)
    if a.active:
        assert any(r['active_heat'] for r in record['rows']),'active control missed heat path'
        if a.scheme in (7,11):assert any(r['active_momentum'] for r in record['rows']),'active control missed momentum path'
    record['memory']={}
    for name,fn in zip(('baseline','candidate'),fns,strict=True):
        t=time.monotonic();exe=fn.lower(*args).compile();mem=exe.memory_analysis()
        record['memory'][name]={k:getattr(mem,k) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        record['memory'][name]['cached_lower_compile_s']=time.monotonic()-t
else:
    assert a.target is not None
    from gpuwrf.backends import platform as capability
    from jax.experimental import topologies
    capability.runtime_backend=lambda:'cuda'
    assert current.pl.cuda_native()
    topology=topologies.get_topology_desc(platform='cuda',target_config=a.target.read_text(),topology='1x1x1')
    sharding=jax.sharding.SingleDeviceSharding(topology.devices[0])
    args=operands(cases[0][1])
    def abstract(v):
        return jax.ShapeDtypeStruct(v.shape,v.dtype,weak_type=bool(getattr(v,'weak_type',False)),
            sharding=sharding if getattr(v,'committed',False) else None)
    fn=call(old if a.mode=='mutant' else current)
    try:
        lowered=fn.trace(*jax.tree.map(abstract,args)).lower(lowering_platforms=('cuda',))
        raw=lowered.as_text().encode()
        assert a.mode!='mutant','deletion mutant survived native lowering'
        record.update(lower='PASS',stablehlo_bytes=len(raw),stablehlo_sha256=hashlib.sha256(raw).hexdigest())
    except (ValueError,NotImplementedError) as exc:
        expected='power of 2' if a.scheme==7 else 'Unimplemented primitive in Pallas GPU lowering: slice'
        assert a.mode=='mutant' and expected in str(exc),str(exc)
        record.update(mutant='KILLED',exception=repr(exc),expected_failure=expected)
record['complete']=True
save()
print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)
