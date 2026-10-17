"""CPU-only CUDA/sm120 lowering and static cubin prescreen for optional land phases."""
from functools import partial
import hashlib,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
TREE,OUT=map(Path,sys.argv[1:3]);family=sys.argv[3]
sys.path[:0]=[str(TREE/'src'),str(TREE/'tests')]
os.environ.update(GPUWRF_NOAHMP_NATIVE_REAL='1',GPUWRF_NOAHMP_LAYER_SELECT='1',GPUWRF_NOAHMP_LAYER_LISTS='1')
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.backends import pallas as facade
from gpuwrf.physics.noahmp.land_columns import layered_column_call
from jax._src.pallas.triton import lowering
assert jax.devices()[0].platform=='cpu'
OUT.mkdir(parents=True,exist_ok=True)
report=dict(platform='cpu',gpu_executed=False,target='cuda/sm120',family=family,
    source_commit=subprocess.check_output(['git','-C',str(TREE),'rev-parse','HEAD'],text=True).strip())


def water_inputs():
    from gpuwrf.physics.noahmp.groundwater_options import GroundwaterState,_equilibrium_soilwater,_prognostic_soilwater
    from gpuwrf.physics.noahmp.runoff_options import RunoffParameters,HydrologyParameters,_advanced_soilwater
    from gpuwrf.physics.noahmp.tile_drain import TileParameters,equivalent_depth
    name='equilibrium_oracle.json' if family=='run2' else 'prognostic_water_oracle.json' if family=='run1' else 'advanced_water_oracle.json'
    fx=json.loads((TREE/'proofs/u2_land'/name).read_text())
    p={k.lower():v for k,v in fx['parameters'].items()}
    run=1 if family=='run1' else 2 if family=='run2' else 8 if family=='run8' else 3
    td=1 if family=='td1' else 2 if family=='td2' else 0
    cases=[c for c in fx['cases'] if c['opt_run']==run and c['opt_inf']==2 and (not td or c.get('opt_tdrn')==td)
        and (run!=8 or c['opt_infdv']==3)][:8]
    n=len(cases);assert n==8
    row=lambda k:jnp.asarray([[c[k] for c in cases]],jnp.float32)
    profile=lambda k:jnp.asarray([c[k] for c in cases],jnp.float32).T[:,None,:]
    par=lambda k:jnp.asarray(p[k],jnp.float32).reshape(4,1,1) if isinstance(p[k],list) else jnp.asarray(p[k],jnp.float32)
    dt,qin,qseva=row('dt'),row('qinsur'),row('qseva')
    sh,sm,et=(profile(k) for k in ('sh2o','smc','etrani'))
    ice=jnp.maximum(0.,sm-sh);z=jnp.asarray([-.1,-.4,-1.,-2.],jnp.float32)[:,None,None]
    dz=jnp.asarray([.1,.3,.6,1.],jnp.float32)[:,None,None]
    urban=jnp.zeros((1,n),bool)
    head=(qin,qseva,et,sh,sm,ice,z,dz)
    if run in (1,2):
        args=(*head,*(par(k) for k in ('bexp','smcmax','smcwlt','dksat','dwsat','psisat')),dt,par('fsatmx'),par('timean'))
        if run==1:args=(*args,GroundwaterState(*(jnp.full((1,n),v,jnp.float32) for v in (2.5,4900.,4900.))))
        return partial(_prognostic_soilwater if run==1 else _equilibrium_soilwater,opt_inf=2),(*args,urban),(1,n)
    rp=RunoffParameters(*(par(k) for k in RunoffParameters._fields)) if run==8 else None
    tp=None
    if td:
        depth=equivalent_depth(par('td_adepth')-par('td_ddrain'),par('td_spac'),par('td_radi'))
        tp=TileParameters(*(par(k) for k in ('tdsmc_fac','td_dc','td_depth','td_dcoef','klat_fac','td_adepth','td_ddrain','td_spac','td_radi')),depth)
    hp=HydrologyParameters(rp,tp,jnp.ones((1,n),jnp.float32) if td else None,jnp.full((1,n),3000.,jnp.float32) if td else None)
    args=(*head,*(par(k) for k in ('bexp','smcmax','smcref','smcwlt','dksat','dwsat','kdt','frzx','slope')),dt,hp,urban)
    return partial(_advanced_soilwater,opt_run=run,opt_inf=2,opt_infdv=3 if run==8 else 0,opt_tdrn=td,drain_layer_opt=4),args,(1,n)


def carbon_inputs():
    from gpuwrf.physics.noahmp.carbon import CarbonState,CarbonParameters,carbon_column
    from gpuwrf.physics.noahmp.crop import CropCarbonState,CropParameters,CropStaticParameters,crop_column
    from gpuwrf.physics.noahmp.tables import _parse_mptable_block
    from gpuwrf.config.paths import wrf_run_dir
    fx=json.loads((TREE/'proofs/u2_land'/('crop_oracle.json' if family=='crop' else 'carbon_oracle.json')).read_text())
    cases=fx['cases'][:8];n=len(cases)
    row=lambda k:jnp.asarray([[c[k] for c in cases]],jnp.float32)
    states=jnp.asarray([c['state'] for c in cases],jnp.float32).T[:,None,:]
    if family=='crop':
        table=_parse_mptable_block((wrf_run_dir()/'MPTABLE.TBL').read_text(),'noahmp_modis_parameters')
        natural=CarbonParameters(*(jnp.full((1,n),table[k.upper()][11],jnp.float32) for k in CarbonParameters._fields))
        crop=CropParameters(*(jnp.asarray([[c['parameters'][k] for c in cases]],jnp.float32) for k in fx['scalar_keys']),
            *(jnp.asarray([c['parameters'][k] for c in cases],jnp.float32).T[:,None,:] for k in fx['array_keys']))
        params=CropStaticParameters(natural,crop,jnp.ones((1,n),jnp.int32))
        state=CropCarbonState(*states,jnp.zeros((1,n),jnp.int32));veg=jnp.full((1,n),12,jnp.int32)
    else:
        params=CarbonParameters(*(jnp.asarray([[c['parameters'][k] for c in cases]],jnp.float32) for k in fx['parameter_keys']))
        state=CarbonState(*states);veg=jnp.asarray([[c['veg'] for c in cases]],jnp.int32)
    soil=jnp.full((4,1,n),.2,jnp.float32);smx=jnp.full((4,1,n),.45,jnp.float32)
    dz=jnp.asarray([.1,.3,.6,1.],jnp.float32)[:,None,None];z=jnp.asarray([-.1,-.4,-1.,-2.],jnp.float32)[:,None,None]
    args=(state,params,row('lai'),row('sai'),row('psn'),row('btran'),row('foln'),row('tv'),row('tv'),soil,smx,dz,z,
          jnp.full((1,n),3,jnp.int32),veg,jnp.ones((1,n),jnp.float32),row('dt'))
    if family=='crop':return partial(crop_column,dveg=5),(*args,row('t2m'),row('julian')),(1,n)
    return carbon_column,args,(1,n)


try:
    fn,args,shape=carbon_inputs() if family in ('carbon','crop') else water_inputs()
    facade.cuda_native=lambda:True
    call=lambda *values:layered_column_call(fn,values,grid_shape=shape,name='b_core_noah_'+family)
    start=time.monotonic();jp=jax.make_jaxpr(call)(*args).jaxpr
    eqns=[e for e in jp.eqns if e.primitive.name=='pallas_call'];assert len(eqns)==1
    p=eqns[0].params
    lowered=lowering.lower_jaxpr_to_triton_module(p['jaxpr'],p['grid_mapping'],'cuda',120)
    ir=str(lowered.module);path=OUT/(family+'.ttir');path.write_text(ir)
    report.update(native_calls=1,lowering_s=time.monotonic()-start,ir_bytes=len(ir))
    import triton
    from triton.backends.compiler import GPUTarget
    start=time.monotonic();kernel=triton.compile(str(path),target=GPUTarget('cuda',120,32))
    ptx=kernel.asm['ptx'];f64=len(re.findall(r'^\s*(?:@\S+\s+)?[A-Za-z][^;\n]*\.f64\b',ptx,re.M))
    report.update(compile_s=time.monotonic()-start,ptx_f64_ops=f64,wrf_double_island=family=='run1')
    assert family=='run1' or f64==0
    cubin=OUT/(family+'.cubin');cubin.write_bytes(kernel.asm['cubin'])
    report['cubin_sha256']=hashlib.sha256(cubin.read_bytes()).hexdigest()
    scan=subprocess.run([sys.executable,str(TREE/'scripts/e41/cubin_scan.py'),str(cubin),'unused',family],cwd=OUT,capture_output=True,text=True,timeout=60)
    (OUT/'scan.log').write_text(scan.stdout+scan.stderr);report['native_scanner_rc']=scan.returncode
    report.update(status='PASS',scope='CPU deviceless component only; opt-in E41 resource flags disclosed, GPU runtime/D6 pending')
except Exception as error:
    report.update(status='FAIL',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()[-4000:])
finally:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if report['status']!='PASS':sys.exit(1)
