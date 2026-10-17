"""Deviceless CUDA preflight of actual SFCDIF2/Jarvis/STC3 canopy caller."""
import hashlib,importlib.util,json,os,re,subprocess,sys,time,traceback
from pathlib import Path
tree,out=map(Path,sys.argv[1:3]);sys.path.insert(0,str(tree/'src'))
os.environ.update(GPUWRF_NOAHMP_NATIVE_REAL='1',GPUWRF_NOAHMP_COLUMN_KERNELS='1',GPUWRF_NOAHMP_LAYER_SELECT='1',GPUWRF_NOAHMP_LAYER_LISTS='1')
import jax
from gpuwrf.backends import pallas as facade
from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
from gpuwrf.io.noahmp_options import NoahMPOptions
from jax._src.pallas.triton import lowering
assert jax.devices()[0].platform=='cpu'
out.mkdir(parents=True,exist_ok=True);report=dict(platform='cpu',gpu_executed=False,source_commit=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip())
try:
    path=tree/'tests/v025/b_noahmp/test_snow_wrf_order_oracle.py'
    spec=importlib.util.spec_from_file_location('canopy_columns',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c=module._build(json.loads((path.parent/'fixtures/swiss_snow_wrf_order.json').read_text())['columns'][:4])
    facade.cuda_native=lambda:True
    fn=lambda land,forcing:noah_mp_step(land,forcing,c['static'],c['dt'],energy_params=c['ep'],rad_params=c['rp'],options=NoahMPOptions(opt_crs=2,opt_sfc=2,opt_stc=3))
    jp=jax.make_jaxpr(fn)(c['land'],c['forcing']).jaxpr
    calls=[e for e in jp.eqns if e.primitive.name=='pallas_call' and 'canopy_flux' in str(e.params.get('name'))]
    assert len(calls)==1
    p=calls[0].params;report['operand_packs']=[dict(shape=list(v.aval.shape),dtype=str(v.aval.dtype)) for v in calls[0].invars]
    start=time.monotonic();ir=str(lowering.lower_jaxpr_to_triton_module(p['jaxpr'],p['grid_mapping'],'cuda',120).module)
    report.update(lowering_s=time.monotonic()-start,ir_bytes=len(ir));path=out/'canopy.ttir';path.write_text(ir)
    import triton
    from triton.backends.compiler import GPUTarget
    start=time.monotonic();kernel=triton.compile(str(path),target=GPUTarget('cuda',120,32))
    report.update(compile_s=time.monotonic()-start,ptx_f64_ops=len(re.findall(r'^\s*(?:@\S+\s+)?[A-Za-z][^;\n]*\.f64\b',kernel.asm['ptx'],re.M)))
    cubin=out/'canopy.cubin';cubin.write_bytes(kernel.asm['cubin']);report['cubin_sha256']=hashlib.sha256(cubin.read_bytes()).hexdigest()
    scan=subprocess.run([sys.executable,str(tree/'scripts/e41/cubin_scan.py'),str(cubin),'unused','canopy'],cwd=out,capture_output=True,text=True,timeout=60)
    (out/'scan.log').write_text(scan.stdout+scan.stderr);report['resources']=json.loads((out/'cubin_scan_canopy.json').read_text())
    assert report['ptx_f64_ops']==0
    report.update(status='PASS',scope='native component static gate only; GPU runtime/D6 pending')
except Exception as error:report.update(status='FAIL',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()[-2500:])
finally:(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));sys.exit(0 if report['status']=='PASS' else 1)
