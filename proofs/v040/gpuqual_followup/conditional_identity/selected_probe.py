"""Full selected native-model JAX program capture, CPU only, no forecast or CUDA compile."""
import gzip,hashlib,json,os,re,subprocess,sys,time,resource
from pathlib import Path
root,case,out=(Path(x).resolve() for x in sys.argv[1:4])
os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0,str(root/'src'))
from gpuwrf._fast_defaults import apply_fast_path_defaults
apply_fast_path_defaults()
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import operational_mode as om
from gpuwrf.contracts import state as state_contract
assert jax.devices()[0].platform=='cpu'
assert Path(om.__file__).resolve().is_relative_to(root)
out.parent.mkdir(parents=True,exist_ok=True)
t0=time.perf_counter()
# Exactly the accepted release-probe placement policy: CPU initialization, native facade only while tracing.
state_contract._gpu_device=lambda:jax.devices('cpu')[0]
facade.cuda_native=lambda:False
cfg=pipeline.NestedPipelineConfig(case,out.parent/'init_out',out.parent/'init_proof',hours=24,max_dom=1)
_,bundles,_,_,_,carries=pipeline._load_domains(cfg,('d01',))
nl,carry=bundles['d01'].namelist,carries['d01']
om._resolve_operational_suite(nl)
clock=om.build_clock_base(nl)
loaded=time.perf_counter()
def arrays(tree):
    result=[]
    for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a=np.asarray(value)
        assert a.dtype.kind!='O'
        result.append({'path':jax.tree_util.keystr(path),'dtype':a.dtype.str,'shape':list(a.shape),'weak_type':bool(getattr(value,'weak_type',False)),'sha256':hashlib.sha256(a.tobytes()).hexdigest()})
    return result
initial=arrays(carry)
actual=json.loads((case.parent/'gpu/proofs/nested_pipeline_run.json').read_text())['metadata']['domains']['d01']['namelist_resolved']
scalar_binding={}
for key,value in actual.items():
    if value is None or isinstance(value,(str,bool,int,float)):
        if hasattr(nl,key):
            current=getattr(nl,key)
            if current is None or isinstance(current,(str,bool,int,float,np.integer,np.floating)):
                scalar_binding[key]={'actual_measured':value,'probe':current.item() if hasattr(current,'item') else current,'equal':bool(current==value)}

facade.cuda_native=lambda:True
# Traced start/count/cadence retain all first-step/ordinary/radiation paths inside the real fori body.
program=jax.make_jaxpr(lambda c:om._advance_chunk_fori(c,nl,jnp.int32(1),clock,n_steps=jnp.int32(1),cadence=jnp.int32(nl.radiation_cadence_steps)))(carry)
raw=str(program)
canonical=re.sub(r'name_and_src_info=[^\]\n]*','',raw)
canonical=re.sub(r'/[^ \n:]*\.py:\d+(:\d+)?','',canonical)
with gzip.open(out.with_suffix('.jaxpr.txt.gz'),'wt') as f:f.write(raw)
with gzip.open(out.with_suffix('.canonical.txt.gz'),'wt') as f:f.write(canonical)
constants=arrays(program.consts)
rec={'source_rev':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source':str(root),'source_clean':not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(),'case':str(case),'platform':'cpu','GPU_executed':False,'CUDA_compiled':False,'native_facade_at_trace':True,'placement_override':'state._gpu_device -> CPU for trace; no mathematics replaced','actual_measured_scalar_bindings':scalar_binding,'initial_arrays':initial,'initial_pytree':str(jax.tree.structure(carry)),'constants':constants,'program_raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'program_canonical_sha256':hashlib.sha256(canonical.encode()).hexdigest(),'program_chars':len(raw),'load_s':loaded-t0,'trace_s':time.perf_counter()-loaded,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'namelist':{k:getattr(nl,k) for k in ('mp_physics','cu_physics','bl_pbl_physics','sf_sfclay_physics','ra_lw_physics','ra_sw_physics','sf_surface_physics','radiation_cadence_steps','cumulus_cadence_steps')},'resolved_GPUWRF_flags':{k:v for k,v in sorted(os.environ.items()) if k.startswith('GPUWRF_')},'scope':'Full selected _advance_chunk_fori native-model trace + all constant/input bytes; regression/identity evidence only, not WRF fidelity or new GPU-executable byte proof.'}
out.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:rec[k] for k in ('source_rev','namelist','program_canonical_sha256','program_chars','load_s','trace_s','host_peak_rss_kib')}),flush=True)
