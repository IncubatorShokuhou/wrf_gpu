"""E41 on rich JIT records bound to the actually selected AOT execution payload.

CPU rehearsal substitutes an existing, hash-bound v043 GPU artifact and its
real run proof. It exercises the actual scanner, never a fabricated PASS.
The CUDA run selects its own newly produced mini/full-grid executable.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zstandard

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root',type=Path,required=True)
ap.add_argument('--version',required=True)
ap.add_argument('--arm',required=True)
ap.add_argument('--cpu-probe',type=int,required=True)
args=ap.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
base=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(base/'src'))
from gpuwrf.runtime import aot_executable as ax,aot_slim_module as asm
spec=importlib.util.spec_from_file_location('perf_e41',base/'scripts/wave_cubin_gate.py')
gate=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gate;spec.loader.exec_module(gate)
L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
version=args.version
if args.cpu_probe:
    version='v043'
    proof=L/'gpu/runs/N43/run/wrfout/proofs/nested_pipeline_run.json'
    cache_dirs=[Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/jax')]
else:
    proof=args.root/'runs'/args.arm/'run/wrfout/proofs/nested_pipeline_run.json'
    lower=L/'gpu/cache_v032/jax' if version=='v032' else Path('<DATA_ROOT>/wrf_gpu2/release_bundles')/('v0.'+('4.0' if version=='v040' else '4.3'))/'jax'
    cache_dirs=[args.root/('layer_'+version)/'upper/jax',lower]
programs=json.loads(proof.read_text())['metadata']['nested_aot']['domains']

def rest_without_hlo(blob):
    payload=gate._gpu_payload(blob,ax,asm)
    _,info,pos=asm._chunks(payload)[-1]
    data,_=asm._read_logical(payload,pos,info['data_size'])
    rest,=asm._decode_simple_chunk(data,1)
    return [f for f in ax._proto_fields(rest) if f[0]!=1]

selected=[];bindings=[];seen=set();donors=[]
for cache in cache_dirs:
    for p in sorted(cache.glob('jit__advance_chunk*-cache'))+sorted(cache.glob('jit_fused_jit*-cache')):
        if p.name in seen:continue
        seen.add(p.name);donors.append(p)
for name,meta in programs.items():
    key=meta['cheap_key'];actual=None
    for cache in cache_dirs:
        found=list((cache/'aot').glob('*/'+name+'/k_'+key+'.xlaexec'))
        if found:assert len(found)==1;actual=found[0];break
    assert actual is not None,(name,key,cache_dirs)
    blob=actual.read_bytes();records=[gate._record(blob,i,ax,asm) for i in (0,1,2)]
    match=None
    for donor in donors:
        rich=zstandard.ZstdDecompressor().decompress(donor.read_bytes(),max_output_size=1<<33)[4:]
        if gate._record(rich,2,ax,asm)!=records[2]:continue
        assert all(gate._record(rich,i,ax,asm)==records[i] for i in (0,1,2))
        assert rest_without_hlo(rich)==rest_without_hlo(blob)
        match=donor;break
    assert match is not None,('no exact rich execution donor',name)
    selected.append(gate.Executable(match,'jax_cache',name))
    bindings.append(dict(program=name,cheap_key=key,aot_path=str(actual),aot_sha256=hashlib.sha256(blob).hexdigest(),
                         rich_path=str(match),rich_sha256=hashlib.sha256(match.read_bytes()).hexdigest(),full_non_hlo_payload_equal=True))
out=args.root/'e41'/args.version
out.mkdir(parents=True,exist_ok=True)
# The scanner and all gates/coverage are unmodified; discovery is restricted to
# exact execution-bound donors instead of unrelated dates/compact aliases.
gate.discover=lambda cache,pattern:selected
report=gate.run_gate(cache_dirs[0],out,run_proof=proof)
report.update(cpu_probe=bool(args.cpu_probe),artifact_version=version,requested_version=args.version,
              artifact_scope='existing GPU donor; CPU launcher rehearsal only' if args.cpu_probe else 'actual selected production/mini program',
              execution_bindings=bindings,proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())
(out/'selected_gate.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'e41_gate':report['e41_gate'],'coverage':report['coverage'],'cpu_probe':bool(args.cpu_probe)}))
raise SystemExit(0 if report['e41_gate']=='PASS' else 2)
