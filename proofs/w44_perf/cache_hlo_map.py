"""Bind rich cache HLO to actual native records, without GPU access.

Only matched PROD programs are exported. Native records 0/1/2 are checked
against sealed AOT identities; debug-only retained HLO can differ.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu' and os.environ['GPUWRF_JAX_CACHE']=='0'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from gpuwrf.runtime import aot_executable as ax,aot_slim_module as asm
import zstandard
spec=importlib.util.spec_from_file_location('gate',ROOT/'scripts/wave_cubin_gate.py');gate=importlib.util.module_from_spec(spec);sys.modules['gate']=gate;spec.loader.exec_module(gate)
sha=lambda b:hashlib.sha256(b).hexdigest()
ids=json.loads(Path('<USER_HOME>/wrf_gpu2_lanes/w43-release/followup/native_ids_v043.json').read_text())
targets={}
for program,prefix in [('d01','6cc0486b3dea'),('d02','4e8d00fa090d')]:
 entry=next(x for x in ids['programs'][program] if any(name.startswith(prefix) for name in x['aliases']))
 targets[entry['native_record2_sha256']]=(program,entry)
a.out.mkdir(parents=True,exist_ok=False)
records=[]
sets=[('v032',Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/gpu/cache_v032/jax')),
      ('v043',Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/jax'))]
for version,cache in sets:
 for path in sorted(cache.glob('jit__advance_chunk_fori-*-cache')):
  blob=zstandard.ZstdDecompressor().decompress(path.read_bytes(),max_output_size=1<<33)[4:]
  native=gate._record(blob,2,ax,asm);native_sha=sha(native)
  if version=='v043' and native_sha not in targets:continue
  identity={}
  if version=='v043':
   program,expected=targets[native_sha]
   for index in (0,1,2):
    actual=sha(gate._record(blob,index,ax,asm));wanted=next(x['sha256'] for x in expected['records'] if x['index']==index)
    identity[str(index)]={'equal':actual==wanted,'actual':actual,'expected':wanted}
   assert all(x['equal'] for x in identity.values()),'native/binding records differ'
  else:
   # Gate inventory binds each real old JIT to the program's native ELF.
   g=json.loads(Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/gpu/E41_v032/wave_gate.json').read_text())
   entry=next(x for x in g['entries'] if x['entry']==path.name)
   count=sum(x['kernels'] for x in entry['elfs']);program={804:'d01',641:'d02'}[count]
  text=gate._optimized_hlo(blob,ax,asm)
  dest=a.out/(version+'_'+program+'.hlo')
  dest.write_text(text)
  rec={'version':version,'program':program,'cache_entry':str(path),'native_record2_sha256':native_sha,
       'hlo_file':str(dest),'hlo_sha256':sha(text.encode()),'native_binding_identity':identity}
  records.append(rec);print(version,program,len(text),native_sha[:16],flush=True)
  del blob,native,text
(a.out/'INDEX.json').write_text(json.dumps({'scope':__doc__,'records':records},indent=2)+'\n')
assert len(records)==4
