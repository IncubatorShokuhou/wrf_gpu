"""Read-only rich-cache fusion comparison by source and normalized DAG (not names)."""
from pathlib import Path
import collections,hashlib,importlib.util,json,os,re,sys,time
D=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc');SRC=D/'snapshot';OUT=D/'e41_diagnostic';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(SRC/'src'),str(SRC/'scripts'),str(SRC/'proofs/w44_e84fix')]
import zstandard
import jaxlib._jax as J
import wave_cubin_gate as W
from gpuwrf.runtime import aot_executable as ax,aot_slim_module as asm
from e41 import hlo_scan
from lowering_audit import debug_from_proto,paths
assert os.environ['JAX_PLATFORMS']=='cpu'
sha=lambda b:hashlib.sha256(b).hexdigest()

def load(path,kind):
 blob=path.read_bytes()
 if kind=='jax_cache':blob=zstandard.ZstdDecompressor().decompress(blob,max_output_size=1<<33)[4:]
 gpu=W._gpu_payload(blob,ax,asm)
 if ax._gpu_blob_format(blob)==ax.GPU_THUNK_FORMAT:
  begin,info,data_pos=asm._chunks(gpu)[-1]
  data,_=asm._read_logical(gpu,data_pos,info['data_size'])
  assert info['num_records']==1
  (gpu,)=asm._decode_simple_chunk(data,info['num_records'])
 hwc=next(v for f,w,v in ax._proto_fields(gpu) if f==1)
 proto=next(v for f,w,v in ax._proto_fields(hwc) if f==1)
 text=J.HloModule.from_serialized_hlo_module_proto(proto).to_string()
 if re.search(r'%?slim\.(?:zero|convert|out|root)',text):return dict(path=str(path),kind=kind,status='COMPACT_UNCOVERED',proto_sha256=sha(proto))
 scan=hlo_scan.scan_text(text)
 debug=debug_from_proto(proto)
 literals={}
 for f,w,c in ax._proto_fields(proto):
  if f!=3 or w!=2:continue
  name=next(v.decode() for k,ww,v in ax._proto_fields(c) if k==1)
  for k,ww,ins in ax._proto_fields(c):
   if k!=2 or ww!=2:continue
   values={kk:vv for kk,www,vv in ax._proto_fields(ins)}
   if 8 in values:literals[(name,values[1].decode())]=sha(values[8])
 # Parse all computations once; use original-protobuf metadata by instruction name.
 comps={};current=None
 for line in text.splitlines():
  if current is None:
   m=hlo_scan.HEADER.match(line)
   if m and not line.startswith('HloModule'):current=m[1];comps[current]=[]
  elif line.strip()=='}':current=None
  elif m:=hlo_scan.INSTR.match(line):comps[current].append((m[1],m[2]))
 fingerprints={}
 def fingerprint(comp,seen=()):
  if comp in fingerprints:return fingerprints[comp]
  assert comp not in seen,('recursive computation',comp)
  local={n:'n'+str(i) for i,(n,_) in enumerate(comps[comp])}
  normalized=[]
  for name,rhs in comps[comp]:
   rhs=rhs.split(', metadata=',1)[0]
   if (comp,name) in literals:rhs=re.sub(r'constant\(.*\)', 'constant(LITERAL_SHA256='+literals[(comp,name)]+')',rhs,count=1)
   def rename(m):
    n=m[1]
    if n in local:return '%'+local[n]
    if n in comps:return '@'+fingerprint(n,seen+(comp,))
    return '%'+n
   rhs=re.sub(r'%([\w.\-]+)',rename,rhs)
   normalized.append(local[name]+'='+rhs)
  result=sha('\n'.join(normalized).encode());fingerprints[comp]=result;return result
 rows=[]
 for fusion in scan['transpose_hero_fusions']:
  if fusion['kind']!='kInput':continue
  comp=fusion['comp'];sites=collections.Counter()
  for name,_ in comps[comp]:
   frames=paths(debug,debug['metadata'].get((comp,name),{}))
   if frames:sites[(frames[0]['file'],frames[0]['function'])]+=1
  row=dict(fusion,dag_sha256=fingerprint(comp),source_functions=[dict(file=k[0],function=k[1],instructions=v) for k,v in sorted(sites.items())])
  rows.append(row)
 return dict(path=str(path),kind=kind,status='RICH',proto_sha256=sha(proto),full_blob_sha256=sha(blob),
             n_fusions=scan['n_fusions'],kinput_fusions=rows)

owner=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-lsmparams/gpu/v11/lsm/e41/wave_gate.json')
owner_gate=json.loads(owner.read_text());target_elf='04c8e702b9aa72bf65e96dbb5d3a4936b8be782b21033f7b462fcf36756c3d51'
entry=next(e for e in owner_gate['entries'] if e.get('kinput_transpose_over500') and any(x['sha256']==target_elf for x in e['elfs']))
if Path(entry['path']).exists():
 origin=load(Path(entry['path']),entry['source_kind']);assert origin['status']=='RICH',origin
 wanted=[f for f in origin['kinput_fusions'] if f['n_instr']==664]
 assert wanted,'original reported 664-body absent from its exact rich executable'
else:
 origin=dict(path=entry['path'],status='MISSING_ORIGIN_RICH_ARTIFACT',ELF_sha256=target_elf)
 wanted=[]
(OUT/'ORIGIN_700a.json').write_text(json.dumps(origin,indent=2)+'\n')
result=dict(authority='manager MAIN 2026-10-26T01:58:22Z',origin_wave_gate_sha256=sha(owner.read_bytes()),
            origin=origin,targets=wanted,v043=[],matches=[],GPU_executed=False)
cache=Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/jax')
# JIT entries contain the rich module; compact aliases cannot prove HLO absence.
entries=[e for e in W.discover(cache) if e.kind=='jax_cache']
assert entries,'no released rich entries; cannot make a baseline classification'
for i,e in enumerate(entries):
 print('READ',i+1,'/',len(entries),e.path.name,flush=True)
 rec=load(e.path,e.kind)
 result['v043'].append(rec)
 if rec['status']=='RICH':
  for f in rec['kinput_fusions']:
   for target in wanted:
    if f['dag_sha256']==target['dag_sha256'] and f['source_functions']:
     result['matches'].append(dict(path=str(e.path),fusion=f,origin_fusion=target))
 (OUT/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
result['classification']=('ORIGIN_DAG_UNAVAILABLE' if not wanted else 'PRE_EXISTING_IN_V043_RELEASE_BUNDLE' if result['matches'] else 'NO_EXACT_DAG_MATCH_IN_RELEASED_RICH_PROGRAMS')
result['absence_limit']='No match does not prove a source regression: cold compilation/fusion partition and program operands can differ. Compact programs remain uncovered.'
(OUT/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['classification'],'matches',len(result['matches']),'released rich entries',len(entries),flush=True)
