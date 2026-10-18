"""Resolve stack_frame_id -> innermost gpuwrf file:line for the E41-flagged fusions (rich JIT HloModuleProto)."""
import collections,json,re,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/scripts');sys.path.insert(0,sys.argv[1]+'/src')
import zstandard,wave_cubin_gate as W
from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
blob=zstandard.ZstdDecompressor().decompress(Path(sys.argv[2]).read_bytes(),max_output_size=1<<33)[4:]
gpu=W._gpu_payload(blob,ax,asm)
begin,info,data_pos=asm._chunks(gpu)[-1];data,_=asm._read_logical(gpu,data_pos,info['data_size']);(gpu,)=asm._decode_simple_chunk(data,info['num_records'])
hwc=next(v for f,w,v in ax._proto_fields(gpu) if f==1);proto=next(v for f,w,v in ax._proto_fields(hwc) if f==1)
fields=collections.defaultdict(list)
for f,w,v in ax._proto_fields(proto):fields[f].append((w,v))
print('module fields',{f:len(v) for f,v in sorted(fields.items())})
idx=None
for f,vals in fields.items():
    for w,v in vals:
        if isinstance(v,(bytes,bytearray)) and b'.py' in v[:200000] and b'gpuwrf' in v:
            try:
                sub=list(ax._proto_fields(v))
                if any(sf==1 for sf,_,_ in sub) and any(sf==4 for sf,_,_ in sub):idx=(f,v)
            except Exception:pass
f,v=idx;print('stack_frame_index field',f)
names=[];funcs=[];locs=[];frames=[]
for sf,w,val in ax._proto_fields(v):
    if sf==1:names.append(bytes(val).decode())
    elif sf==2:funcs.append(bytes(val).decode())
    elif sf==3:locs.append({k:x for k,_,x in ax._proto_fields(val)})
    elif sf==4:frames.append({k:x for k,_,x in ax._proto_fields(val)})
print(len(names),len(funcs),len(locs),len(frames))
def chain(fid):
    out=[];seen=0
    while fid and seen<200:
        fr=frames[fid-1];loc=locs[fr.get(1,0)-1] if fr.get(1) else {}
        out.append((names[loc.get(1,0)-1] if loc.get(1) else '?',loc.get(3,0),funcs[loc.get(2,0)-1] if loc.get(2) else '?'))
        fid=fr.get(2,0);seen+=1
    return out
text=Path(sys.argv[3]).read_text();blocks={};cur=None
for line in text.split('\n'):
    m=re.match(r'%?([\w.\-]+) \(',line)
    if m and not line.startswith(' '):cur=m.group(1);blocks[cur]=[]
    elif cur:blocks[cur].append(line)
res={}
for comp in json.loads(sys.argv[4]):
    c=collections.Counter()
    for line in blocks.get(comp,[]):
        m=re.search(r'stack_frame_id=(\d+)',line)
        if not m:continue
        ch=chain(int(m.group(1)))
        inner=next(((fn.split('/src/gpuwrf/')[-1],ln,fu) for fn,ln,fu in ch if '/gpuwrf/' in fn),None)
        c[inner]+=1
    tot=sum(c.values());files=collections.Counter()
    for k,n in c.items():files[k[0] if k else None]+=n
    res[comp]=dict(instr_with_frames=tot,files=files.most_common(6),top=[(k,n) for k,n in c.most_common(5)])
    print(comp,tot,files.most_common(4))
Path(sys.argv[5]).write_text(json.dumps(res,indent=1,default=str)+'\n')
