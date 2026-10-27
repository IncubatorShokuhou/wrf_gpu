"""Read exact compiled thunk-to-HLO bindings; no name transfer between versions.

Selected existing JCT records only. This emits source instruction/native symbol
bindings, copy sizes and scalar launch dimensions, never full IR or process env.
"""
import hashlib
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import sys
import zstandard

L=Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf')
ROOT=Path(__file__).resolve().parents[2]
assert os.environ['JAX_PLATFORMS']=='cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path.insert(0,str(ROOT/'src'))
from gpuwrf.runtime import aot_executable as ax,aot_slim_module as asm
spec=importlib.util.spec_from_file_location('delta_gate',ROOT/'scripts/wave_cubin_gate.py')
g=importlib.util.module_from_spec(spec);sys.modules['delta_gate']=g;spec.loader.exec_module(g)


def fields(data):
    try:return list(ax._proto_fields(data))
    except (ValueError,IndexError,struct.error):return []


def symbol_names(elf):
    # ELF64/LE CUDA symbol tables; never execute or disassemble the cubin.
    assert elf[:6]==b'\x7fELF\x02\x01'
    off=struct.unpack_from('<Q',elf,40)[0]
    entsize,n=struct.unpack_from('<HH',elf,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',elf,off+i*entsize) for i in range(n)]
    result=set()
    for section in sections:
        if section[1] not in (2,11):continue
        strings=sections[section[6]]
        table=elf[strings[4]:strings[4]+strings[5]]
        for pos in range(section[4],section[4]+section[5],section[9] or 24):
            name,info,_,_,_,_=struct.unpack_from('<IBBHQQ',elf,pos)
            if info&15==2 and name<len(table):
                end=table.find(b'\0',name)
                result.add(table[name:end].decode('utf-8'))
    return result


def numbers(data,depth=0):
    if depth>6:return {}
    out={}
    for f,w,v in fields(data):
        if w==0:out.setdefault(str(f),[]).append(v)
        elif w==2:
            sub=numbers(v,depth+1)
            if sub:out.setdefault(str(f),[]).append(sub)
    return out


def walk(data,instructions,symbols,ancestors=(),path=(),depth=0):
    if depth>35:return []
    fs=fields(data)
    if not fs:return []
    first=next((v for f,w,v in fs if f==1 and w==2),None)
    name=None
    if first is not None:
        leaf=next((v for f,w,v in fields(first) if f==1 and w==2),None)
        if leaf is not None:
            try:s=leaf.decode('utf-8')
            except UnicodeDecodeError:s=''
            if s in instructions:name=s
    result=[]
    if name:
        kernel=next((v for f,w,v in fs if f==8 and w==2),None)
        copy=next((v for f,w,v in fs if f==13 and w==2),None)
        if kernel is not None:
            kf=fields(kernel)
            symbol=next((v.decode('utf-8') for f,w,v in kf if f==3 and w==2),None)
            if symbol in symbols:
                dims=next((v for f,w,v in kf if f==4 and w==2),b'')
                result.append({'kind':'kernel','hlo_instruction':name,'native_symbol':symbol,
                               'launch_dimensions_proto':numbers(dims),
                               'ancestors':list(ancestors),'proto_path':list(path)})
                return result
        if copy is not None:
            size=next((v for f,w,v in fields(copy) if f==3 and w==0),None)
            if size is None:
                d2d=next((v for f,w,v in fields(copy) if f==1 and w==2),b'')
                size=next((v for f,w,v in fields(d2d) if f==3 and w==0),None)
            result.append({'kind':'copy','hlo_instruction':name,'bytes':size,
                           'ancestors':list(ancestors),'proto_path':list(path)})
            return result
        ancestors=(*ancestors,name)
    # Runtime thunk messages only: the caller excludes the retained HLO field.
    for i,(f,w,v) in enumerate(fs):
        if w==2:
            try:s=v.decode('utf-8')
            except UnicodeDecodeError:s=None
            if s is not None and s.isprintable():continue
            result.extend(walk(v,instructions,symbols,ancestors,(*path,f,i),depth+1))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(exist_ok=False)
    idx=json.loads((L/'ACTUAL_HLO/INDEX.json').read_text());allrows=[]
    for rec in idx['records']:
        text=Path(rec['hlo_file']).read_text()
        assert hashlib.sha256(text.encode()).hexdigest()==rec['hlo_sha256']
        instruction_names=set(re.findall(r'^\s*(?:ROOT )?%([\w.\-]+) = ',text,re.M))
        blob=zstandard.ZstdDecompressor().decompress(Path(rec['cache_entry']).read_bytes(),max_output_size=1<<33)[4:]
        native=g._record(blob,2,ax,asm)
        assert hashlib.sha256(native).hexdigest()==rec['native_record2_sha256']
        symbols=set()
        for elf in g._elfs(native):symbols|=symbol_names(elf)
        payload=g._gpu_payload(blob,ax,asm)
        _,info,pos=asm._chunks(payload)[-1]
        data,_=asm._read_logical(payload,pos,info['data_size'])
        rest,=asm._decode_simple_chunk(data,1)
        runtime=[v for f,w,v in fields(rest) if f==13 and w==2]
        rows=[]
        for i,data in enumerate(runtime):rows+=walk(data,instruction_names,symbols,path=(13,i))
        assert rows and any(r['kind']=='copy' for r in rows)
        assert all(type(r['bytes']) is int for r in rows if r['kind']=='copy')
        arm='N32' if rec['version']=='v032' else 'N43'
        cache=L/'gpu/cache_v032/jax' if arm=='N32' else Path('<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.3/jax')
        proof=L/'gpu/runs'/arm/'run/wrfout/proofs/nested_pipeline_run.json'
        key=json.loads(proof.read_text())['metadata']['nested_aot']['domains'][rec['program']]['cheap_key']
        actual_paths=list((cache/'aot').rglob('k_'+key+'.xlaexec'));assert len(actual_paths)==1
        raw=actual_paths[0].read_bytes();payload=g._gpu_payload(raw,ax,asm)
        _,ai,apos=asm._chunks(payload)[-1];ad,_=asm._read_logical(payload,apos,ai['data_size'])
        actual_rest,=asm._decode_simple_chunk(ad,1)
        actual_runtime=[v for f,w,v in fields(actual_rest) if f==13 and w==2]
        assert runtime==actual_runtime,'runtime-thunk donor differs from actual loaded artifact'
        record_equal={str(i):g._record(blob,i,ax,asm)==g._record(raw,i,ax,asm) for i in (0,1,2)}
        assert all(record_equal.values()),'native/interface donor differs'
        non_hlo_equal=[x for x in fields(rest) if x[0]!=1]==[x for x in fields(actual_rest) if x[0]!=1]
        assert non_hlo_equal,'constants/allocation/thunk payload donor differs'
        binding={'arm':arm,'proof_sha256':hashlib.sha256(proof.read_bytes()).hexdigest(),
                 'cheap_key':key,'aot_path':str(actual_paths[0]),'aot_sha256':hashlib.sha256(raw).hexdigest(),
                 'native_record2_equal':True,'runtime_thunks_field13_equal':True,
                 'records_0_1_2_equal':record_equal,'all_rest_fields_except_retained_hlo_equal':non_hlo_equal}
        result={**rec,'runtime_field13_sha256':hashlib.sha256(b''.join(runtime)).hexdigest(),
                'native_symbols':sorted(symbols),'thunks':rows,'actual_execution_binding':binding}
        path=out/(rec['version']+'_'+rec['program']+'.json')
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(rec['version'],rec['program'],'symbols',len(symbols),'kernel thunks',sum(r['kind']=='kernel' for r in rows),
              'copy thunks',sum(r['kind']=='copy' for r in rows),'helpers in native',[n for n in ['memcpy32_post','set_case_condition','set_while_condition'] if n in symbols],flush=True)
        allrows.append({'version':rec['version'],'program':rec['program'],'path':str(path),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (out/'INDEX.json').write_text(json.dumps(allrows,indent=2)+'\n')


if __name__=='__main__':main()

