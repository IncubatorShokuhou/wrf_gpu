"""Read native SWR93 calls, retaining only physically used cloud slots."""
import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path
import numpy as np


def shape(dimensions,nz):
    names={key:1 for key in ('its','ite','ims','ime','jts','jte','jms','jme','kts','kms','kds','ids','jds')}
    names.update(ide=2,jde=2,kte=nz,kme=nz+1,kde=nz+1,nb=12,nbly=15)
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.FloorDiv:operator.floordiv}

    def value(node):
        if isinstance(node,ast.Constant) and isinstance(node.value,int):return node.value
        if isinstance(node,ast.Name):return names[node.id]
        if isinstance(node,ast.BinOp) and type(node.op) in ops:return ops[type(node.op)](value(node.left),value(node.right))
        raise ValueError(('unsupported static source dimension',ast.dump(node)))

    def integer(text):return value(ast.parse(text,mode='eval').body)
    result=[]
    for dim in dimensions:
        if ':' in dim:
            lower,upper=dim.split(':');result.append(integer(upper)-integer(lower)+1)
        else:result.append(integer(dim))
    return tuple(result)


def read(raw,manifest):
    meta=json.loads(manifest.read_text());rows=[]
    cloud={'camt','ktopsw','kbtmsw','crr','ctt'}
    with raw.open('rb') as f:
        while True:
            header=np.fromfile(f,np.int32,2)
            if header.size==0:break
            assert header.size==2
            nz,ncloud=map(int,header)
            assert 3<=nz<=100 and 0<=ncloud<nz
            row=dict(nlayers=np.asarray(nz,np.int32),nclds=np.asarray([ncloud],np.int32))
            for output,names in ((False,meta['inputs']),(True,meta['outputs'])):
                for key in names:
                    full=shape(meta['shapes'].get(key,()),nz)
                    used=full
                    if key in cloud:used=full[:-1]+(ncloud+meta['cloud_slot_offsets'][key],)
                    if output and key in {'hswc','hswl'}:used=full[:-1]+(nz,)
                    dtype=np.int32 if meta['types'][key]=='i' else np.float32
                    size=int(np.prod(used));value=np.fromfile(f,dtype,size)
                    assert value.size==size and np.isfinite(value).all(),key
                    value=value.reshape(used,order='F')
                    if not output and used!=full:
                        padded=np.zeros(full,dtype=dtype)
                        padded[tuple(slice(0,n) for n in used)]=value
                        value=padded
                    row[('out_' if output else '')+key]=value
            rows.append(row)
    assert rows
    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]},meta


def main(raw,manifest,final,reference,output):
    assert final.read_bytes()==reference.read_bytes(),'Native writes changed final output'
    data,meta=read(raw,manifest)
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(dict(
        native_final_BYTE=True,calls=len(data['nlayers']),layers=np.unique(data['nlayers']).tolist(),
        cloud_counts=np.unique(data['nclds']).tolist(),
        fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        native_final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        source_capture=meta,scope='native true caller SWR93 operands/results; KBTMSW NCLDS+2 sentinel included; noncontributing padding zero; heating valid1..L'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('raw','manifest','final','reference','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.raw,a.manifest,a.final,a.reference,a.output)
