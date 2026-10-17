"""Exact LWR88 operands/results and native grid-dependent CO2 provider."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from capture_gfdl_sw import shape

MATRICES={'co251','cdt51','cdt58','c2d51','c2d58','co258'}
LAYERS={'co2m51','co2m58','cdtm51','cdtm58','c2dm51','c2dm58'}


def read(raw,manifest):
    meta=json.loads(manifest.read_text());rows=[];provider=None;static_provider=None
    with raw.open('rb') as f:
        while True:
            header=np.fromfile(f,np.int32,1)
            if header.size==0:break
            nz=int(header[0]);assert 3<=nz<=100
            row=dict(nlayers=np.asarray(nz,np.int32))
            for key in meta.get('environment',()):
                value=np.fromfile(f,np.float32,1)
                assert value.size==1 and np.isfinite(value).all()
                row['environment_'+key]=value.reshape(())
            domain=meta.get('domain_static',{})
            fields=([(name,'input') for name in meta['inputs']]+[(name,'co2') for name in meta['co2_tables']]
                    +[(name,'domain') for name in domain]+[(name,'output') for name in meta['outputs']])
            for name,phase in fields:
                co2=phase=='co2';output=phase=='output';static=phase=='domain'
                if co2:dims=(nz+1,nz+1) if name in MATRICES else (nz,) if name in LAYERS else (nz+1,)
                elif static:dims=tuple(domain[name][1])
                else:dims=shape(meta['shapes'].get(name,()),nz)
                dtype=np.int32 if (static and domain[name][0]=='i') or (not co2 and not static and meta['types'][name]=='i') else np.float32
                used=dims
                if name in meta.get('active_mass_only',()):used=dims[:1]+(nz,)+dims[2:]
                n=int(np.prod(used));value=np.fromfile(f,dtype,n)
                assert value.size==n and np.isfinite(value).all(),name
                value=value.reshape(used,order='F')
                if used!=dims:
                    padded=np.zeros(dims,dtype=dtype)
                    padded[tuple(slice(0,n) for n in used)]=value;value=padded
                row[('co2_' if co2 else 'domain_' if static else 'out_' if output else '')+name]=value
            tables={name:row.pop('co2_'+name) for name in meta['co2_tables']}
            if provider is None:provider=tables
            else:assert all(provider[k].tobytes()==v.tobytes() for k,v in tables.items()),'Native CO2 provider changed within case'
            static={name:row.pop('domain_'+name) for name in domain}
            if static_provider is None:static_provider=static
            else:assert all(static_provider[k].tobytes()==v.tobytes() for k,v in static.items()),'Native domain static provider changed'
            rows.append(row)
    assert rows and provider is not None
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    arrays.update({'co2_'+k:v for k,v in provider.items()})
    arrays.update({'domain_'+k:v for k,v in static_provider.items()})
    return arrays,meta


def main(raw,manifest,final,reference,output):
    assert final.read_bytes()==reference.read_bytes(),'Native diagnostics changed final output'
    data,meta=read(raw,manifest)
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(dict(
        native_final_BYTE=True,calls=len(data['nlayers']),layers=np.unique(data['nlayers']).tolist(),
        fixture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        native_final_sha256=hashlib.sha256(final.read_bytes()).hexdigest(),
        co2_provider_same_all_calls=True,source_capture=meta,
        scope='native LWR88 caller operands/3physical outputs +26initialized grid-specific CO2 tables; not operational CO2 initialization'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('raw','manifest','final','reference','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.raw,a.manifest,a.final,a.reference,a.output)
