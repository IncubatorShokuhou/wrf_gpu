"""Changed-artifact solar check with source-only per-field FMA envelopes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import ra_gfdleta_support as port


def read(path):
    cases=[]
    with path.open('rb') as f:
        assert np.fromfile(f,np.float32,633).size==633
        for _ in range(112):
            date=np.fromfile(f,np.int32,4);value=np.fromfile(f,np.float32,6)
            assert date.size==4 and value.size==6 and np.isfinite(value).all()
            cases.append((date,value))
        assert not f.read(1)
    return cases


def main(native,fma,output):
    assert jax.devices()[0].platform=='cpu'
    n,f=read(native),read(fma)
    reference=np.asarray([v[3:] for d,v in n],np.float32)
    contracted=np.asarray([v[3:] for d,v in f],np.float32)
    assert all(np.array_equal(a[0],b[0]) and np.array_equal(a[1][:3],b[1][:3]) for a,b in zip(n,f))
    envelope=np.max(np.abs(contracted.astype(np.float64)-reference.astype(np.float64)),axis=0)
    run=jax.jit(port.zenith,static_argnames=('year','month','day','start_hour'))
    values=[];compiled=[]
    started=time.perf_counter()
    for i,(date,fields) in enumerate(n):
        month,day,year,hour=map(int,date);times,lat,lon=fields[:3]
        operands=tuple(map(jnp.asarray,(times,lon,lat)))
        keywords=dict(year=year,month=month,day=day,start_hour=hour)
        if i%7==0:
            artifact=run.lower(*operands,**keywords).compile()
            hlo=artifact.as_text()
            memory=artifact.memory_analysis()
            compiled.append(dict(year=year,month=month,day=day,
                f64_instructions=sum(bool(re.search(r'\bf64\[',line)) for line in hlo.splitlines()),
                hlo_lines=len(hlo.splitlines()),temp_bytes=memory.temp_size_in_bytes))
        values.append(run(*operands,**keywords))
    candidate=np.asarray(values,np.float32)
    assert candidate.shape==reference.shape and np.isfinite(candidate).all()
    error=np.abs(candidate.astype(np.float64)-reference.astype(np.float64))
    allowance=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12
    report=dict(source_sha256=hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(),
        fields={name:dict(max_abs=float(error[:,i].max()),source_FMA_Linf=float(envelope[i]),
                         outside_strict=int(np.count_nonzero(error[:,i]>allowance[:,i])),
                         outside_source_envelope=int(np.count_nonzero(error[:,i]>envelope[i]+allowance[:,i])))
                for i,name in enumerate(('cosine','day','hour'))},
        compiled_artifacts=compiled,wall_s=time.perf_counter()-started,
        scope='CPU changed solar artifact112cases; no GPU/R3 timing claim; literalBYTE carried by finite nextafter identity')
    output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),candidate=candidate,reference=reference,source_fma=contracted)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('native','fma','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.native,a.fma,a.output)
