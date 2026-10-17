"""Native 112-case qualification of explicit dynamic GFDL date operands."""
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
from literal import load
from qualify_zenith_compiled import read


def main(native,fma,output):
    assert jax.devices()[0].platform=='cpu'
    cases=read(native);contracted=read(fma)
    reference=np.asarray([v[3:] for d,v in cases],np.float32)
    source_fma=np.asarray([v[3:] for d,v in contracted],np.float32)
    assert all(np.array_equal(a[0],b[0]) and np.array_equal(a[1][:3],b[1][:3]) for a,b in zip(cases,contracted))
    literal=load(Path(port.__file__))
    lit=[];compiled=[];run=jax.jit(port.zenith_dynamic);artifact=None
    for date,fields in cases:
        times,lat,lon=fields[:3]
        operands=(times,lon,lat,date[:3],np.asarray(date[3],np.int32))
        lit.append(literal.zenith_dynamic(*operands))
        args=tuple(map(jnp.asarray,operands))
        if artifact is None:
            started=time.perf_counter();executable=run.lower(*args).compile()
            hlo=executable.as_text();memory=executable.memory_analysis()
            artifact=dict(compile_s=time.perf_counter()-started,
                f64_instructions=sum(bool(re.search(r'\bf64\[',line)) for line in hlo.splitlines()),
                hlo_lines=len(hlo.splitlines()),temporary_bytes=memory.temp_size_in_bytes)
            output.with_suffix('.hlo').write_text(hlo)
        compiled.append(executable(*args))
    lit=np.asarray(lit);compiled=np.asarray(compiled)
    assert lit.dtype==compiled.dtype==np.float32 and lit.shape==compiled.shape==reference.shape
    assert np.isfinite(lit).all() and np.isfinite(compiled).all() and np.isfinite(reference).all()
    envelope=np.max(np.abs(source_fma.astype(np.float64)-reference.astype(np.float64)),axis=0)
    results={}
    for i,key in enumerate(('cosine','day','hour')):
        error=np.abs(compiled[:,i].astype(np.float64)-reference[:,i].astype(np.float64))
        allowance=16.*np.spacing(np.abs(reference[:,i])).astype(np.float64)+1.e-12
        results[key]=dict(literal_BYTE=lit[:,i].tobytes()==reference[:,i].tobytes(),max_compiled_abs=float(error.max()),
            strict_outside=int(np.count_nonzero(error>allowance)),source_FMA_Linf=float(envelope[i]),
            outside_source_envelope=int(np.count_nonzero(error>envelope[i]+allowance)))
    output.write_text(json.dumps(dict(results=results,compiled_artifact=artifact,
        source_sha256=hashlib.sha256(Path(port.__file__).read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        scope='112originalWRFZENITHcalls, dynamicnativeIDAT/startHour, oneCPUartifact; savedMONTH policy sameFG19; noGPU/D6claim'),indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),literal=lit,compiled=compiled,native=reference)
    print(json.dumps(dict(results=results,artifact=artifact),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('native','fma','output'):p.add_argument(key,type=Path)
    a=p.parse_args();main(a.native,a.fma,a.output)
