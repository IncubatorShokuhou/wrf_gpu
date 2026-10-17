"""Compare literal/compiled SOLARD against independent unchanged WRF body."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import ra_gfdleta_support as support
from literal import load


def main(native,output):
    assert jax.devices()[0].platform=='cpu'
    dtype=np.dtype([('args',np.int32,(4,)),('r1',np.float32)])
    data=np.fromfile(native,dtype=dtype);assert len(data)==384
    literal=load(Path(support.__file__))
    run=jax.jit(jax.vmap(support.solar_distance))
    args=tuple(jnp.asarray(data['args'][:,i]) for i in range(4))
    start=time.perf_counter();exe=run.lower(*args).compile();compile_s=time.perf_counter()-start
    candidate=np.asarray(exe(*args));hlo=exe.as_text();memory=exe.memory_analysis()
    strict=np.asarray([literal.solar_distance(*row) for row in data['args']],np.float32)
    ref=data['r1'];error=np.abs(candidate.astype(np.float64)-ref.astype(np.float64))
    bound=16.*np.spacing(np.abs(ref)).astype(np.float64)+1.e-12
    report=dict(records=len(data),literal_BYTE=strict.tobytes()==ref.tobytes(),
        literal_max_abs=float(np.max(np.abs(strict-ref))),compiled_max_abs=float(error.max()),
        compiled_outside16ulp=int(np.count_nonzero(error>bound)),finite=bool(np.isfinite(candidate).all()),
        compile_s=compile_s,temporary_bytes=memory.temp_size_in_bytes,hlo_lines=len(hlo.splitlines()),
        f64_rows=sum('f64[' in line for line in hlo.splitlines()),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(support.__file__).read_bytes()).hexdigest(),
        scope='native SOLARD 8years x12months x4hours, includes actual2026; CPU only, noGPU claim')
    np.savez_compressed(output.with_suffix('.npz'),args=data['args'],reference=ref,literal=strict,compiled=candidate)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.native,a.output)
