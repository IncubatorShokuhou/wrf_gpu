"""Finite-first original WRF qualification of scalar GFDL support functions."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from literal import load
from gpuwrf.physics import ra_gfdleta_support as support


def main(native, output):
    assert jax.devices()[0].platform == 'cpu'
    literal = load(Path(support.__file__))
    with native.open('rb') as f:
        gaussian = np.fromfile(f,np.float32,311)
        temperatures = np.fromfile(f,np.float32,161)
        saturation = np.fromfile(f,np.float32,161)
        cases = []
        for _ in range(112):
            date = np.fromfile(f,np.int32,4)
            fields = np.fromfile(f,np.float32,6)
            assert date.size==4 and fields.size==6
            cases.append((date,fields))
        assert not f.read(1)
    results = {}

    def check(name, candidate, reference):
        candidate = np.asarray(candidate)
        reference = np.asarray(reference)
        assert candidate.shape==reference.shape and candidate.dtype==reference.dtype==np.float32
        assert np.isfinite(candidate).all() and np.isfinite(reference).all()
        error=np.abs(candidate.astype(np.float64)-reference.astype(np.float64))
        bound=16.*np.spacing(np.abs(reference)).astype(np.float64)+1.e-12
        results[name] = {'BYTE':candidate.tobytes()==reference.tobytes(),
                         'outside_16ulp':int(np.count_nonzero(error>bound)),
                         'max_abs':float(error.max())}

    x = np.arange(311,dtype=np.float32)*np.float32(.01)
    check('gaussian_literal',np.asarray([literal.gaussian_area(v) for v in x]),gaussian)
    check('gaussian_compiled',jax.jit(jax.vmap(support.gaussian_area))(jnp.asarray(x)),gaussian)
    check('saturation_literal',literal.saturation_pressure(temperatures),saturation)
    check('saturation_compiled',jax.jit(support.saturation_pressure)(jnp.asarray(temperatures)),saturation)
    literal_zen=[];compiled_zen=[];reference_zen=[]
    run=jax.jit(support.zenith,static_argnames=('year','month','day','start_hour'))
    for date,fields in cases:
        month,day,year,hour=map(int,date)
        args=dict(year=year,month=month,day=day,start_hour=hour)
        times,lat,lon=fields[:3]
        literal_zen.append(literal.zenith(times,lon,lat,**args))
        compiled_zen.append(run(jnp.asarray(times),jnp.asarray(lon),jnp.asarray(lat),**args))
        reference_zen.append(fields[3:])
    check('zenith_literal',np.asarray(literal_zen,np.float32),np.asarray(reference_zen,np.float32))
    check('zenith_compiled',np.asarray(compiled_zen,np.float32),np.asarray(reference_zen,np.float32))
    output.write_text(json.dumps(dict(results=results,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(support.__file__).read_bytes()).hexdigest(),
        scope='GAUSIN311; FPVS_new161 alternative-not-active-Ferrier; ZENITH112 across16dates+7latitudes'),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('native',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();main(a.native,a.output)
