"""Direct original Fortran REAL GAMMA intrinsic vs P3 REAL implementation."""
import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma as jax_gamma
import numpy as np

from gpuwrf.physics.p3.process_support import gamma


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--compiler',required=True)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    assert jax.devices()[0].platform=='cpu'
    x=np.unique(np.r_[np.linspace(1,22,4096,dtype=np.float32),np.arange(1,23,dtype=np.float32),
                      np.nextafter(np.arange(1,23,dtype=np.float32),np.float32(np.inf)),
                      np.nextafter(np.arange(1,23,dtype=np.float32),np.float32(-np.inf))]).astype(np.float32)
    x=x[x>=1]
    (a.out/'input.bin').write_bytes(x.tobytes())
    source=f'''program intrinsic_gamma
implicit none
real :: x({len(x)}),y({len(x)})
integer :: u,i
open(newunit=u,file='input.bin',form='unformatted',access='stream',status='old');read(u)x;close(u)
do i=1,size(x)
y(i)=gamma(x(i))
enddo
open(newunit=u,file='output.bin',form='unformatted',access='stream',status='replace');write(u)y;close(u)
end program
'''
    (a.out/'gamma.f90').write_text(source)
    subprocess.run([a.compiler,'-O2','-ffp-contract=off','-ffree-line-length-none','gamma.f90','-o','gamma'],cwd=a.out,check=True,timeout=30)
    subprocess.run([str(a.out/'gamma')],cwd=a.out,check=True,timeout=30)
    ref=np.fromfile(a.out/'output.bin',dtype=np.float32)
    report={}
    for name,fn in [('baseline_jax',jax_gamma),('eager',gamma),('jit',jax.jit(gamma))]:
        actual=np.asarray(fn(jnp.asarray(x)))
        assert np.isfinite(actual).all() and np.isfinite(ref).all()
        ulp=np.abs(actual.view(np.int32).astype(np.int64)-ref.view(np.int32).astype(np.int64))
        relative=np.abs((actual.astype(np.float64)-ref)/ref)
        report[name]={'max_ulp':int(ulp.max()),'max_relative':float(relative.max()),'worst_argument':float(x[ulp.argmax()]),'samples':len(x)}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if all(report[k]['max_relative']<=32*np.finfo(np.float32).eps for k in ('eager','jit')) else 1


if __name__=='__main__':raise SystemExit(main())
