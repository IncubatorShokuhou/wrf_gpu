"""Literal REAL4 expression check against pristine's no-vector/no-FMA build.

The optimized WRF compiler uses libmvec EXPF, which differs from scalar
EXPF. This CPU proof isolates that compiler intrinsic from expression ordering:
production Python arithmetic executes unfused with scalar EXPF and compares
to independent no-vector/no-FMA Fortran FDDAGD savepoints. Production JAX
retains its exponential and passes the separate bounded compiled gate.
"""
from pathlib import Path
import ctypes,ctypes.util,importlib.util,json,sys
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.assimilation import fdda_grid as fg,fdda_surface as sf


def run(reference_dir):
    assert jax.devices()[0].platform=='cpu'
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('surface_literal_fixtures',root/'tests/v040/sfdda/test_surface_oracle.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    lib=ctypes.CDLL(ctypes.util.find_library('m'))
    expf=lib.expf;expf.argtypes=[ctypes.c_float];expf.restype=ctypes.c_float
    intrinsic=[]
    original=sf.jnp
    class ScalarExp:
        def __getattr__(self,name):return getattr(original,name)
        def exp(self,a):
            a=np.asarray(a);assert a.dtype==np.float32
            ref=np.array([expf(float(x)) for x in a.flat],np.float32).reshape(a.shape)
            backend=np.asarray(original.exp(jnp.asarray(a)))
            ulp=np.abs(backend.view(np.uint32).astype(np.int64)-ref.view(np.uint32).astype(np.int64))
            intrinsic.append(dict(values=a.size,different=int(np.count_nonzero(ulp)),max_ulp=int(ulp.max())))
            return original.asarray(ref)
    rows=[]
    jax.clear_caches()
    try:
        sf.jnp=ScalarExp()
        with jax.disable_jit():
            for name in ('swiss_sfc1_plain_r4.npz','swiss_sfc2_plain_r4.npz',
                         'swiss_sfc1_wind_census_r4.npz','swiss_sfc2_wind_census_r4.npz'):
                d,inp=m.load(name);c=fg.initial_fdda_carry(inp,jnp.float32)
                reference=dict(np.load(Path(reference_dir)/name))
                for key,a in d.items():
                    if not key.startswith('call'):
                        assert a.dtype==reference[key].dtype and a.shape==reference[key].shape and a.tobytes()==reference[key].tobytes(),key
                f=lambda n:jnp.asarray(d['frame1_'+n])
                c=fg.fdda_driver_tendencies(c,inp,f('u'),f('v'),f('th_phy'),f('qv'),f('pblh'),d['ht'],f('z_at_w'),
                    0.,1,jnp.float32,surface_context=(f('t_phy'),f('p_phy'),f('regime'),f('znt')))
                count=0
                for field,a in zip(('run','rvn','rth','rqv'),c.tendencies):
                    a,b=np.asarray(a),reference['call0_unc_'+field]
                    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),(name,field,np.max(np.abs(a-b)))
                    count+=a.size
                rows.append(dict(file=name,values=count,all_byte_exact=True))
    finally:
        sf.jnp=original
        jax.clear_caches()
    return dict(literal_real4=True,jit=False,expf_library=lib._name,reference_dir=str(reference_dir),
                reference_flags='-fno-tree-vectorize -fno-tree-slp-vectorize -ffp-contract=off',rows=rows,intrinsic_spread=intrinsic,
                production_exponential_unchanged=True)


if __name__=='__main__':
    result=run(sys.argv[2]);Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
