"""Full pristine SMS operator and caller gates; no JAX-vs-JAX physics truth."""
from pathlib import Path
import sys
import numpy as np
import pytest
import jax

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from proofs.s1_km5 import full_oracle as ora


@pytest.mark.parametrize('key',ora.CASES)
def test_sms_complete_pristine_operator(key):
    _assert_operator(key)


def test_sms_complete_compiled_pristine_operator():
    _assert_operator('real_mixed',compiled=True)


def _checked_outputs(result):
    return (result.ru_tendf,result.rv_tendf,result.rw_tendf,result.t_tendf,
            result.tke_tendf,*result.moist_tendf,
            *(result.sms_updates[n] for n in ('sms_gamu','sms_gamv','pblh','hfx')))


def _assert_operator(key,compiled=False):
    from gpuwrf.dynamics.les3d_smagorinsky import les3d_smagorinsky_memory,_Mem
    with np.load(ROOT/'data/fixtures/sms3dtke-full-wrf-v1.npz') as z:
        inp,cfg=ora.inputs(z,key)
        fn=lambda x:les3d_smagorinsky_memory(x,cfg)
        # Compile each checked output separately: XLA can eliminate unrelated
        # tendencies instead of retaining the complete 12.5 GiB CPU executable.
        # Every original output, pristine operand and tolerance is still checked.
        result=None if compiled else _checked_outputs(fn(inp))
        def output(i):
            if not compiled:
                return result[i]
            try:
                return np.asarray(jax.jit(lambda x:_checked_outputs(fn(x))[i])(inp))
            finally:
                jax.clear_caches()
        nz,ny,nx1=inp.u.shape;m=_Mem(nx1-1,ny,nz,np.float32)
        fields=[('ru_tendf',inp.u,1,0,nz),
                ('rv_tendf',inp.v,0,1,nz),
                ('rw_tendf',inp.w,0,0,nz+1),
                ('t_tendf',inp.thp,0,0,nz),
                ('tke_tendf',inp.tke,0,0,nz)]
        fields += [(name,inp.moist[i],0,0,nz) for i,name in enumerate(('qv','qc','qr','qi','qs','qg','Ni','Nr'))]
        mass=inp.c1h[:nz,None,None]*inp.mut[None]+inp.c2h[:nz,None,None]
        eps=32*np.finfo(np.float32).eps
        for i,(name,entry,istag,jstag,nk) in enumerate(fields):
            a=output(i)
            r=z[key+'/out/'+name];a=np.asarray(m.interior(a,istag=istag,jstag=jstag,nk=nk))
            r=np.asarray(m.interior(r,istag=istag,jstag=jstag,nk=nk))
            assert np.isfinite(a).all() and np.isfinite(r).all(),(key,name)
            # Independently registered field-equivalent subtraction floor.
            atol=eps*float(abs(mass).max())*float(abs(entry).max())/float(inp.dt)
            np.testing.assert_allclose(a,r,rtol=eps,atol=atol,err_msg=key+'/'+name)
        for i,name in enumerate(('sms_gamu','sms_gamv','pblh','hfx'),start=len(fields)):
            a=np.asarray(output(i));r=z[key+'/out/'+name][3:3+ny,3:3+nx1-1]
            assert np.isfinite(a).all() and np.isfinite(r).all()
            np.testing.assert_allclose(a,r,rtol=eps,atol=eps*abs(r).max(),err_msg=name)


def test_sms_three_own_steps_nine_rk_stages_pristine():
    from proofs.s1_km5.stage_oracle import advance
    actual,_=advance()
    with np.load(ROOT/'data/fixtures/sms3dtke-stages-wrf-v1.npz') as z:
        reference=np.stack([z[f'{s}/tke'] for s in range(3)])
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    assert actual.dtype==np.float32 and np.count_nonzero(reference)>0
    eps=32*np.finfo(np.float32).eps
    np.testing.assert_allclose(actual,reference,rtol=eps,atol=eps*abs(reference).max())
