"""Fresh-start carry wiring and source-authority lifecycle regressions."""
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.coupling import physics_couplers as C

@dataclass(frozen=True)
class Carry:
    qke: object
    el_pbl: object
    qsq: object
    theta: object
    def replace(self,**updates):return replace(self,**updates)

@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
@pytest.mark.parametrize('traced',[False,True])
def test_fresh_first_call_seeds_carried_wrf_moments(monkeypatch,dtype,traced):
    q=jnp.full((4,2,3),25.,dtype);s=Carry(q,q,q,q)
    calls=[]
    def seed(state,grid,*,return_moments,**kw):
        assert return_moments
        calls.append(1)
        return tuple(jnp.full(q.shape,v,jnp.float32) for v in (7.,3.,.25))
    monkeypatch.setattr(C,'mynn_coldstart_qke_from_state',seed)
    call=lambda first:C._mynn_state_with_first_call_qke(s,None,first)
    out=call(jnp.asarray(True) if traced else True)
    assert calls
    for name,value in zip(('qke','el_pbl','qsq'),(7.,3.,.25)):
        a=np.asarray(getattr(out,name));assert a.dtype==np.dtype(dtype)
        np.testing.assert_array_equal(a,np.full(q.shape,np.float32(value),np.dtype(dtype)))
    assert out.theta is s.theta
    later=call(jnp.asarray(False) if traced else False)
    for name in ('qke','el_pbl','qsq','theta'):
        np.testing.assert_array_equal(getattr(later,name),getattr(s,name))
    assert C._mynn_state_with_first_call_qke(s,None,True,restart=True) is s


def test_missing_qsq_seed_mutant_fails_same_gate(monkeypatch):
    import inspect
    source=inspect.getsource(C._mynn_state_with_first_call_qke)
    needle='seed_names = tuple(name for name in ("el_pbl", "qsq")'
    assert source.count(needle)==1
    namespace={}
    exec(compile(source.replace(needle,needle.replace('"qsq"','')),'<missing-qsq-seed>','exec'),C.__dict__,namespace)
    monkeypatch.setattr(C,'_mynn_state_with_first_call_qke',namespace['_mynn_state_with_first_call_qke'])
    with pytest.raises(AssertionError):
        test_fresh_first_call_seeds_carried_wrf_moments(monkeypatch,jnp.float32,True)


def test_carried_moments_match_original_wrf(tmp_path):
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import gpuwrf
    root=Path(__file__).resolve().parents[1]
    data=root/'tests/data/w44_qsqseed'
    packet=data/'original.npz'
    provenance=json.loads((data/'provenance.json').read_text())
    assert hashlib.sha256(packet.read_bytes()).hexdigest()==provenance['original_sha256']
    # This REAL4 qualification uses the SAME fixed release/no-FMA condition
    # as test_w44_cycling_wrf_witness. Parent FAST0/1 separately exercises
    # lifecycle wiring and legacy/current carry dtypes above.
    env={k:v for k,v in os.environ.items() if not k.startswith('GPUWRF_') and k!='XLA_FLAGS'}
    env.update(JAX_PLATFORMS='cpu',GPUWRF_JAX_CACHE='0',GPUWRF_FAST_DEFAULTS='1',
             PYTHONPATH=str(Path(gpuwrf.__file__).resolve().parents[1]),XLA_FLAGS='--xla_cpu_max_isa=AVX')
    report=tmp_path/'original_comparison.json'
    run=subprocess.run([sys.executable,str(root/'proofs/w44_qsqseed/check_init.py'),str(packet),str(report)],
                       env=env,capture_output=True,text=True,timeout=180)
    assert run.returncode==0, run.stdout[-4000:]+run.stderr[-2000:]
    result=json.loads(report.read_text())
    assert result['passed'] and result['backend']=='cpu'
    assert all(r['truth_nonzero']>1000 for r in result['records'])
