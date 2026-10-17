"""Independent original-WRF EOS with separately fedback ALB, not self parity."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.contracts.state import BaseState
from gpuwrf.kernels import dyn_rk_fp32 as native

FIXTURE=Path(__file__).parent/'fixtures/u2_alb_eos_oracle.npz'
META=json.loads(FIXTURE.with_suffix('.json').read_text())


@pytest.mark.parametrize('index,case',list(enumerate(META['cases'])))
def test_independent_alb_native_eos(index,case,monkeypatch):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==META['fixture_sha256']
    with np.load(FIXTURE) as file:
        vals={name.rsplit('_',1)[0]:jnp.asarray(value) for name,value in file.items()
              if name.endswith('_'+str(index))}
    state=SimpleNamespace(theta=vals['T']+jnp.float32(300.),ph_perturbation=vals['PH'],mu_perturbation=vals['MU'])
    base=BaseState(pb=vals['PB'],phb=vals['PHB'],mub=vals['MUB'],t0=jnp.zeros_like(vals['T']),
                   theta_base=jnp.zeros_like(vals['T']),alb=vals['alb'])
    metrics=SimpleNamespace(**{name:vals[name] for name in ('c1h','c2h','c3h','c4h','c3f','c4f','dnw','rdnw')},p_top=vals['ptop'])
    # Preserve the product's explicit PTX RN semantics in CPU interpretation.
    def rn(op,a,b,*,interpret):
        with jax.enable_x64(True):
            a,b=a.astype(jnp.float64),b.astype(jnp.float64)
            exact={'add':lambda:a+b,'sub':lambda:a-b,'mul':lambda:a*b,'div':lambda:a/b}[op]()
            return jax.lax.reduce_precision(exact,exponent_bits=8,mantissa_bits=23).astype(jnp.float32)
    monkeypatch.setattr(native,'_EOS_INTERPRET',True)
    monkeypatch.setattr(native,'_rn',rn)
    # CPU XLA logf and the pristine compiler's libm logf differ by one
    # REAL ulp, amplified by AL=ALT-ALB cancellation. Hold this external
    # math primitive to the original CPU library when checking the wiring.
    # This is an emulated CPU operator gate; GPU libdevice remains pending.
    import ctypes,ctypes.util
    lib=ctypes.CDLL(ctypes.util.find_library('m'))
    lib.logf.argtypes=[ctypes.c_float];lib.logf.restype=ctypes.c_float
    def logf(a):
        def host(x):
            x=np.asarray(x)
            return np.fromiter((lib.logf(float(v)) for v in x.flat),np.float32).reshape(x.shape)
        return jax.pure_callback(host,jax.ShapeDtypeStruct(a.shape,jnp.float32),a)
    monkeypatch.setattr(native.jnp,'log',logf)
    actual=native.diagnose_pressure_fp32(state,base,metrics,hypsometric_opt=case['hypsometric_opt'])
    jax.block_until_ready(actual)
    assert all(np.isfinite(np.asarray(v)).all() for v in actual)
    np.testing.assert_array_equal(actual[1],vals['al'])
    # libm pow vs CPU XLA pow differs in last REAL bits; unchanged EOS bound.
    np.testing.assert_allclose(actual[0],vals['p'],rtol=0,atol=.0625)
