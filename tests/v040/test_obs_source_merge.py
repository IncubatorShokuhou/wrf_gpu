"""Observation sources enter the production merge before moist-theta conversion.

Reference sources are the original WRF driver before/after savepoints. The
conversion formula is module_big_step_utilities_em.F:6699-6700. This gate
isolates the actual production merge from the independently tested station
operator, rather than comparing two port implementations.
"""
from pathlib import Path
from types import SimpleNamespace
import inspect
import jax.numpy as jnp
import numpy as np
from gpuwrf.runtime import operational_mode as om

ROOT=Path(__file__).resolve().parents[2]/'proofs/s1_obsnudge/full/savepoints'


def test_production_merge_uses_original_obs_sources_before_moist_conversion():
    a=np.load(ROOT/'obs_probe_000002.npz');r=np.float32;eps=np.finfo(r).eps
    before,after=a['tend_before'],a['tend_after']
    mass=a['c1h'][:,None,None]*a['mut']+a['c2h'][:,None,None]
    increment=after-before
    # Add unchanged first/last wind faces that the root driver does not own.
    sources=(np.pad(increment[0],((0,0),(0,0),(0,1))),
             np.pad(increment[1],((0,0),(0,1),(0,0))),increment[2],increment[3])
    q=a['qv'];rr=r(r(461.6)/r(287));fac=r(1)+rr*q
    theta=(a['tb']+r(300))*fac
    metrics=SimpleNamespace(c1h=jnp.asarray(a['c1h']),c2h=jnp.asarray(a['c2h']),precision='fp32')
    dry=om._source_leaf_dry_tendencies(jnp.asarray(a['mut']),jnp.asarray(before[2]/mass),
        None,jnp.asarray(before[3]/mass),None,None,jnp.asarray(q),jnp.asarray(theta),
        metrics,jnp.float32,real_glue=True,obs_tendf=tuple(jnp.asarray(x) for x in sources))
    expected=fac*after[2]+rr*theta/fac*after[3]
    actual=np.asarray(dry.t_tendf)
    bound=16*eps*(np.abs(expected)+np.abs(fac*after[2])+np.abs(rr*theta/fac*after[3])+r(1))
    assert np.isfinite(actual).all()
    assert np.all(np.abs(actual-expected)<=bound)
    for value,truth in ((dry.ru_tendf,sources[0]),(dry.rv_tendf,sources[1])):
        actual=np.zeros_like(truth) if value is None else np.asarray(value)
        assert np.count_nonzero(truth)>0
        assert np.all(np.abs(actual-truth)<=4*eps*np.maximum(np.abs(truth),r(1)))
    # Guard the actual caller's forwarding as well as the numerical merge.
    source=inspect.getsource(om._physics_step_forcing)
    assert '"obs_tendf": obs_tendf' in source
    assert 'scalars["qv"] + qv_rate' in source
