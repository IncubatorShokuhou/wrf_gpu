"""Original pristine NUDOB multi-level sounding and negative-QV innovation arms."""
from pathlib import Path
import hashlib,json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.assimilation.obs_nudging import SingleLevelObservations,SingleLevelNudgingOptions,single_level_nudob
from gpuwrf.assimilation.obs_profiles import ProfileData

jax.config.update('jax_enable_x64',True)
SAVE=Path(__file__).resolve().parents[2]/'proofs/s1_obsnudge/full/savepoints'
MANIFEST=json.loads((SAVE/'SPREAD_MANIFEST.json').read_text())


@pytest.mark.parametrize('name',sorted(MANIFEST['cases']))
def test_profile_nudob_pristine(name):
    meta=MANIFEST['cases'][name]
    assert hashlib.sha256((SAVE/name).read_bytes()).hexdigest()==meta['sha256']
    with np.load(SAVE/name) as source:d={k:source[k] for k in source.files}
    o=SingleLevelNudgingOptions(**meta['options'])
    obs=SingleLevelObservations(*(jnp.asarray(d[k]) for k in SingleLevelObservations._fields))
    profile=ProfileData(*(jnp.asarray(d[k]) for k in ('prof_pressure','prof_innovation','prof_values','prof_q_model','nlevels')))
    args={k:jnp.asarray(d[k]) for k in ('mu','c1h','c2h','msfx','msfy','pbase','pp','kpbl','vih1','vih2','terrain','z')}
    call=jax.jit(lambda a,s,k,x,t,r,kr:single_level_nudob(o,obs,a,s,ktau=k,xtime=x,tfaci=t,restart=r,ktaur=kr,
                       **args,_profiles=profile,_qv=jnp.asarray(d['qv']),_scale_negative_qv=meta['scale'],_maxsnd_gap=meta['maxgap']))
    a,s=jnp.asarray(d['initial_aten']),jnp.asarray(d['initial_saved'])
    eps=np.finfo(d['pbase'].dtype).eps
    valid_innov=d['prof_innovation'][np.abs(d['prof_innovation'])<=9e4]
    amp=max(np.abs(valid_innov).max(),np.abs(d['innovation'][np.abs(d['innovation'])<=9e4]).max())
    mass=np.abs(d['mu']/d['msfy' if o.ivar==1 else 'msfx']).max() if o.ivar<3 else np.abs(d['c1h'][:,None]*d['mu']+d['c2h'][:,None]).max()
    ba=16*eps*(np.abs(d['initial_aten'])+o.coefficient*mass*8*amp)
    bs=16*eps*8*amp
    for n,((ktau,ktaur,rest),(xtime,tfaci)) in enumerate(zip(d['calls'],d['times'])):
        a,s=call(a,s,ktau,xtime,tfaci,bool(rest),ktaur)
        for actual,expected,bound in ((a,d['oracle_aten'][n],ba),(s,d['oracle_saved'][n],bs)):
            value=np.asarray(actual)
            assert np.isfinite(value).all() and np.isfinite(expected).all()
            assert np.all(np.abs(value-expected)<=bound),(name,n,float(np.abs(value-expected).max()),float(np.max(bound)))
