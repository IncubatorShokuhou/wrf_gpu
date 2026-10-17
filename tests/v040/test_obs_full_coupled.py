"""Full reader/driver cadence on actual pristine Swiss physics/RK boundaries.

Tolerance fixed before running the composed candidate: 32 REAL eps of the
operand scale (wind40 m/s, dry theta600 K, QV.05 kg/kg). The coupled tendency
bound propagates that envelope through the original coefficient and mass.
Missing values are checked separately, never used to inflate the bound.
"""
from datetime import datetime
from pathlib import Path
import hashlib,json
import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.assimilation.obs_driver import ObservationInputs,initial_observation_carry,observation_driver
from gpuwrf.io.obs_domain import read_obs_domain
from gpuwrf.io.obs_namelist import DEFAULTS
from gpuwrf.io.obs_prepare import prepare_reports
from gpuwrf.io.obs_projection import ObservationProjection

ROOT=Path(__file__).resolve().parents[2]/'proofs/s1_obsnudge/full/savepoints'


def prepared_inputs():
    meta=json.loads((ROOT/'COUPLED_MANIFEST.json').read_text())
    options=dict(DEFAULTS,**meta['original_provenance']['options'])
    projection=ObservationProjection(**meta['projection'])
    reports=read_obs_domain(ROOT/'OBS_DOMAIN101')
    data=prepare_reports(reports,projection,datetime.fromisoformat(meta['original_provenance']['start']),
                         options,18.,42,42)
    return ObservationInputs(**{k:jnp.asarray(data[k]) for k in
          ('times','arrivals','deactivated','rio','rjo','rko','variants','variant_steps','lengths')},
          options=tuple(sorted(options.items())),max_reports=data['max_reports'],
          fingerprint=hashlib.sha256((ROOT/'OBS_DOMAIN101').read_bytes()).hexdigest()),meta


def test_full_reader_and_resident_driver_on_pristine_boundaries():
    inputs,meta=prepared_inputs()
    carry=initial_observation_carry(inputs,(44,42,42))
    eps=np.finfo(np.float32).eps
    scales=np.asarray([40,40,600,.05],np.float32)
    args=('u','v','tb','qv','pbase','pp','z','uratx','vratx','tratx','kpbl','regime','pblh',
          'terrain','mut','muu','muv','c1h','c2h','msfux','msfuy','msfvx','msfvy','msftx','msfty')
    call=jax.jit(lambda carry,kw:observation_driver(carry,inputs,dt_s=18.,dx=3000.,**kw))
    for arm in meta['arms']:
        path=ROOT/arm['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==arm['sha256']
        a=dict(np.load(path));kw={k:jnp.asarray(a[k]) for k in args}
        kw.update(ktau=jnp.asarray(a['step']),xtime=jnp.asarray(a['xtime']))
        carry,sources=call(carry,kw)
        sources=[np.asarray(s)[:,:42,:42] for s in sources]
        weights=np.stack([np.asarray(w) for w in carry.weights])
        assert np.isfinite(weights).all()
        weight_bound=32*eps*(np.abs(a['weights_after'])+scales[:,None,None,None])
        weight_diff=np.abs(weights-a['weights_after'])
        assert np.all(weight_diff<=weight_bound),dict(step=int(a['step']),
            maxdiff=weight_diff.reshape(4,-1).max(axis=1).tolist(),
            location=np.unravel_index(np.argmax(weight_diff/weight_bound),weight_diff.shape))
        for ivar,source in enumerate(sources):
            assert np.isfinite(source).all()
            # NUDOB adds directly to the already merged RA/BL/CU/grid source.
            actual=a['tend_before'][ivar]+source
            mass=(a['muu']/a['msfuy'] if ivar==0 else a['muv']/a['msfvx'] if ivar==1
                  else a['c1h'][:,None,None]*a['mut']+a['c2h'][:,None,None])
            bound=32*eps*(np.abs(actual)+np.abs(mass)*3e-4*scales[ivar])
            assert np.all(np.abs(actual-a['tend_after'][ivar])<=bound),(arm['step'],ivar,
                    float(np.max(np.abs(actual-a['tend_after'][ivar]))))
        if int(a['step'])==0:
            # Independent source checks cover host projection/QC/RH conversion.
            selected=np.asarray(carry.varobs)[np.asarray(inputs.lengths)>0]
            flat=np.concatenate([selected[i,:int(n)] for i,n in enumerate(np.asarray(inputs.lengths))])
            assert np.all(np.abs(flat[:,:4]-a['varobs_after'][:,:4])<=32*eps*(
                          np.abs(a['varobs_after'][:,:4])+np.asarray([40,40,600,.05])))
        assert any(np.count_nonzero(s) for s in sources)
