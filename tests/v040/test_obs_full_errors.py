"""Full WRF ERROB oracle boundary (not yet operational feature admission)."""
import hashlib,json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.assimilation.obs_errors import errob

jax.config.update('jax_enable_x64',True)
SAVE=Path(__file__).resolve().parents[2]/'proofs/s1_obsnudge/full/savepoints'
MANIFEST=json.loads((SAVE/'ERROR_MANIFEST.json').read_text())


@pytest.mark.parametrize('name',sorted(MANIFEST['cases']))
def test_errob_pristine(name):
    meta=MANIFEST['cases'][name]
    assert hashlib.sha256((SAVE/name).read_bytes()).hexdigest()==meta['sha256']
    with np.load(SAVE/name) as source:d={k:source[k] for k in source.files}
    args={k:jnp.asarray(d[k]) for k in ('u','v','tb','qv','pbase','pp','z','uratx','vratx','tratx','kpbl')}
    call=jax.jit(lambda var,rko:errob(var,rko,jnp.asarray(d['rio']),jnp.asarray(d['rjo']),
                        **args,**{k:meta[k] for k in ('wind','temperature','moisture')}))
    var,rko=map(jnp.asarray,(d['initial_varobs'],d['initial_rko']))
    eps=np.finfo(d['tb'].dtype).eps
    scales=np.asarray([40,40,600,.05,100,100,100,100,2*d['tb'].shape[0],.05])
    vscale=np.asarray([40,40,600,.05,100,20000])
    for c in range(3):
        var,rko,err=call(var,rko)
        for actual,expect,scale in ((var,d['oracle_varobs'][c],vscale),(rko,d['oracle_rko'][c],2*d['tb'].shape[0]),
                                    (err,d['oracle_errf'][c],scales)):
            a=np.asarray(actual)
            assert np.isfinite(a).all() and np.isfinite(expect).all()
            # Missing sentinel must match exactly; its magnitude is never an error allowance.
            missing=np.abs(expect)>8e5
            np.testing.assert_array_equal(a[missing],expect[missing])
            assert np.all(np.abs(a-expect)[~missing] <= np.broadcast_to(32*eps*scale,a.shape)[~missing]),(name,c,float(np.abs(a-expect).max()))
