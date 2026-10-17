"""CAM_INIT estblf, using the existing independently qualified WRF host table."""
import numpy as np
import jax.numpy as jnp
from gpuwrf.physics.bl_camuw import ESTBL,_TMIN,_TMAX


def estblf(td,*,xp=jnp,_globals=None):
    e=xp.maximum(xp.minimum(td,np.float64(_TMAX)),np.float64(_TMIN))
    ai=xp.floor(e-np.float64(_TMIN))
    i=ai.astype(xp.int32)
    tab=_globals.get('__estbl_ref') if _globals and '__estbl_ref' in _globals else xp.asarray(ESTBL,dtype=xp.float64)
    return (np.float64(_TMIN)+ai-e+np.float64(1))*tab[i]-(np.float64(_TMIN)+ai-e)*tab[i+1]
