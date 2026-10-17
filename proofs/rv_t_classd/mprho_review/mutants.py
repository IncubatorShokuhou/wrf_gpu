"""In-process numeric mutants; immutable snapshot source is never edited."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'snapshot'
sys.path[:0]=[str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'tests')]
import cpu_guard
import pytest
import jax.numpy as jnp
import gpuwrf.coupling.scan_adapters as sa

mode=sys.argv[1]
original=sa._wrf_moist_prep_den_delz
def changed(state, grid, dtype):
    rho, dz=original(state,grid,dtype)
    if mode=='moist':
        return rho*(jnp.asarray(1,rho.dtype)+jnp.asarray(state.qv,rho.dtype)),dz
    assert mode=='dzclamp'
    return rho,jnp.maximum(dz,jnp.asarray(1,dz.dtype))
sa._wrf_moist_prep_den_delz=changed
target=str(ROOT/'tests/test_t_classd_mp_rho_callers.py')
if mode=='dzclamp': target+='::test_nssl_dz_has_no_one_metre_clamp'
sys.exit(pytest.main(['-q','-p','no:cacheprovider',target,'--basetemp',
    '<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/pytest','--junitxml',str(ROOT.parent/f'RV28_{mode}.xml')]))
