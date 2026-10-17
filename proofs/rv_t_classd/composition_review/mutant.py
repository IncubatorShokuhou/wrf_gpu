import sys
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/composition_review/snapshot')
sys.path[:0]=[str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'tests')]
import cpu_guard
import pytest
import jax.numpy as jnp
import gpuwrf.coupling.scan_adapters as sa
original=sa._with_morrison_side_outputs

def drop_precipitation(*args,**kwargs):
    out=original(*args,**kwargs)
    # Keep valid State/radar and a valid precipitation schema, but erase the
    # newly composed precipitation values. The caller must fail numerically.
    if isinstance(out,tuple) and len(out)==3:
        state,precip,context=out
        return state,{k:jnp.zeros_like(v) for k,v in precip.items()},context
    return out

sa._with_morrison_side_outputs=drop_precipitation
sys.exit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/v040/test_classd_morrison_precip_radar_compose.py')+'::test_one_morrison_call_feeds_land_precipitation_and_radar[10]',
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/pytest',
    '--junitxml',str(ROOT.parent/'RV22_mutant.xml')]))
