"""Re-run only the compiled caller gate with explicitly verified REAL32 input."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'snapshot'
sys.path[:0]=[str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'tests/v040')]
import cpu_guard
import pytest
import jax
import jax.numpy as jnp
import test_jensen_radar as tests
from gpuwrf.physics.jensen_ishmael import adapter

old_state=tests._state
def real_state(*args,**kwargs):
    fixture,grid,state=old_state(*args,**kwargs)
    state=jax.tree.map(lambda x:x.astype(jnp.float32) if hasattr(x,'dtype') and jnp.issubdtype(x.dtype,jnp.floating) else x,state)
    assert state.theta.dtype==state.qv.dtype==state.ph_total.dtype==jnp.float32
    return fixture,grid,state
tests._state=real_state
real_adapter=adapter.jensen_adapter
def checked(state,*args,**kwargs):
    assert state.theta.dtype==state.qv.dtype==jnp.float32
    out=real_adapter(state,*args,**kwargs)
    assert out[0].theta.dtype==out[0].qv.dtype==out[2].dtype==jnp.float32
    return out
adapter.jensen_adapter=checked
sys.exit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/v040/test_jensen_radar.py')+'::test_compiled_step_writes_the_call_reflectivity',
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/pytest','--junitxml',str(ROOT.parent/'RV39_REAL_caller.xml')]))
