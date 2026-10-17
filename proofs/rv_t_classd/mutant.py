import sys
from pathlib import Path
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/snapshot')
sys.path[:0]=[str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'tests')]
import cpu_guard
import gpuwrf.diagnostics.online as online
import jax.numpy as jnp
import pytest
online.history_due=lambda *args,**kwargs:jnp.bool_(True)
sys.exit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/v040/test_classd_morrison_radar.py'),
    '-k','caller_clamp_and_history_cadence','--basetemp',str(ROOT.parent/'pytest'),
    '--junitxml',str(ROOT.parent/'RV06_mutant.xml')]))
