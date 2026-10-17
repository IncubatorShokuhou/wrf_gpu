"""Restore history-only cadence for Jensen; one genuine assertion required."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'snapshot'
sys.path[:0]=[str(ROOT/'scripts/v025'),str(ROOT/'src'),str(ROOT/'tests')]
import cpu_guard
import pytest
import jax.numpy as jnp
import gpuwrf.diagnostics.online as online
original=online.radar_from_context
def history_only(previous,context,namelist,step,spec_zone):
    result=original(previous,context,namelist,step,spec_zone)
    if int(namelist.mp_physics)==55:
        return jnp.where(online.history_due(step,namelist.dt_s,namelist.diagnostic_history_interval),result,previous)
    return result
online.radar_from_context=history_only
sys.exit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/v040/test_jensen_radar.py')+'::test_operational_seed_and_every_call_update',
    '--basetemp','<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/pytest','--junitxml',str(ROOT.parent/'RV36_mutant.xml')]))
