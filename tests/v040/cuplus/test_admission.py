"""Explicit supported profiles and original-fatal boundary reporting."""
from pathlib import Path
import sys
import pytest
import jax.numpy as jnp
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from test_v022_camuw_pbl import _grid,_state,_namelist


@pytest.mark.parametrize('cu,bl,sf,lsm',[(11,5,5,4),(10,5,5,4),(7,9,5,4),(7,2,2,0)])
def test_cli_profiles_are_explicitly_admitted(cu,bl,sf,lsm):
    from gpuwrf.io.namelist_check import validate_operational_namelist
    from gpuwrf.io.scheme_catalog import classify_scheme,SupportStatus
    config={'physics':{'cu_physics':[cu],'mp_physics':[8],'bl_pbl_physics':[bl],
        'sf_sfclay_physics':[sf],'sf_surface_physics':[lsm],'ra_sw_physics':[4],'ra_lw_physics':[4]}}
    validate_operational_namelist(config)
    status=classify_scheme('cu_physics',cu)
    assert status.status is SupportStatus.IMPLEMENTED
    assert 'CPU-oracle-qualified' in status.reason and 'pending' in status.reason


@pytest.mark.parametrize('changes',[
    {'mp_physics':[6]}, {'bl_pbl_physics':[1]}, {'ra_sw_physics':[1]},
    {'kfeta_trigger':2}, {'shcu_aerosols_opt':[2]}, {'numbins':[2]}, {'thbinsize':[0.]},
    {'mindeepfreq':[1.1]}, {'cu_rad_feedback':[True]},
])
def test_unqualified_cup_combinations_fail_closed(changes):
    from gpuwrf.io.namelist_check import validate_operational_namelist,UnsupportedSchemeError
    config={'physics':{'cu_physics':[10],'mp_physics':[8],'bl_pbl_physics':[5],
        'sf_sfclay_physics':[5],'sf_surface_physics':[4],'ra_sw_physics':[4],'ra_lw_physics':[4],**changes}}
    with pytest.raises(UnsupportedSchemeError):validate_operational_namelist(config)


def test_original_cam_fatal_status_is_reported_even_when_state_is_finite():
    from gpuwrf.coupling.cuplus import initial_carry,CuPlusCarry
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.finite_state_guard import assert_state_finite_at_boundary
    grid=_grid();state=_state(grid);cu=initial_carry(state,7)
    cu=CuPlusCarry(cu.w0avg,dict(cu.held,ENTROPY_FAILED=cu.held['ENTROPY_FAILED'].at[1].set(1)))
    carry=initial_operational_carry(state).replace(cuplus=cu)
    with pytest.raises(RuntimeError,match='CAM-ZM entropy.*column=1'):
        assert_state_finite_at_boundary(state,domain='d01',step=3,physics_carry=carry,enabled=False)
