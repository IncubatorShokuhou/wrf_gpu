"""P3 caller optics versus pristine LW/SW blocks, and Registry history."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.p3.radiation import mp_re_active, optical_view
from gpuwrf.physics.rrtmg_mp_re import prepare_radiation_radii
from gpuwrf.io.wrfout_writer import (
    MICROPHYSICS_EXTRA_VARIABLES, WRFOUT_VARIABLE_SPECS, _optional_field_array,
    _STATE_SOURCE_ALIASES,
)
from gpuwrf.physics.p3.registry import SCHEMES, input_variables


@pytest.mark.parametrize('mp', SCHEMES)
def test_p3_radiation_flags_and_original_special_optics(mp):
    root=Path(__file__).resolve().parents[1]/'proofs/s2_p3/fixtures'
    with np.load(root/'radiation.npz') as f:
        x,expected=f['inputs'],f['sw'][:,:,1]
    qi= jnp.asarray(x[0]+x[1] if mp==52 else x[0])[None,:]
    state=SimpleNamespace(re_cloud=jnp.full((len(x[0]),1,1),10.e-6,jnp.float32),
        re_ice=jnp.asarray(x[2])[:,None,None],xland=jnp.ones((1,1)))
    nl=SimpleNamespace(mp_physics=mp,use_mp_re=1,ra_sw_physics=4,ra_lw_physics=4)
    assert mp_re_active(nl)
    assert not mp_re_active(SimpleNamespace(**{**vars(nl),'use_mp_re':0}))
    ice,snow,radii=optical_view(state,qi)
    prepared=prepare_radiation_radii(jnp.full_like(qi,250),jnp.ones_like(qi),
        radii['xland'],radii['re_cloud'],radii['re_ice'],radii['re_snow'])
    assert np.asarray(ice).tobytes()==expected[1].tobytes()
    assert np.asarray(snow).tobytes()==np.asarray(qi).tobytes()
    assert np.asarray(prepared.snow_um).tobytes()==expected[2].tobytes()
    assert np.asarray(prepared.ice_um).tobytes()==expected[3].tobytes()
    assert state.re_ice.shape==(len(x[0]),1,1)  # no State view mutation


@pytest.mark.parametrize('mp', SCHEMES)
def test_history_uses_only_selected_original_registry_particles(mp):
    source=Path('data/wrf_pristine/WRF/Registry/Registry.EM_COMMON').read_text()
    for name,variable in input_variables(mp).items():
        if name.startswith('mp_'):
            assert variable not in MICROPHYSICS_EXTRA_VARIABLES
            continue
        assert variable in MICROPHYSICS_EXTRA_VARIABLES
        spec=WRFOUT_VARIABLE_SPECS[variable]
        assert spec.description in source and spec.units in source
        value=np.full((3,2,1),1.25,np.float32)
        state=SimpleNamespace(**{name:value})
        actual=_optional_field_array(state,_STATE_SOURCE_ALIASES[variable],value.shape)
        assert actual.tobytes()==value.tobytes()
