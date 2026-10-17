"""t-wsmdz: WSM3/5/6 + WDM5/6/7 held radii are restart carry once physics_init activates them (E78)."""
import pickle

import jax.numpy as jnp
import pytest

from gpuwrf.io.restart import read_restart, write_restart
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
from gpuwrf.physics import rrtmg_mp_re as R
from test_mp_re_activation import _case, _exact_finite

NAMES = ("re_cloud", "re_ice", "re_snow")


@pytest.mark.parametrize("mp", (3, 4, 6, 14, 16, 26))
def test_wsm_radius_carry_roundtrips_and_pre_radius_checkpoint_refuses(monkeypatch, tmp_path, mp):
    grid, nml, carry = _case(monkeypatch, mp_physics=mp)
    assert R.mp_re_active(nml)
    assert all(getattr(carry.state, n) is not None for n in NAMES)
    carry = carry.replace(state=carry.state.replace(
        **{n: jnp.full_like(carry.state.qc, v) for n, v in zip(NAMES, (17e-6, 80e-6, 200e-6))}))
    pkl = tmp_path / "carry.pkl"
    write_restart(carry, nml, grid, 2, pkl)
    _exact_finite(carry, read_restart(pkl)[0])
    nc = tmp_path / "carry.nc"
    write_wrfrst_carry(carry, grid, nml, nc, valid_time="2026-02-28_00:01:48",
                       run_start="2026-02-28_00:00:00", step_index=2)
    _exact_finite(carry, read_wrfrst_carry(nc)[0])
    # A checkpoint written before WSM radii were carried must refuse loudly.
    with pkl.open("rb") as stream:
        payload = pickle.load(stream)
    payload["carry"]["state_field_order"].remove("re_cloud")
    del payload["carry"]["state_fields"]["re_cloud"]
    with pkl.open("wb") as stream:
        pickle.dump(payload, stream)
    with pytest.raises(ValueError, match="E78"):
        read_restart(pkl)
