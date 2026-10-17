"""Source-owned caller/unit/Registry tests; numerical oracles are separate."""
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.state import State
from gpuwrf.contracts.precision import GODDARD4ICE_CARRY_LEAVES
from gpuwrf.io.namelist_check import validate_operational_namelist, UnsupportedSchemeError
from test_ferrier_carry import grid


@pytest.mark.parametrize("option,expected,numbers", [
    (5, ("qv", "qc", "qr", "qi", "qt"), ()),
    (15, ("qv", "qc", "qr", "qi", "qrimef"), ()),
    (7, ("qv", "qc", "qr", "qi", "qs", "qg", "qh"), ()),
    (38, ("qv", "qc", "qr", "qi", "qs", "qg", "qvolg", "Ng", "Nc", "nwfa", "nifa"), ("Ni", "Nr")),
])
def test_exact_wrf_registry_transport_members(option, expected, numbers):
    from gpuwrf.runtime.operational_mode import _advected_scalar_species, _nested_number_scalar_species, _diffused_scalar_species
    nl = SimpleNamespace(mp_physics=option)
    assert _advected_scalar_species(nl) == expected
    assert _nested_number_scalar_species(nl) == numbers
    assert _diffused_scalar_species(nl) == expected + numbers


@pytest.mark.parametrize("option", [5, 15])
def test_ferrier_unqualified_cumulus_refused_before_compute(option):
    with pytest.raises(UnsupportedSchemeError, match="F_QS=false"):
        validate_operational_namelist({"physics": {"mp_physics": option, "cu_physics": 1}})


@pytest.mark.parametrize("option,key", [(5, "have_bcs_scalar"), (15, "have_bcs_scalar"), (7, "have_bcs_moist")])
def test_extra_species_unqualified_specified_records_refused_before_compute(option, key):
    with pytest.raises(UnsupportedSchemeError, match="extra-species specified records"):
        validate_operational_namelist({"physics": {"mp_physics": option}, "bdy_control": {key: True}})


def test_goddard_original_bottom_w_unclamped_dz_and_radius_units(monkeypatch):
    from gpuwrf.coupling.goddard4ice_adapter import goddard4ice_adapter
    from gpuwrf.physics.goddard4ice import native
    from gpuwrf.physics.rrtmg_mp_re import mp_re_active
    domain = grid()
    state = State.zeros(domain, mp_physics=7)
    heights = jnp.broadcast_to(jnp.arange(9, dtype=jnp.float32)[:, None, None] * jnp.float32(.25), (9, 2, 2))
    state = state.replace(theta=jnp.full_like(state.theta, 300), p=jnp.full_like(state.p, 95000),
        ph=heights * jnp.float32(9.81), w=heights * jnp.float32(8), xland=jnp.ones_like(state.xland))
    captured = []

    def original_interface(*args, **kwargs):
        captured.append((args, kwargs))
        output = dict(zip(("theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"), args[:8]))
        for index, name in enumerate(GODDARD4ICE_CARRY_LEAVES):
            if name != "goddard_sr":
                output[name] = jnp.full_like(args[0], index + 1)
        output["phys_tot"] = jnp.zeros_like(args[0])
        output.update({name: jnp.zeros_like(state.xland) for name in
            ("rainnc", "snownc", "graupelnc", "hailnc", "sr", "rain_other")})
        return output

    monkeypatch.setattr(native, "native_column", original_interface)
    result, precip = goddard4ice_adapter(state, 18., domain, step_index=9, return_precipitation=True)
    args, kwargs = captured[0]
    # solve_em passes W_2 directly; phy_prep uses PH/g face differences.
    np.testing.assert_array_equal(args[13], np.asarray(state.w[:-1], np.float32))
    np.testing.assert_allclose(args[11], .25, rtol=0, atol=2e-7)
    assert kwargs["itimestep"] == 9
    assert mp_re_active(use_mp_re=1, mp_physics=7, ra_lw_physics=4, ra_sw_physics=4)
    for name in ("re_cloud", "re_ice", "re_snow"):
        np.testing.assert_array_equal(getattr(result, name),
            np.asarray(getattr(result, name + "_gsfc"), np.float32) * np.float32(1e-6))
    assert "total" in precip and "hail" in precip
