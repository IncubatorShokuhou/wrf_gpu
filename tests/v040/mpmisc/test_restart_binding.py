"""Selected MP carry survives native NetCDF; static GID follows WRF domains."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.precision import FERRIER_CARRY_LEAVES, GODDARD4ICE_CARRY_LEAVES
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.io.wrfrst_netcdf import write_wrfrst_state, read_wrfrst_state
from gpuwrf.physics.ferrier.binding import bind_ferrier_domain
from gpuwrf.runtime.operational_mode import OperationalNamelist
from test_ferrier_carry import grid


@pytest.mark.parametrize("option", [5, 15, 7])
def test_netcdf_restart_exact_selected_carries(tmp_path, option):
    domain = grid()
    state = State.zeros(domain, mp_physics=option).ensure_conditional_leaves(mp_physics=option)
    names = FERRIER_CARRY_LEAVES if option != 7 else ("qh", "hail_acc", *GODDARD4ICE_CARRY_LEAVES)
    updates = {}
    for i, name in enumerate(names):
        value = getattr(state, name)
        if value is not None:
            updates[name] = jnp.arange(value.size, dtype=jnp.float32).reshape(value.shape) * jnp.float32(.001) + jnp.float32(i + 1)
    state = state.replace(**updates)
    path = tmp_path / "wrfrst_d01"
    write_wrfrst_state(state, domain, {"mp_physics": option}, path,
                       valid_time="2026-07-25_18:00:18", run_start="2026-07-25_18:00:00", step_index=1)
    restored, metadata = read_wrfrst_state(path)
    for name in names:
        actual, expected = getattr(restored, name), getattr(state, name)
        if expected is None:
            assert actual is None
        else:
            assert np.asarray(actual).dtype == np.asarray(expected).dtype
            if name not in ("qh", "hail_acc"):
                assert np.asarray(actual).dtype == np.float32
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("option", [5, 15])
def test_ferrier_gid_is_static_pytree_aux_and_per_domain(option):
    domain = grid()
    nl = OperationalNamelist(grid=domain, tendencies=Tendencies.zeros(domain), metrics=domain.metrics,
                             dt_s=18., acoustic_substeps=4, mp_physics=option)
    root = bind_ferrier_domain(nl, {"domains": {"grid_id": [1, 9]}}, "d01")
    child = bind_ferrier_domain(nl, {"domains": {"grid_id": [1, 9]}}, "d02")
    assert root.ferrier_gid == 1 and child.ferrier_gid == 9
    leaves, tree = jax.tree.flatten(child)
    assert jax.tree.unflatten(tree, leaves).ferrier_gid == 9
    assert jax.tree.structure(root) != jax.tree.structure(child)
    from gpuwrf.runtime.aot_cheap_key import static_config_hash
    assert static_config_hash(root) != static_config_hash(child)
    with pytest.raises(ValueError, match="positive"):
        bind_ferrier_domain(nl, {"domains": {"grid_id": [0]}}, "d01")


def test_optional_hail_survives_surface_previous_step_mapping():
    from gpuwrf.runtime.noahmp_precipitation import precipitation_from_step
    from gpuwrf.diagnostics.precipitation import thompson_amounts
    p = {name: jnp.full((2, 2), value, jnp.float32) for name, value in
         (("rain", 1), ("snow", 2), ("graupel", 3), ("ice", 0), ("hail", 4))}
    forcing = precipitation_from_step(p, jnp.zeros((2, 2)), 2.)
    np.testing.assert_array_equal(forcing.prcpnonc, 5)
    np.testing.assert_array_equal(forcing.prcpsnow, 1)
    np.testing.assert_array_equal(forcing.prcpgrpl, 1.5)
    np.testing.assert_array_equal(forcing.prcphail, 2)
    total, snow = thompson_amounts(p)
    np.testing.assert_array_equal(total, 10)
    np.testing.assert_array_equal(snow, 2)


@pytest.mark.parametrize("subset", [None, ("PHYS_TOT",)])
def test_original_goddard_total_latent_history_is_retained(tmp_path, subset):
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, WRFOUT_VARIABLE_SPECS
    state, domain, namelist = synthetic_case()
    state.phys_tot = np.arange(state.qv.size, dtype=np.float32).reshape(state.qv.shape) / np.float32(2**20)
    payload = prepare_wrfout_payload(state, domain, namelist, tmp_path / "wrfout_d02",
        domain="d02", domain_authority=writer_authority(domain),
        valid_time="2026-07-25_18:00:18", run_start="2026-07-25_18:00:00", lead_hours=.005,
        variable_subset=subset)
    np.testing.assert_array_equal(payload.fields["PHYS_TOT"], state.phys_tot)
    assert WRFOUT_VARIABLE_SPECS["PHYS_TOT"].units == "K s-1"
