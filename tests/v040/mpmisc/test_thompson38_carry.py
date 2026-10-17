"""Original mp38 held-emission caller/restart seams and scoped admission."""
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.precision import THOMPSON_HAIL_SURFACE_LEAVES
from gpuwrf.contracts.state import State
from gpuwrf.io.restart_schema import validate_held_state
from test_ferrier_carry import grid


def test_emissions_are_real_surface_arrays_for_38_only():
    for option in (8, 28, 7, 38):
        state = State.zeros(grid(), mp_physics=option)
        for name in THOMPSON_HAIL_SURFACE_LEAVES:
            value = getattr(state, name)
            if option == 38:
                assert value.shape == (2, 2) and value.dtype == jnp.float32
            else:
                assert value is None


@pytest.mark.parametrize("missing", ["nwfa", "nifa"])
def test_initial_presence_checks_are_independent_and_emissions_held(monkeypatch, missing):
    from gpuwrf.coupling.thompson_hail_adapter import thompson_hail_coldstart_init
    from gpuwrf.physics.thompson_hail import init
    state = State.zeros(grid(), mp_physics=38)
    state = state.replace(nwfa=jnp.ones_like(state.qv), nifa=jnp.full_like(state.qv, 2),
                          nwfa2d=jnp.full_like(state.xland, 123.25), nifa2d=jnp.full_like(state.xland, 7.5))
    state = state.replace(**{missing: jnp.zeros_like(state.qv)})
    def profiles(hgt):
        return np.full(hgt.shape, 100, np.float32), np.full(hgt.shape, 200, np.float32), np.full(hgt.shape[:-1], 3, np.float32)
    monkeypatch.setattr(init, "initial_profiles", profiles)
    result = thompson_hail_coldstart_init(state)
    np.testing.assert_array_equal(result.nwfa, 100 if missing == "nwfa" else 1)
    np.testing.assert_array_equal(result.nifa, 200 if missing == "nifa" else 2)
    np.testing.assert_array_equal(result.nwfa2d, 3 if missing == "nwfa" else 123.25)
    np.testing.assert_array_equal(result.nifa2d, 7.5)


def test_emission_follows_column_physics_and_dry_rho_unclamped_dz(monkeypatch):
    from gpuwrf.coupling.thompson_hail_adapter import thompson_hail_adapter
    from gpuwrf.physics.thompson_hail import column
    state = State.zeros(grid(), mp_physics=38)
    heights = jnp.broadcast_to(jnp.arange(9, dtype=jnp.float32)[:, None, None] * jnp.float32(.25), (9, 2, 2))
    state = state.replace(theta=jnp.full_like(state.theta, 300), p=jnp.full_like(state.p, 95000),
                          qv=jnp.full_like(state.qv, .01), ph=heights*jnp.float32(9.81), w=heights*8,
                          nwfa2d=jnp.full_like(state.xland, 2), nifa2d=jnp.full_like(state.xland, 3))
    state = state.replace(Ns=jnp.full_like(state.Ns, 7))
    captured = []
    amounts = (("rain",1),("snow",2),("graupel",3),("ice",4))
    def physics(col, dt):
        captured.append(col)
        return col.replace(nwfa=jnp.full_like(col.nwfa, 10), nifa=jnp.full_like(col.nifa, 20), Ns=jnp.full_like(col.Ns, 999)), {
            name:jnp.full_like(state.xland, value) for name,value in amounts}
    monkeypatch.setattr(column, "step_thompson_hail_column_with_precip", physics)
    result, precip = thompson_hail_adapter(state, 18, grid(), return_precipitation=True)
    col = captured[0]
    np.testing.assert_array_equal(col.w, np.moveaxis(np.asarray(state.w[:-1],np.float32),0,-1))
    np.testing.assert_allclose(col.dz, .25, rtol=0, atol=2e-7)
    rho = np.float32(.622)*np.asarray(col.p)/(np.float32(287.04)*np.asarray(col.T)*(np.asarray(col.qv)+np.float32(.622)))
    np.testing.assert_allclose(col.rho, rho, rtol=2e-7)
    np.testing.assert_array_equal(result.nwfa[0], 46)
    np.testing.assert_array_equal(result.nwfa[1:], 10)
    np.testing.assert_array_equal(result.nifa[0], 74)
    np.testing.assert_array_equal(result.nifa[1:], 20)
    np.testing.assert_array_equal(result.Ns, 7)
    np.testing.assert_array_equal(precip["total"], 10)
    np.testing.assert_array_equal(result.thompson_sr, np.float32(.9))
    np.testing.assert_array_equal(result.thompson_rainnc, np.float32(10))
    np.testing.assert_array_equal(result.thompson_snownc, np.float32(6))
    assert result.re_cloud is result.re_ice is result.re_snow is None
    # At this REAL exponent, each source +1 rounds away. Pre-summing the
    # increments, or using the legacy DOUBLE accumulator helper, changes totals.
    amounts = (("rain",1),("snow",1),("graupel",0),("ice",1))
    state = state.replace(thompson_rainnc=jnp.full((2,2),2**24,jnp.float32),
                          thompson_snownc=jnp.full((2,2),2**24,jnp.float32))
    result = thompson_hail_adapter(state,18,grid())
    np.testing.assert_array_equal(result.thompson_rainnc,np.float32(2**24))
    np.testing.assert_array_equal(result.thompson_snownc,np.float32(2**24))


def test_38_missing_or_malformed_restart_carries_are_refused():
    required = ("qv", "qvolg", "nwfa", "nifa", "Nc", "Ng", "Ni", "Nr", *THOMPSON_HAIL_SURFACE_LEAVES)
    fields = {n:np.zeros((2,2) if n in THOMPSON_HAIL_SURFACE_LEAVES else (8,2,2),np.float32) for n in required}
    validate_held_state(fields, {"mp_physics":38,"use_mp_re":0})
    for name in required[1:]:
        with pytest.raises(ValueError, match="missing Thompson-hail.*E78"):
            validate_held_state({k:v for k,v in fields.items() if k!=name}, {"mp_physics":38,"use_mp_re":0})
        for bad in (fields[name][..., :1], fields[name].astype(np.int32)):
            with pytest.raises(ValueError, match=f"{name} shape/dtype differs.*E78"):
                validate_held_state({**fields,name:bad}, {"mp_physics":38,"use_mp_re":0})


@pytest.mark.parametrize("fmt", ["pickle", "netcdf", "checkpoint"])
def test_38_surface_emissions_survive_exact_restart(tmp_path, fmt):
    from types import SimpleNamespace
    from gpuwrf.io import restart as rp, wrfrst_netcdf as nc
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint
    domain = grid()
    state = State.zeros(domain, mp_physics=38).replace(
        nwfa2d=jnp.full((2,2),123.25,jnp.float32), nifa2d=jnp.full((2,2),7.5,jnp.float32), thompson_sr=jnp.full((2,2),.375,jnp.float32),
        thompson_rainnc=jnp.full((2,2), 123456.75,jnp.float32), thompson_snownc=jnp.full((2,2),17.25,jnp.float32))
    nl = OperationalNamelist(grid=domain, metrics=domain.metrics, tendencies=None, mp_physics=38)
    carry = initial_operational_carry(state)
    path = tmp_path / fmt
    if fmt == "pickle":
        rp.write_restart(carry, nl, domain, 2, path)
        after = rp.read_restart(path)[0].state
    elif fmt == "checkpoint":
        write_checkpoint(state, nl, domain, 2, path)
        after = read_checkpoint(path)[0]
    else:
        nc.write_wrfrst_carry(carry, domain, nl, path, valid_time="2026-07-25_18:00:36",
                             run_start="2026-07-25_18:00:00", step_index=2)
        after = nc.read_wrfrst_carry(path)[0].state
    for name in THOMPSON_HAIL_SURFACE_LEAVES:
        expected, actual = np.asarray(getattr(state,name)), np.asarray(getattr(after,name))
        assert actual.dtype == expected.dtype == np.float32
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("subset", [None, ("QNWFA2D", "QNIFA2D", "RAINNC", "SNOWNC")])
def test_held_surface_emissions_are_original_history_fields(tmp_path, subset):
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    state, domain, nl = synthetic_case()
    state.nwfa2d = np.full(state.qv.shape[-2:],123.25,np.float32)
    state.nifa2d = np.full(state.qv.shape[-2:],7.5,np.float32)
    state.thompson_rainnc = np.full(state.qv.shape[-2:],123456.75,np.float32)
    state.thompson_snownc = np.full(state.qv.shape[-2:],17.25,np.float32)
    out = prepare_wrfout_payload(state,domain,nl,tmp_path / "history",domain="d02",
        domain_authority=writer_authority(domain),valid_time="2026-07-25_18:00:18",
        run_start="2026-07-25_18:00:00",lead_hours=.005,variable_subset=subset)
    np.testing.assert_array_equal(out.fields["QNWFA2D"],state.nwfa2d)
    np.testing.assert_array_equal(out.fields["QNIFA2D"],state.nifa2d)
    np.testing.assert_array_equal(out.fields["RAINNC"],state.thompson_rainnc)
    np.testing.assert_array_equal(out.fields["SNOWNC"],state.thompson_snownc)


def test_qualified_descriptor_and_nondefault_aerosol_boundary_remain_explicit():
    from gpuwrf.contracts.physics_interfaces import scheme_step_spec
    from gpuwrf.io.namelist_check import _mpmisc_admission_failures
    from gpuwrf.io.scheme_catalog import classify_scheme, SupportStatus
    spec = scheme_step_spec("microphysics",38)
    assert "CPU-oracle-qualified" in spec.notes
    assert "Ns" not in spec.writes_state
    assert classify_scheme("mp_physics",38).status is SupportStatus.IMPLEMENTED
    failures = _mpmisc_admission_failures({"physics":{"mp_physics":38,"aer_init_opt":1}})
    assert any(f.key == "aer_init_opt" and f.supported_values == (0,) for f in failures)


@pytest.mark.parametrize("option", [5,7,15,38])
@pytest.mark.parametrize("pbl,expected", [(16,("pek_adv","pep_adv")), (17,("tke_adv","diss_adv","tpe_adv"))])
def test_new_mp_preserves_main_epsilon_scalar_diffusion(option,pbl,expected):
    from gpuwrf.runtime.operational_mode import OperationalNamelist, _diffused_scalar_species
    domain = grid()
    nl = OperationalNamelist(grid=domain,metrics=domain.metrics,tendencies=None,
                            mp_physics=option,bl_pbl_physics=pbl)
    species = _diffused_scalar_species(nl)
    assert set(expected) <= set(species)
    assert len(species) == len(set(species))
