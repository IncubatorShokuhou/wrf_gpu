"""YSU consumes WRF FM/FHH; raw neutral psi must not enter the PBL solve."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import BCMetadata, DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State
from gpuwrf.coupling.scan_adapters import ysu_pbl_adapter

ROOT = Path(__file__).resolve().parents[1]


def column_state():
    """Original WRF sounding 1, with neutral BR and active surface heat flux."""
    data = json.loads((ROOT / "proofs/v060/savepoints/ysu_case_1.json").read_text())
    c, s = data["columns"], data["scalars"]
    nz = len(c["U"])
    eta = jnp.linspace(1., 0., nz+1)
    projection = Projection("lambert", 28.3, -16.4, 3000., 3000., 1, 1)
    terrain = TerrainProvenance("pristine-YSU-sounding1", "pristine-YSU-sounding1", (1, 1), "m", "native", 0., True)
    grid = GridSpec(projection, terrain, VerticalCoord("hybrid_eta", nz, 5000., eta),
        BCMetadata("ideal", (), 1, "linear", True), eta, jnp.zeros((1, 1)),
        metrics=DycoreMetrics.flat(ny=1, nx=1, nz=nz, eta_levels=eta, top_pressure_pa=5000.))
    col = lambda k: jnp.asarray(c[k], jnp.float32).reshape(nz, 1, 1)
    dz = col("DZ")
    state = State.zeros(grid).replace(
        u=jnp.broadcast_to(col("U"), (nz, 1, 2)), v=jnp.broadcast_to(col("V"), (nz, 2, 1)),
        # State.theta is theta_m (use_theta_m=1); the caller hands YSU WRF's dry th_phy.
        theta=col("T")/col("PI")*(1+jnp.float32(461.6/287.)*col("QV")), qv=col("QV"), p=col("P"),
        ph=9.81*jnp.concatenate([jnp.zeros((1, 1, 1)), jnp.cumsum(dz, axis=0)]),
        roughness_m=jnp.full((1, 1), s["ZNT"]), ustar=jnp.full((1, 1), s["UST"]),
        hfx=jnp.full((1, 1), s["HFX"]), qfx=jnp.full((1, 1), s["QFX"]),
        t_skin=jnp.full((1, 1), 300.), xland=jnp.ones((1, 1)), rhosfc=jnp.ones((1, 1)))
    forcing = {k: jnp.full((1, 1), s[v]) for k, v in
        (("fm", "PSIM"), ("fh", "PSIH"), ("wspd", "WSPD"), ("u10", "U10"), ("v10", "V10"))}
    forcing["br"] = jnp.zeros((1, 1))
    return state, grid, forcing


def test_active_neutral_ysu_matches_pristine_caller(monkeypatch):
    from gpuwrf.coupling import scan_adapters as adapters
    state, _, forcing = column_state()
    # No new surface solve is permitted when the selected surface driver already
    # produced these fields. Deliberately distinct raw-psi values are irrelevant.
    def forbidden(*args, **kwargs):
        raise AssertionError("YSU recomputed a surface solve instead of consuming FM/FHH")
    monkeypatch.setattr(adapters, "surface_layer_with_diagnostics", forbidden)
    monkeypatch.setattr(adapters, "sfclay_revised_mm5_adapter", forbidden)
    _, output = jax.jit(lambda st, sf: ysu_pbl_adapter(st, 60., surface_forcing=sf, _with_sources=True))(state, forcing)
    expected = json.loads((ROOT / "proofs/t_c02nan/savepoints/active_neutral.json").read_text())
    names = {"u": "RUBLTEN", "v": "RVBLTEN", "theta": "RTHBLTEN", "qv": "RQVBLTEN", "exch_h": "EXCH_H", "exch_m": "EXCH_M"}
    for key, original in names.items():
        actual = np.asarray(output[key]).reshape(-1)
        reference = np.asarray(expected["columns"][original])
        assert np.isfinite(actual).all() and np.isfinite(reference).all(), key
        # Existing frozen YSU operator bounds (test_v060_pbl_ysu.py).
        abs_tol = 2.e-4 if key.startswith("exch_") else 2.e-6
        np.testing.assert_allclose(actual, reference, atol=abs_tol, rtol=2.e-3, err_msg=key)


def test_operational_ysu_reuses_selected_surface_solve(monkeypatch):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.coupling import scan_adapters as adapters
    from test_v013_operational_smoke import _grid, _base_state, _namelist
    grid = _grid(nz=12, ny=3, nx=3)
    state = _base_state(grid)
    namelist = _namelist(grid, mp_physics=0, cu_physics=0, bl_pbl_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=0, use_noahmp=False,
        ra_lw_physics=0, ra_sw_physics=0)
    from gpuwrf.coupling import surface_handoff
    paid = []
    surface = surface_handoff.revised_mm5_surface_step
    def surface_once(*args, **kwargs):
        result = surface(*args, **kwargs)
        paid.append(result[1])
        return result
    monkeypatch.setattr(surface_handoff, 'revised_mm5_surface_step', surface_once)
    def pbl(st, dt, grid, **kwargs):
        assert len(paid) == 1
        assert kwargs.get("surface_forcing") is paid[0], "YSU did not receive selected surface forcing"
        zero = jnp.zeros((grid.ny*grid.nx,grid.nz),jnp.float32)
        return st, {key:zero for key in ('u','v','theta','qv')}
    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS, 1, pbl)
    from gpuwrf.runtime.operational_state import initial_operational_carry
    om._physics_step_forcing(initial_operational_carry(state), namelist, 0.,
        run_radiation=False, first_timestep=True)
    assert len(paid) == 1


@pytest.mark.parametrize('scheme,kernel,resistances', [
    (1,'ysu_columns',True),(11,'shinhong_columns',True),
    (3,'gfs_columns',False),(12,'gbm_columns',False),(99,'mrf_columns',False),
    (7,'acm2_columns',False),(8,'boulac_columns',False),
])
def test_each_sf1_pbl_consumes_selected_contract(monkeypatch,scheme,kernel,resistances):
    from gpuwrf.coupling import scan_adapters as adapters
    from gpuwrf.coupling.surface_handoff import revised_mm5_surface_step
    from test_v013_operational_smoke import _grid,_base_state
    grid = _grid(nz=12,ny=3,nx=3)
    state, sf, _ = revised_mm5_surface_step(_base_state(grid),grid)
    # LSM updates HFX/QFX after the surface solve; PBL must use these values.
    state = state.replace(hfx=jnp.full_like(state.hfx,123.),qfx=jnp.full_like(state.qfx,2.e-5))
    seen = []
    def columns(*args,**kwargs):
        seen.append(kwargs)
        zero = jnp.zeros((grid.ny*grid.nx,grid.nz),jnp.float32)
        out = {key:zero for key in ('u','v','theta','qv','qc','qi')}
        out.update(tke=jnp.ones_like(zero),pblh=jnp.ones((grid.ny*grid.nx,),jnp.float32),
                   ust=kwargs['ust'])
        if scheme == 12:
            # GBM KETHL is a REAL32 (column, level) output retained as EXCH_TKE.
            out['kethl'] = zero
        if scheme in (3,99):
            out.update({key:zero for key in ('rublten','rvblten','rthblten','rqvblten','rqcblten','rqiblten')})
        return out
    monkeypatch.setattr(adapters,kernel,columns)
    if scheme == 3:  # with surface forcing: the WRF BL_GFS call (t-cfix-sfc F13)
        monkeypatch.setattr(adapters,'bl_gfs_wrf_columns',columns)
    def forbidden(*args,**kwargs):
        raise AssertionError('PBL recomputed surface forcing')
    monkeypatch.setattr(adapters,'surface_layer_with_diagnostics',forbidden)
    adapters.PBL_SCAN_ADAPTERS[scheme](state,18.,grid,surface_forcing=sf,_with_sources=True)
    assert len(seen)==1
    got = seen[0]
    np.testing.assert_array_equal(np.asarray(got['hfx']),np.full(9,np.float32(123.)))
    np.testing.assert_array_equal(np.asarray(got['qfx']),np.full(9,np.float32(2.e-5)))
    np.testing.assert_array_equal(np.asarray(got['ust']),np.asarray(state.ustar).reshape(-1))
    if scheme not in (7,8):
        np.testing.assert_array_equal(np.asarray(got['psim']),np.asarray(sf['fm' if resistances else 'psim']).reshape(-1))
        np.testing.assert_array_equal(np.asarray(got['psih']),np.asarray(sf['fh' if resistances else 'psih']).reshape(-1))
        np.testing.assert_array_equal(np.asarray(got['wspd']),np.asarray(sf['wspd']).reshape(-1))
    if scheme in (3,12,99):
        np.testing.assert_array_equal(np.asarray(got['gz1oz0']),np.asarray(sf['gz1oz0']).reshape(-1))
        np.testing.assert_array_equal(np.asarray(got['tsk']),np.asarray(sf['tsk']).reshape(-1))


@pytest.mark.parametrize('scheme',[1,7,8,11,12,3,99])
def test_selected_sf1_pbl_rates_enter_rk_once(monkeypatch,scheme):
    from gpuwrf.runtime import operational_mode as om
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from test_v013_operational_smoke import _grid,_base_state,_namelist
    grid=_grid(nz=12,ny=3,nx=3);state=_base_state(grid)
    ns=_namelist(grid,dt_s=18.,mp_physics=0,cu_physics=0,bl_pbl_physics=scheme,
        sf_sfclay_physics=1,sf_surface_physics=0,use_noahmp=False,
        ra_lw_physics=0,ra_sw_physics=0,rad_rk_tendf=1)
    original=json.loads((ROOT/'proofs/t_c02nan/savepoints/active_neutral.json').read_text())
    values={key:np.float32(original['columns'][name][0]) for key,name in
        (('theta','RTHBLTEN'),('qv','RQVBLTEN'),('u','RUBLTEN'),('v','RVBLTEN'))}
    rates={key:jnp.full((9,12),value,jnp.float32) for key,value in values.items()}
    def pbl(st,dt,grid,**kwargs):
        assert kwargs['surface_forcing'] is not None and kwargs['_with_sources']
        # A deliberately observable Euler update must be removed at the RK seam.
        return st.replace(theta=st.theta+22.,qv=st.qv+.1,u=st.u+5.,v=st.v-4.,
                          pblh=jnp.full_like(st.pblh,750.)),rates
    monkeypatch.setitem(om.PBL_SCAN_ADAPTERS,scheme,pbl)
    monkeypatch.setattr(om,'_phys_tend_rk_wrf_enabled',lambda:True)
    forcing=om._physics_step_forcing(initial_operational_carry(state),ns,0.,
        run_radiation=False,first_timestep=True)
    for key in ('u','v','theta','qv'):
        np.testing.assert_array_equal(np.asarray(getattr(forcing.state,key)),np.asarray(getattr(state,key)))
    np.testing.assert_array_equal(np.asarray(forcing.state.pblh),np.full((3,3),750.))
    # Original WRF calculate_phy_tend + conv_t_tendf_to_moist equations, using
    # pristine YSU rates; no State-difference estimate or second Euler increment.
    mass=np.asarray(ns.metrics.c1h)[:,None,None]*np.asarray(state.mu_total)[None]+np.asarray(ns.metrics.c2h)[:,None,None]
    rvrd=461.6/287.;q=np.asarray(state.qv);thm=np.asarray(state.theta)
    expected=mass*((1+rvrd*q)*float(values['theta'])+rvrd*thm/(1+rvrd*q)*float(values['qv']))
    np.testing.assert_allclose(np.asarray(forcing.dry_tendencies.t_tendf),expected,rtol=8*np.finfo(np.float32).eps,atol=0.)
    np.testing.assert_allclose(np.asarray(forcing.dry_tendencies.moist_tendf['qv']),mass*float(values['qv']),rtol=8*np.finfo(np.float32).eps,atol=0.)


def test_gbm_writes_reconstructed_ust_to_state(monkeypatch):
    """Pristine GBM2D :539/:835 updates UST on this retained real c08 column."""
    from gpuwrf.coupling import scan_adapters as adapters
    from test_v013_operational_smoke import _grid, _base_state
    data = json.loads((ROOT/'proofs/t_c02nan/savepoints/gbm_c08_col0.json').read_text())
    d = data['input']; nz = len(d['u'])
    grid = _grid(nz=nz, ny=1, nx=1)
    col = lambda key: jnp.asarray(d[key], jnp.float32).reshape(1, nz)
    flat = lambda key: jnp.asarray([d[key]], jnp.float32)
    state = _base_state(grid).replace(_cast=False, ustar=flat('ust').reshape(1, 1),
        qc=col('qc').reshape(nz, 1, 1), qke=col('tke').reshape(nz, 1, 1))
    forcing = dict(ncol=1, nz=nz, ny=1, nx=1)
    forcing.update({key+'_cols':col(name) for key, name in
        [('u','u'),('v','v'),('T','t'),('qv','qv'),('p','p'),('pii','pii'),('dz','dz')]})
    forcing.update({key:flat(key) for key in
        ('psfc','znt','ust','hfx','qfx','tsk','gz1oz0','wspd','br','psim','psih','xland')})
    # Preserve the real kernel and adapter; isolate only the input assembly seam.
    monkeypatch.setattr(adapters, '_pbl_surface_forcing', lambda *args, **kwargs: forcing)
    with jax.enable_x64(False):
        result, rates = adapters.gbm_pbl_adapter(state, d['dt'], grid,
            surface_forcing={}, _with_sources=True)
    expected = np.asarray([[data['reference']['UST_REAL4']]], np.float32)
    assert not np.array_equal(np.asarray(state.ustar), expected)
    np.testing.assert_array_equal(np.asarray(rates['ust']).reshape(1, 1), expected)
    np.testing.assert_array_equal(np.asarray(result.ustar), expected)
    assert result.ustar.dtype == state.ustar.dtype
