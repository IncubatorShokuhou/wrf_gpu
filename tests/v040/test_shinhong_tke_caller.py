"""WRF Shin-Hong conditional shared TKE, including the operational caller."""
import dataclasses
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    spec = importlib.util.spec_from_file_location(
        'shin_caller_fixture', ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    grid = f._grid(nz=8, ny=2, nx=2)
    return f, grid


@pytest.mark.parametrize('mp', [8, 11])
@pytest.mark.parametrize('sf', [0, 1])
@pytest.mark.parametrize('diag', [0, 1])
def test_shinhong_mode_reaches_actual_pbl_and_post_rk_cammg(mp, sf, diag, monkeypatch):
    """The source conditional is module_bl_shinhong.F:1478,1601-1612.

    A producer stub tests the actual caller, not Shin-Hong numerical parity.
    Its default 1 deliberately models the kernel's old implicit default, so
    omitted binding fails the held-mode assertion rather than fixture setup.
    """
    from gpuwrf.coupling import scan_adapters as scan
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    f, grid = fixture()
    state = f._state(grid).ensure_conditional_leaves(mp_physics=mp)
    held = jnp.arange(state.qke.size, dtype=jnp.float32).reshape(state.qke.shape)*.001+.002
    state = state.replace(qke=held if mp != 11 else jnp.full_like(held, 31.))
    if mp == 11:
        state = state.replace(cam_tke_pbl=jnp.concatenate((held, jnp.full_like(held[:1], 73.))))
    energy = held+2.
    cols = lambda value: jnp.moveaxis(value, 0, -1).reshape(4, 8)
    seen = []
    def producer(*args, shinhong_tke_diag=1, **kwargs):
        seen.append((shinhong_tke_diag, args[8], kwargs))
        zero = jnp.zeros((4, 8), jnp.float32)
        return dict(u=zero, v=zero, theta=zero, qv=zero, qc=zero, qi=zero,
                    tke=cols(energy) if shinhong_tke_diag == 1 else args[8],
                    exch_h=cols(energy+10), el_pbl=zero)  # kernel return set (bl_shinhong.py)
    monkeypatch.setattr(scan, 'shinhong_columns', producer)
    monkeypatch.setattr(op, 'surface_adapter', lambda s, *a, **kw: s)
    if sf == 1:
        from gpuwrf.coupling import surface_handoff
        def paid_surface(s, *args, **kwargs):
            z = jnp.zeros_like(s.t_skin); one = jnp.ones_like(z)
            return s, dict(fm=one, fh=one, psim=z, psih=z, br=z, wspd=5*one,
                u10=2*one, v10=3*one, gz1oz0=4*one, rmol=z, tsk=s.t_skin), None
        monkeypatch.setattr(surface_handoff, 'revised_mm5_surface_step', paid_surface)
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF', '1')
    monkeypatch.setenv('GPUWRF_MICROPHYSICS_WRF_ORDER', '1')
    nl = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0),
        mp_physics=mp, bl_pbl_physics=11, cu_physics=0, sf_surface_physics=0,
        sf_sfclay_physics=sf, use_noahmp=False, run_physics=True, rad_rk_tendf=1)
    # Also runs against the old dataclass without turning the old-wire witness
    # into a constructor/setup error. The binding/treedef has its own gate.
    object.__setattr__(nl, 'shinhong_tke_diag', diag)
    forcing = op._physics_step_forcing(initial_operational_carry(state), nl, 18.,
                                      run_radiation=False, first_timestep=False)
    post = op._apply_physics_non_dry_updates(state, state, forcing.state)
    expected = energy if diag == 1 else held
    if mp == 11:
        from gpuwrf.physics.cammg.adapter import operands
        actual, _ = operands(post, grid)
        np.testing.assert_array_equal(actual['tke_pbl'][:, :-1], cols(expected),
                                      err_msg='WRF conditional shared TKE caller')
        np.testing.assert_array_equal(post.cam_tke_pbl[-1:], 73.)
    np.testing.assert_array_equal(post.qke, expected, err_msg='WRF held/diagnosed TKE')
    assert len(seen) == 1 and seen[0][0] == diag
    np.testing.assert_array_equal(seen[0][1], cols(held))
    # module_pbl_driver.F:1301-1329: QC3D/QI3D and CORF=f reach Shin-Hong.
    kw = seen[0][2]
    np.testing.assert_array_equal(kw['qc'], cols(state.qc).astype(np.float32))
    np.testing.assert_array_equal(kw['qi'], cols(state.qi).astype(np.float32))
    np.testing.assert_array_equal(kw['corf'], np.asarray(grid.metrics.f, np.float32).reshape(-1))


@pytest.mark.parametrize('mp', [8, 11])
def test_shinhong_cold_init_is_original_epsq2l_over_two(mp):
    from gpuwrf.runtime import operational_mode as op
    f, grid = fixture()
    state = f._state(grid).ensure_conditional_leaves(mp_physics=mp)
    if mp == 11:
        state = state.replace(cam_tke_pbl=jnp.full_like(state.cam_tke_pbl, 73.))
    nl = dataclasses.replace(f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0),
        mp_physics=mp, bl_pbl_physics=11, cu_physics=0, sf_surface_physics=0,
        sf_sfclay_physics=0, use_noahmp=False)
    carry = op._initial_carry_for_run(state, nl)
    # Original shinhonginit: EPSQ2L=.01; .not.restart only, kts:kte.
    np.testing.assert_array_equal(carry.state.qke, np.float32(.005))
    if mp == 11:
        np.testing.assert_array_equal(carry.state.cam_tke_pbl[:-1], np.float32(.005))
        np.testing.assert_array_equal(carry.state.cam_tke_pbl[-1:], 73.)


@pytest.mark.parametrize('diag', [0, 1])
def test_shinhong_registry_binding_and_jit_key(diag):
    from gpuwrf.io.namelist_binding import bind_namelist_options
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    f, grid = fixture()
    base = f._namelist(grid, ra_sw_physics=0, ra_lw_physics=0)
    nl = bind_namelist_options(base, {'physics': {
        'bl_pbl_physics': 11, 'shinhong_tke_diag': [1-diag, diag]}}, 'd02')
    assert nl.shinhong_tke_diag == diag
    rebuilt = jax.tree.unflatten(jax.tree.structure(nl), jax.tree.leaves(nl))
    assert rebuilt.shinhong_tke_diag == diag
    other = dataclasses.replace(nl, shinhong_tke_diag=1-diag)
    assert jax.tree.structure(nl) != jax.tree.structure(other)
    assert OperationalNamelist(grid, None, None, 18.).shinhong_tke_diag == 0


@pytest.mark.parametrize('diag', [0, 1])
def test_shinhong_restart_preserves_mode_and_shared_tke(diag, tmp_path):
    from gpuwrf.runtime.checkpoint import write_checkpoint, read_checkpoint
    f, grid = fixture()
    state = f._state(grid).ensure_conditional_leaves(mp_physics=11)
    held = jnp.arange(state.cam_tke_pbl.size,dtype=jnp.float32).reshape(state.cam_tke_pbl.shape)/128
    state = state.replace(cam_tke_pbl=held)
    nl = dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),
        mp_physics=11,bl_pbl_physics=11,shinhong_tke_diag=diag,cu_physics=0,
        sf_sfclay_physics=0,sf_surface_physics=0,use_noahmp=False)
    path=tmp_path/'restart.pkl'
    write_checkpoint(state,nl,grid,2,path)
    restored,saved,_,step=read_checkpoint(path)
    assert saved.shinhong_tke_diag==diag and step==2
    assert np.asarray(restored.cam_tke_pbl).tobytes()==np.asarray(held).tobytes()
