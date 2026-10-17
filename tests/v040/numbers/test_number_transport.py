"""Missing two-moment transport: pristine REAL4 operators + production routing."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'proofs/s1_numbers'), str(ROOT/'tests/v025/b_core')]
import qnn_oracle
import scalar_oracle
import pd_scalar_oracle
import update_scalar_oracle
from gpuwrf.coupling import boundary_apply as bd
from gpuwrf.dynamics.flux_advection import CoupledVelocities
from gpuwrf.runtime import operational_mode as op

CORE = ('qv', 'qc', 'qr', 'qi', 'qs', 'qg')
EXTRAS = {10: ('Ns', 'Ng'), 14: ('Nc', 'Nr', 'Nn'), 16: ('Nc', 'Nr', 'Nn'),
          26: ('qh', 'Nc', 'Nr', 'Nn')}
CFG = bd.BoundaryConfig(have_bcs_moist=False, have_bcs_scalar=False)


@pytest.fixture(autouse=True)
def cpu_state_constructors(monkeypatch):
    from gpuwrf.contracts import state as sc
    monkeypatch.setattr(sc, '_gpu_device', lambda: jax.devices('cpu')[0])


class Fields(SimpleNamespace):
    def replace(self, **changes):
        changes.pop('_cast', None)
        return Fields(**(vars(self) | changes))


def nml(mp, moist=1, scalar=1):
    return SimpleNamespace(mp_physics=mp, moist_adv_opt=moist, scalar_adv_opt=scalar)


def test_fixture_manifest_matches_tracked_arrays():
    p = Path(__file__).parent
    manifest = json.loads((p/'swiss_transport.json').read_text())
    assert hashlib.sha256((p/'swiss_transport.npz').read_bytes()).hexdigest() == manifest['files']['swiss_transport.npz']['sha256']
    with np.load(p/'swiss_transport.npz') as arrays:
        assert set(arrays.files) == set(manifest['variables'])
        for name, a in arrays.items():
            spec = manifest['variables'][name]
            assert list(a.shape) == spec['shape'] and str(a.dtype) == spec['dtype']
            assert np.isfinite(a).all() and spec['units']


@pytest.mark.parametrize('mp', EXTRAS)
def test_species_and_root_boundary_routes(mp):
    names = op._advected_scalar_species(nml(mp))
    assert names == CORE + EXTRAS[mp]
    assert len(names) == len(set(names))
    relaxed, flow = bd.root_scalar_rk1_split(CFG, names, **op._root_flow_only(nml(mp)))
    assert relaxed == ('qv',)
    assert set(flow) == set(names)-{'qv'}
    assert set(op._diffused_scalar_species(nml(mp))) >= set(EXTRAS[mp])
    if mp != 10:
        assert 'Ni' not in op._diffused_scalar_species(nml(mp))


@pytest.mark.parametrize('mp', EXTRAS)
@pytest.mark.parametrize('moist,scalar', [(0, 1), (1, 0), (1, 1)])
def test_scalar_option_selection_and_single_stacked_call(monkeypatch, mp, moist, scalar):
    calls = []
    def spy(state, ns, **kwargs):
        calls.append((kwargs['species'], kwargs['advection_opt']))
        return tuple(kwargs['species'])
    monkeypatch.setattr(op, '_scalar_transport_coupled_tendencies', spy)
    species = op._advected_scalar_species(nml(mp))
    result = op._moisture_coupled_tendencies(None, nml(mp, moist, scalar), rk_step=3, step_origin=None)
    assert result == species
    expected = [(tuple(s for s in species if s not in op._SCALAR_FAMILY_SPECIES), moist),
                (tuple(s for s in species if s in op._SCALAR_FAMILY_SPECIES), scalar)]
    assert calls == ([(species, moist)] if moist == scalar else expected)


def test_qnn_precedes_have_bcs_scalar():
    cfg = replace(CFG, have_bcs_scalar=True)
    assert bd.root_scalar_rk1_split(cfg, ('Nn',), flow_only=('Nn',)) == ((), ('Nn',))
    with pytest.raises(NotImplementedError):
        bd.root_scalar_rk1_split(cfg, ('Nc',), flow_only=('Nc',))


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    p = tmp_path_factory.mktemp('pristine_numbers')
    # Extracted pristine bodies compile in the lane directory. No shared WRF
    # executable/build is run, and no reference source/output is modified.
    return {n: module.build(p/n) for n, module in (
        ('bc', qnn_oracle), ('plain', scalar_oracle), ('pd', pd_scalar_oracle), ('update', update_scalar_oracle))}


@pytest.fixture(scope='module')
def operands():
    with np.load(Path(__file__).with_name('swiss_transport.npz')) as f:
        a = dict(f)
    f32 = np.float32
    mu = (a['MU']+a['MUB']).astype(f32)
    my, mx = mu.shape
    muu = .5*(np.pad(mu, ((0, 0), (1, 0)), mode='edge')+np.pad(mu, ((0, 0), (0, 1)), mode='edge'))
    muv = .5*(np.pad(mu, ((1, 0), (0, 0)), mode='edge')+np.pad(mu, ((0, 1), (0, 0)), mode='edge'))
    ru = ((a['C1H'][:, None, None]*muu+a['C2H'][:, None, None])*a['U']/a['MAPFAC_U']).astype(f32)
    rv = ((a['C1H'][:, None, None]*muv+a['C2H'][:, None, None])*a['V']/a['MAPFAC_V']).astype(f32)
    # Active, explicitly constructed vertical transport; horizontal velocities,
    # mass, map and eta coefficients are real Swiss operands.
    rom = np.broadcast_to((100*np.sin(np.linspace(0, np.pi, len(a['C1H'])+1)))[:, None, None],
                          (len(a['C1H'])+1, my, mx)).copy().astype(f32)
    rom[[0, -1]] = 0
    v = CoupledVelocities(jnp.asarray(ru[:, :, :-1]), jnp.asarray(rv[:, :-1, :]), jnp.asarray(rom),
                         msftx=jnp.asarray(a['MAPFAC_M']), msfux=jnp.asarray(a['MAPFAC_U'][:, :-1]),
                         msfvy=jnp.asarray(a['MAPFAC_V'][:-1]), msfvx=jnp.asarray(a['MAPFAC_V'][:-1]),
                         specified=True, ru_full=jnp.asarray(ru), rv_full=jnp.asarray(rv))
    metrics = SimpleNamespace(**{k: jnp.asarray(a[n]) for k, n in {
        'c1h': 'C1H', 'c2h': 'C2H', 'fnm': 'FNM', 'fnp': 'FNP', 'rdnw': 'RDNW', 'msfty': 'MAPFAC_M'}.items()})
    base = a['QVAPOR']/a['QVAPOR'].max()
    fields = {n: jnp.asarray((scale*base**(1+i%3)*(1+.3*np.sin(np.arange(mx)/2)[None, None, :])).astype(f32))
              for i, (n, scale) in enumerate({'Ns': 2e5, 'Ng': 4e3, 'Nc': 1.5e8, 'Nr': 3e4,
                                            'Nn': 2e8, 'qh': 3e-4}.items())}
    return a, Fields(mu_total=jnp.asarray(mu), **fields), metrics, v


def assert_machine_band(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    # Pre-registered REAL4 bound: 32 ulps at the field maximum, for independent
    # XLA CPU vs noncontracting gfortran operator composition.
    ulp = np.spacing(np.float32(np.max(np.abs(reference))))
    assert np.max(np.abs(actual.astype(np.float64)-reference)) <= 32*ulp


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_gate_rejects_nonfinite(bad):
    with pytest.raises(AssertionError):
        assert_machine_band(np.array([bad]), np.array([1.]))
    with pytest.raises(AssertionError):
        assert_machine_band(np.array([1.]), np.array([bad]))


@pytest.mark.parametrize('spec', [1, 2, 3])
@pytest.mark.parametrize('ccn', [0., 1e8, 2.7e8])
def test_qnn_boundary_is_exact_pristine(libraries, operands, spec, ccn):
    a, state, m, vel = operands
    q = np.asarray(state.Nn)
    # Mixed inflow, outflow and exact-zero normal velocity on every side.
    u, v = np.asarray(vel.ru_full).copy(), np.asarray(vel.rv_full).copy()
    u[..., ::3] = 0; u[..., 1::3] *= -1
    v[..., ::3, :] = 0; v[..., 1::3, :] *= -1
    ref = qnn_oracle.evaluate(libraries['bc'], q, u, v, spec=spec, qnn=True, ccn=ccn)
    got = jax.jit(lambda q, u, v: bd.flow_dep_bdy_qnn(q, u, v, replace(CFG, spec_zone=spec),
                                                   ccn_conc=ccn))(q, u, v)
    assert np.array_equal(np.asarray(got), ref)
    if ccn:
        wrong = bd.flow_dep_bdy(jnp.asarray(q), jnp.asarray(u), jnp.asarray(v), replace(CFG, spec_zone=spec))
        assert not np.array_equal(np.asarray(wrong), ref)  # genuine zero-inflow mutant


@pytest.mark.parametrize('mp', [10, 16])
@pytest.mark.parametrize('opt', [0, 1])
def test_scalar_advection_rk_boundary_chain_vs_pristine(libraries, operands, mp, opt):
    a, state, metrics, vel = operands
    ns = nml(mp, opt, opt)
    ns.metrics = metrics
    ns.grid = SimpleNamespace(projection=SimpleNamespace(dx_m=float(a['dx']), dy_m=float(a['dy'])))
    ns.dt_s, ns.rk_order = 18., 3
    names = EXTRAS[mp]
    original, wrf = state, {n: np.asarray(getattr(state, n)) for n in names}
    rdx = 1/float(a['dx'])
    for stage, dt in enumerate((6., 9., 18.), 1):
        adv = op._scalar_transport_coupled_tendencies(state, ns, rk_step=stage, step_origin=original,
                                                     species=names, advection_opt=opt, transport_velocities=vel)
        raw = {}
        for name in names:
            arr = dict(q=wrf[name], qold=np.asarray(getattr(original, name)),
                       ru=np.asarray(vel.ru_full), rv=np.asarray(vel.rv_full), rom=np.asarray(vel.rom),
                       mu=np.asarray(original.mu_total), map=a['MAPFAC_M'],
                       mut=np.asarray(original.mu_total), mub=a['MUB'], muold=a['MU'],
                       mx=a['MAPFAC_M'], my=a['MAPFAC_M'], ux=a['MAPFAC_U'], uy=a['MAPFAC_U'],
                       vx=a['MAPFAC_V'], vy=a['MAPFAC_V'], fzm=a['FNM'], fzp=a['FNP'],
                       rdzw=a['RDNW'], c1=a['C1H'], c2=a['C2H'])
            raw[name] = (pd_scalar_oracle.evaluate(libraries['pd'], arr, rdx, rdx, 18.) if opt and stage == 3
                         else scalar_oracle.evaluate(libraries['plain'], arr, rdx))
            upd = dict(s1=np.asarray(getattr(original, name)), tend=raw[name], sct=np.zeros_like(raw[name]),
                       mx=a['MAPFAC_M'], my=a['MAPFAC_M'], c1=a['C1H'], c2=a['C2H'],
                       muold=a['MU'], munew=a['MU'], mub=a['MUB'])
            q = update_scalar_oracle.evaluate(libraries['update'], upd, stage, dt, 1)
            wrf[name] = qnn_oracle.evaluate(libraries['bc'], q, np.asarray(vel.ru_full), np.asarray(vel.rv_full),
                                            qnn=name == 'Nn', ccn=2.7e8)
        tend = op._root_scalar_stage_tendencies(adv, names, state, CFG, metrics.msfty, bounded=True)
        state = op._apply_moisture_large_step(state, original, q_tendencies=tend, dt_rk=dt,
                                              metrics=metrics, species=names)
        state = op._root_flow_dep(state, names, vel.ru_full, vel.rv_full, CFG, ccn_conc=2.7e8)
        for name in names:
            try:
                assert_machine_band(getattr(state, name), wrf[name])
            except AssertionError as exc:
                raise AssertionError(f'mp={mp} option={opt} RK={stage} species={name}: {exc}') from exc
    # E39: deleting transport must fail the numerical gate, including interior.
    for name in names:
        assert not np.array_equal(np.asarray(getattr(state, name))[:, 2:-2, 2:-2],
                                  np.asarray(getattr(original, name))[:, 2:-2, 2:-2])


@pytest.mark.parametrize('mp', EXTRAS)
def test_restart_preserves_numbers_and_rejects_missing(tmp_path, mp):
    from gpuwrf.io.restart_schema import validate_held_state
    from gpuwrf.runtime.checkpoint import read_checkpoint, write_checkpoint
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    grid = build_flat_grid(nx=6, ny=5, nz=8, dx_m=3000.)
    state = build_neutral_state(grid).ensure_conditional_leaves(mp_physics=mp)
    names = ('Ni', 'Nr', 'Ns', 'Ng') if mp == 10 else ('Nc', 'Nr', 'Nn')
    state = state.replace(**{n: jnp.full_like(getattr(state, n), 123+i) for i, n in enumerate(names)})
    ns = replace(op.OperationalNamelist.from_grid(grid, metrics=grid.metrics), mp_physics=mp, ccn_conc=2.7e8)
    target = write_checkpoint(state, ns, grid, 7, tmp_path/'numbers.pkl')
    recovered, rebound, _, step = read_checkpoint(target)
    assert step == 7 and rebound.ccn_conc == ns.ccn_conc
    for name in names:
        assert np.array_equal(np.asarray(getattr(recovered, name)), np.asarray(getattr(state, name)))
        fields = {n: getattr(state, n) for n in state.active_field_names() if n != name}
        with pytest.raises(ValueError, match='prognostic number scalars'):
            validate_held_state(fields, ns)


def test_ccn_binding_and_jit_key():
    from gpuwrf.io.namelist_binding import bind_namelist_options, namelist_honour_report
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid
    grid = build_flat_grid(nx=6, ny=5, nz=8, dx_m=3000.)
    ns = op.OperationalNamelist.from_grid(grid, metrics=grid.metrics)
    cfg = {'physics': {'ccn_conc': 2.7e8}}
    new = bind_namelist_options(ns, cfg, 'd01')
    assert new.ccn_conc == 2.7e8
    assert jax.tree.structure(ns) != jax.tree.structure(new)
    restored = jax.tree.unflatten(jax.tree.structure(new), jax.tree.leaves(new))
    assert restored.ccn_conc == new.ccn_conc
    assert not any('ccn_conc' in p for p in namelist_honour_report(cfg, ('d01',))[0])


@pytest.mark.parametrize('mp', [14, 16, 26])
def test_wdm_initializes_ccn_only_on_first_own_step(monkeypatch, mp):
    from gpuwrf.validation.moving_nest_testbed import build_flat_grid, build_neutral_state
    grid = build_flat_grid(nx=6, ny=5, nz=8, dx_m=3000.)
    state = build_neutral_state(grid).ensure_conditional_leaves(mp_physics=mp)
    state = state.replace(Nn=jnp.full_like(state.Nn, 2e8))
    ns = replace(op.OperationalNamelist.from_grid(grid, metrics=grid.metrics), mp_physics=mp,
                 run_physics=True, ccn_conc=2.7e8)
    seen = []
    def adapter(s, *args):
        seen.append(np.asarray(s.Nn))
        return s
    monkeypatch.setitem(op.MP_SCAN_ADAPTERS, mp, adapter)
    first = op._apply_post_rk_microphysics(state, ns, step_index=1)
    resumed = op._apply_post_rk_microphysics(state, ns, step_index=7)
    assert np.all(np.asarray(first.Nn) == np.float32(ns.ccn_conc))
    assert np.array_equal(np.asarray(resumed.Nn), np.asarray(state.Nn))
    assert len(seen) == 2


def test_morrison_real32_column_traces_without_dtype_promotion():
    from gpuwrf.physics.microphysics_morrison import morrison_run, polysvp
    a = jnp.ones((1, 8), jnp.float32)
    q = a*1e-5
    args = (a*270, a*.003, q, q, q, q, q, a*1e4, a*1e4, a*1e4, a*1e4,
            a*.9, a*80000, a*300, a*0)
    result = jax.eval_shape(lambda *x: morrison_run(*x, 18.), *args)
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(result))
    assert jax.eval_shape(lambda t: polysvp(t, 0), a).dtype == jnp.float32
    assert jax.eval_shape(lambda t: polysvp(t, 1), a).dtype == jnp.float32


@pytest.mark.parametrize('mp', [8, 26])
def test_wrfinput_number_fields_reach_native_state(tmp_path, monkeypatch, mp):
    import re
    import shutil
    from netCDF4 import Dataset
    from gpuwrf.integration import d02_replay as replay
    source = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu')
    assert (source/'wrfinput_d01').exists(), 'retained Swiss reference required for input seam gate'
    case = tmp_path/'case'
    case.mkdir()
    shutil.copy2(source/'wrfinput_d01', case/'wrfinput_d01')
    text, count = re.subn(r'^\s*mp_physics\s*=.*$', f' mp_physics = {mp},',
                          (source/'namelist.input').read_text(), flags=re.M | re.I)
    assert count == 1
    (case/'namelist.input').write_text(text)
    expected = {}
    with Dataset(case/'wrfinput_d01', 'a') as ds:
        shape = ds['QVAPOR'].shape
        for i, (leaf, name) in enumerate({'Ns': 'QNSNOW', 'Ng': 'QNGRAUPEL', 'Nc': 'QNCLOUD', 'Nn': 'QNCCN'}.items()):
            q = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape)+i+1)*128
            var = ds[name] if name in ds.variables else ds.createVariable(name, 'f4', ds['QVAPOR'].dimensions)
            var[:] = q
            expected[leaf] = q[0]
        hail = np.full(shape, np.float32(1e-4))
        var = ds['QHAIL'] if 'QHAIL' in ds.variables else ds.createVariable('QHAIL', 'f4', ds['QVAPOR'].dimensions)
        var[:] = hail
    # Number input reads are independent of MYNN's expensive cold-start solve.
    monkeypatch.setattr(replay, '_wrf_mynn_coldstart_qke', lambda qke, **kw: (qke, False))
    loaded = replay.build_replay_case(case, domain='d01', standalone=True, load_lateral_boundaries=False)
    for leaf, ref in expected.items():
        assert np.array_equal(np.asarray(getattr(loaded.state, leaf)), ref), leaf
    if mp == 26:
        assert np.array_equal(np.asarray(loaded.state.qh), hail[0])
    else:
        assert loaded.state.qh is None


@pytest.mark.parametrize('mp', EXTRAS)
def test_single_domain_only_and_catalog(mp):
    from gpuwrf.io.namelist_check import NotOperationallyWiredError, refuse_single_domain_mp_nested
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme
    cfg = {'physics': {'mp_physics': [mp, mp]}}
    refuse_single_domain_mp_nested(cfg, 1)
    with pytest.raises(NotOperationallyWiredError, match='single domains only'):
        refuse_single_domain_mp_nested(cfg, 2)
    scheme = classify_scheme('mp_physics', mp)
    assert scheme.status is SupportStatus.IMPLEMENTED
    assert 'CPU-oracle-qualified; GPU/coupled-forecast qualification pending' in scheme.reason
