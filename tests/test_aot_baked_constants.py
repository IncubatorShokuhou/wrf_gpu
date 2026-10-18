"""Cache regressions use independent literal values, never a forecast oracle."""
from dataclasses import dataclass, replace
from functools import partial
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.runtime import aot_cheap_key as ck, aot_executable as ax, aot_precompile as ap
from gpuwrf.integration.init_kernels import _StaticKwargs


EMPTY = _StaticKwargs(())


def _factory(value):
    table = np.array([value], np.float32)
    @jax.jit
    def kernel(x):
        return x + table[0]
    return kernel


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _KeywordTable:
    value: float
    def tree_flatten(self):
        return (), self.value
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)


@jax.jit
def _keyword_kernel(x, *, table):
    return x + table.value


def test_closed_table_values_separate_keys_and_actual_warm_blobs(tmp_path):
    x = jnp.arange(4, dtype=jnp.float32)
    records = []
    for value in (.12, .20):
        fn = _factory(value)
        key = ck.cheap_key(fn, (x,), {}, EMPTY)
        assert key is not None
        lowered = fn.lower(x)
        # Lookup comes BEFORE compilation, as in the real warm path. With the
        # old key the Z call silently executes the already cached B constant.
        loaded, status = ap.load_domain_blob('table', tmp_path, cheap_key=key, return_status=True)
        if loaded is None:
            status = ap._serialize_domain_blob('table', lowered.compile(), tmp_path,
                        lowered=lowered, cheap_key=key, key_schema=ck.KEY_SCHEMA)
            assert status['aot_written'], status
        records.append((key, ax.hlo_sha256_from_lowered(lowered)))
        fn.clear_cache()
        loaded, status = ap.load_domain_blob('table', tmp_path, cheap_key=key, return_status=True)
        assert loaded is not None, status
        expected = np.arange(4, dtype=np.float32) + np.float32(value)
        np.testing.assert_array_equal(loaded(x), expected)
        np.testing.assert_array_equal(fn(x), expected)  # direct path control
    assert records[0][0] != records[1][0], 'different baked table values shared a key'
    assert records[0][1] != records[1][1]


@pytest.mark.parametrize('kind', ['positional', 'keyword', 'partial'])
def test_bound_default_values_change_identity(kind):
    def make(value):
        table = np.array([value], np.float32)
        if kind == 'positional':
            def fn(x, table=table):
                return x + table[0]
        elif kind == 'keyword':
            def fn(x, *, table=table):
                return x + table[0]
        else:
            def body(x, table):
                return x + table[0]
            fn = partial(body, table=table)
        return fn
    assert ck.fn_identity_hash(make(.12)) != ck.fn_identity_hash(make(.20))
    assert ck.fn_identity_hash(make(.12)) == ck.fn_identity_hash(make(.12))


def test_keyword_static_aux_changes_key_and_hlo_but_dynamic_values_do_not():
    x = jnp.ones(3, jnp.float32)
    keys, hlos = [], []
    for value in (.12, .20):
        kwargs = {'table': _KeywordTable(value)}
        keys.append(ck.cheap_key(_keyword_kernel, (x,), kwargs, EMPTY))
        hlos.append(ax.hlo_sha256_from_lowered(_keyword_kernel.lower(x, **kwargs)))
    assert all(keys) and len(set(keys)) == len(set(hlos)) == 2
    assert keys[0] == ck.cheap_key(_keyword_kernel, (x+1,), {'table':_KeywordTable(.12)}, EMPTY)


def test_second_positional_argument_is_not_blindly_assumed_to_be_namelist():
    x = jnp.ones(3, jnp.float32)
    @jax.jit
    def fn(x, table):
        return x + table.value
    keys = [ck.cheap_key(fn, (x, _KeywordTable(value)), {}, EMPTY) for value in (.12, .2)]
    assert all(keys) and len(set(keys)) == 2


def test_quarantine_keeps_first_witness_and_still_removes_new_stale_payloads(tmp_path):
    name, key = 'init_table', 'a'*64
    detail = {'actual': 'hlo-Z', 'stored': 'hlo-B'}
    assert ap.quarantine_cheap_key(name, key, tmp_path, reason='HLO mismatch', detail=detail)
    marker = ap._cheap_key_quarantine_path(name, key, tmp_path)
    original = marker.read_bytes()
    paths = ap._aot_blob_paths(name, tmp_path, cheap_key=key)
    for path in paths:
        path.write_bytes(b'stale')
    assert ap.quarantine_cheap_key(name, key, tmp_path, reason='previously-quarantined cheap_key')
    assert marker.read_bytes() == original
    assert json.loads(original)['detail'] == detail
    assert not any(path.exists() for path in paths)


def test_closed_namelist_children_keyed_and_dead_date_remains_inert():
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    nl = OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.,
                             radiation_static=jnp.array([.12], jnp.float32))
    def make(config):
        @jax.jit
        def fn(x):
            return x + config.radiation_static[0]
        return fn
    a, b = make(nl), make(replace(nl, radiation_static=jnp.array([.2], jnp.float32)))
    assert ck.fn_identity_hash(a) != ck.fn_identity_hash(b)
    assert ck.fn_identity_hash(a) == ck.fn_identity_hash(make(replace(nl, time_utc='2026-10-14_12:00:00')))


def test_static_holder_subclasses_use_contents_not_repr():
    from gpuwrf.runtime.operational_mode import _UrbanBundleHolder
    assert ck.canonical_digest(_UrbanBundleHolder(np.array([.12]))) == ck.canonical_digest(_UrbanBundleHolder(np.array([.12])))
    assert ck.canonical_digest(_UrbanBundleHolder(np.array([.12]))) != ck.canonical_digest(_UrbanBundleHolder(np.array([.2])))


def test_unreadable_baked_array_refuses_a_cheap_key():
    class Unreadable:
        shape, dtype = (3,), np.dtype('float32')
        def __array__(self, *args, **kwargs):
            raise ValueError('baked value is unavailable')
    table = Unreadable()
    def fn(x):
        return x + table
    assert ck.cheap_key(fn, (jnp.ones(3),), {}, EMPTY) is None


def _namelist(**kwargs):
    from gpuwrf.runtime.operational_mode import OperationalNamelist
    from gpuwrf.contracts.grid import GridSpec
    return OperationalNamelist(GridSpec.canary_3km_template(), None, None, 18.,
            mp_physics=0, ra_sw_physics=0, ra_lw_physics=0,
            sf_sfclay_physics=0, cu_physics=0, **kwargs)


def test_every_noah_parameter_field_changes_static_key():
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters
    params = load_noahmp_parameters()
    nl = _namelist(noahmp_energy_params=params)
    x = jnp.ones(1, jnp.float32)
    def fn(x, nl):
        return x
    baseline = ck.cheap_key(fn, (x, nl), {}, nl)
    assert baseline is not None
    for name in params._fields:
        value = getattr(params, name)
        if value is None:
            continue
        if isinstance(value, dict):
            changed = {**value, 'key-regression-witness': 1}
        elif hasattr(value, 'dtype'):
            host = np.asarray(value).copy()
            host.flat[0] = np.nextafter(host.flat[0], np.inf) if host.dtype.kind == 'f' else host.flat[0]+1
            changed = jnp.asarray(host)
        elif isinstance(value, str):
            changed = value + '-changed'
        else:
            changed = value + 1
        variant = replace(nl, noahmp_energy_params=params._replace(**{name: changed}))
        assert baseline != ck.cheap_key(fn, (x, variant), {}, variant), name


@pytest.mark.parametrize('selection', [dict(mp_physics=8), dict(mp_physics=28),
    dict(ra_sw_physics=4), dict(ra_lw_physics=4), dict(ra_lw_physics=1),
    dict(ra_sw_physics=3, ra_lw_physics=3), dict(ra_sw_physics=5), dict(mp_physics=18),
    dict(sf_sfclay_physics=5), dict(sf_sfclay_physics=4, bl_pbl_physics=4),
    dict(cu_physics=1), dict(cu_physics=10), dict(cu_physics=11),
    dict(cu_physics=94), dict(cu_physics=95),
    *[dict(mp_physics=mp) for mp in (50,51,52,53)]])
def test_external_table_snapshot_is_nonempty_and_stable(selection):
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    nl = replace(_namelist(), **selection)
    def fn(x, nl):
        return x
    a = baked_table_hash(fn, nl)
    assert len(a) == 64 and a == baked_table_hash(fn, nl)
    assert a != baked_table_hash(fn, _namelist())


def test_actual_rrtmg_bound_default_table_changes_init_key(monkeypatch):
    from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column, RRTMGSWColumnState
    from gpuwrf.integration.init_kernels import radiation_init_key
    shape = (2, 3)
    f = lambda v: jnp.full(shape, v, jnp.float64)
    operand = RRTMGSWColumnState(f(280), f(80000), f(.005), f(0), f(0), f(0), f(0), f(0),
                                jnp.full(2, .2), jnp.full(2, .5), f(100), f(1))
    before = radiation_init_key(solve_rrtmg_sw_column, (operand,), {})
    assert before is not None
    target = solve_rrtmg_sw_column.__wrapped__
    table = target.__defaults__[0]
    changed = table._replace(sw_cloud_liquid_extinction=table.sw_cloud_liquid_extinction.at[0].add(1.))
    with monkeypatch.context() as patch:
        patch.setattr(target, '__defaults__', (changed,))
        solve_rrtmg_sw_column.clear_cache()
        assert before != radiation_init_key(solve_rrtmg_sw_column, (operand,), {})
    solve_rrtmg_sw_column.clear_cache()


def test_actual_module_table_value_changes_step_key(monkeypatch):
    from gpuwrf.physics import thompson_column as tc
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    nl = replace(_namelist(), mp_physics=8)
    def fn(x, nl):
        return x
    before = baked_table_hash(fn, nl)
    table = tc.THOMPSON_TABLES
    with monkeypatch.context() as patch:
        patch.setattr(tc, 'THOMPSON_TABLES', table._replace(snow_sa=table.snow_sa.at[0].add(1.)))
        assert before != baked_table_hash(fn, nl)


def test_actual_cached_lw_cloud_table_changes_init_key(monkeypatch):
    from gpuwrf.physics import rrtmg_lw as lw
    from gpuwrf.integration.init_kernels import radiation_init_key
    f = lambda v: jnp.full((2, 3), v, jnp.float64)
    operand = lw.RRTMGLWColumnState(f(280), f(80000), f(.005), f(0), f(0), f(0), f(0), f(0),
                                    jnp.full(2, 280.), jnp.full(2, .98), f(100), f(1))
    before = radiation_init_key(lw.solve_rrtmg_lw_column, (operand,), {})
    hlo_before = ax.hlo_sha256_from_lowered(lw.solve_rrtmg_lw_column.lower(operand))
    assert before is not None
    table = lw._native_lw_cloud_tables()
    changed = table._replace(liquid=table.liquid+1.)
    with monkeypatch.context() as patch:
        patch.setattr(lw, '_native_lw_cloud_tables', lambda: changed)
        lw.solve_rrtmg_lw_column.clear_cache()
        assert hlo_before != ax.hlo_sha256_from_lowered(lw.solve_rrtmg_lw_column.lower(operand))
        assert before != radiation_init_key(lw.solve_rrtmg_lw_column, (operand,), {})
    lw.solve_rrtmg_lw_column.clear_cache()


def test_readonly_mapping_table_values_are_content_hashed():
    from types import MappingProxyType
    a = MappingProxyType({'table':np.arange(2048,dtype=np.float32)})
    changed = np.arange(2048,dtype=np.float32)
    changed[1024] += 1  # hidden by NumPy's truncated repr
    b = MappingProxyType({'table':changed})
    assert repr(a) == repr(b)
    assert ck.canonical_digest(a) != ck.canonical_digest(b)


# Each actual external reader needs a disposition. A new table loader must not
# silently enter the traced program without a holder, operand, or table hash.
_EXTERNAL_READERS = {
    'physics/cammg/tables.py': 'initialization.npz -> cached host _raw() -> get_globals traced constants; baked_table_hash keys cammg.tables._raw actual values for mp11',
    'physics/ntu/tables.py': 'data/aerosol_{r4,r8}.npz -> host_tables -> selected_tables; baked_table_hash keys ntu.aot_tables.resolved_bindings across native/portable/init providers, both kinds and all CCNTY banks for mp56',
    'kernels/_triton_kernel_call.py': 'tree_digest reads installed stock Triton/compiler bytes; baked_table_hash keys stock_compiler_identity for mp11, and stock_compile cache takes the same identity; compiler input, not a physics table',
    'coupling/urban_inputs.py': 'UrbanInputs traced child; content-keyed when closed over; seal IO is host-only',
    'coupling/urban_slucm_hook.py': 'UrbanBundle static holder',
    'physics/noahmp/tables.py': 'NoahMPStatic parameters in static holder',
    'physics/ruclsm.py': 'RucStaticBundle static holder',
    'physics/urban_bep_static.py': 'urban bundle static holder',
    'physics/urban_slucm.py': 'urban bundle static holder',
    'physics/wrf_clwrf_ghg.py': 'CLWRFGasClock traced operand; init gases traced column leaves',
    'physics/ra_flg.py': 'baked_table_hash ra_flg._host_tables actual NPZ values for option7/direct solver; collision regression tests/v040/radiation/test_s2_flg_gfdl_aot_keys.py',
    'physics/ra_gfdleta.py': 'baked_table_hash ra_gfdleta/_gfdleta_radtn._host_tables actual bound values for99/direct solver; same collision regression',
    'physics/gfdleta_co2.py': 'once-at-init bundled native source/data/compiler produce host CO2 arrays; actual captured provider+sigma/top in GfdlGrid.baked_tables/static aux; provider-content factory key and CO2 collision regression',
    **{f'physics/{name}.py':'baked_table_hash actual values' for name in
       ('cumulus_kfcup_tables','gfs_sas_shallow','nssl2mom/constants','p3/tables','ra_cam_common','ra_cam_sw','ra_goddard_common',
        'ra_lw_rrtm','rrtmg_lw','rrtmg_mp_re','rrtmg_tables','surface_layer',
        'sfclay_qnse','thompson_aero_tables','thompson_tables','wrf_cam_ozone',
        'ferrier/saturation','thompson_hail/native','thompson_hail/context')},
    'physics/wrf_tegen_aerosol.py': 'aer_opt=1: aot_baked_tables keys the Tegen asset loader + module tables (sw4 & aer_opt 1 only); aerodm/pina are traced namelist children; key regression tests/v025/w41_aerosol/test_tegen_wiring.py::test_aot_baked_table_key_names_the_tegen_asset_only_on_aer_opt1',
    'physics/jensen_ishmael/tables.py': 'baked_table_hash actual _host_tables values and _native.configuration for mp55; collision regressions tests/v040/test_jensen_ishmael_table_keys.py',
}


def test_every_external_model_reader_has_a_key_disposition():
    import ast
    from pathlib import Path
    import gpuwrf
    root = Path(gpuwrf.__file__).resolve().parent
    readers = set()
    for area in ('physics','kernels','coupling'):
        for path in (root/area).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node,ast.Call):
                    continue
                fn = node.func
                direct = isinstance(fn,ast.Name) and fn.id=='open'
                method = isinstance(fn,ast.Attribute) and fn.attr in {'read_text','read_bytes','open'}
                numpy = (isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Name)
                         and fn.value.id in {'np','numpy','_np'}
                         and fn.attr in {'load','fromfile','loadtxt','genfromtxt'})
                if direct or method or numpy:
                    readers.add(path.relative_to(root).as_posix())
    assert readers == set(_EXTERNAL_READERS), readers.symmetric_difference(_EXTERNAL_READERS)


def test_nssl_aliases_hash_actual_cached_constants(monkeypatch):
    from gpuwrf.physics.nssl2mom.constants import get_constants
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    def fn(x, nl):
        return x
    constants = get_constants('fp32')
    field = next(name for name,value in vars(constants).items() if isinstance(value,np.ndarray) and value.size)
    original = getattr(constants,field)
    changed = original.copy()
    changed.flat[0] += 1
    for mp in (17,18,22):
        nl = replace(_namelist(),mp_physics=mp)
        before = baked_table_hash(fn,nl)
        assert before != baked_table_hash(fn,_namelist())
        with monkeypatch.context() as patch:
            patch.setattr(constants,field,changed)
            assert before != baked_table_hash(fn,nl), (mp,field)


@pytest.mark.parametrize('mode', ['r4', 'r8'])
def test_gfs_shallow_cached_table_values_change_active_keys(monkeypatch, mode):
    from gpuwrf.physics.gfs_sas_shallow import _tables
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    def fn(x, nl):
        return x
    tables = _tables(mode)
    field = next(name for name, value in tables.items() if value.size)
    changed = tables[field].copy()
    changed.flat[0] += 1
    inactive = baked_table_hash(fn, _namelist())
    for cu in (94, 95):
        nl = replace(_namelist(), cu_physics=cu)
        before = baked_table_hash(fn, nl)
        with monkeypatch.context() as patch:
            patch.setitem(tables, field, changed)
            assert before != baked_table_hash(fn, nl), (mode, cu, field)
            assert inactive == baked_table_hash(fn, _namelist())


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('kind', ['table', 'surface_constants', 'pbl_constants'])
def test_qnse_cached_values_change_active_keys(monkeypatch, dtype, kind):
    from gpuwrf.physics import sfclay_qnse as surface, pbl_qnse as pbl
    from gpuwrf.runtime.aot_baked_tables import baked_table_hash
    def fn(x, nl):
        return x
    nl = replace(_namelist(), sf_sfclay_physics=4, bl_pbl_physics=4)
    before = baked_table_hash(fn, nl)
    inactive = baked_table_hash(fn, _namelist())
    if kind == 'table':
        table = surface._tables(dtype)[0]
        changed = table.copy()
        changed.flat[0] += 1
        original = surface._tables
        with monkeypatch.context() as patch:
            patch.setattr(surface, '_tables', lambda mode:
                          (changed, *original(mode)[1:]) if mode == dtype else original(mode))
            assert before != baked_table_hash(fn, nl), (dtype, kind)
            assert inactive == baked_table_hash(fn, _namelist())
    else:
        constants = (vars(surface._surface_constants(dtype)) if kind == 'surface_constants'
                     else pbl._constants(dtype))
        field = next(iter(constants))
        with monkeypatch.context() as patch:
            patch.setitem(constants, field, constants[field] + 1)
            assert before != baked_table_hash(fn, nl), (dtype, kind)
            assert inactive == baked_table_hash(fn, _namelist())


@pytest.mark.parametrize('family,mode', [('qnse', 'float32'), ('qnse', 'float64'),
                                        ('gfs', 'r4'), ('gfs', 'r8')])
def test_new_cached_providers_change_full_key_hlo_and_output(monkeypatch, family, mode):
    from gpuwrf.physics import sfclay_qnse as surface, gfs_sas_shallow as shallow
    if family == 'qnse':
        nl = replace(_namelist(), sf_sfclay_physics=4, bl_pbl_physics=4)
        original = surface._tables
        pm, ph, dzeta, maximum = original(mode)
        changed = (pm + np.dtype(mode).type(1), ph, dzeta, maximum)
        x = jnp.asarray(np.dtype(mode).type(-5) + np.dtype(mode).type(500) * dzeta)
    else:
        nl = replace(_namelist(), cu_physics=94)
        original = shallow._tables
        tables = original(mode)
        changed = dict(tables, tdp=tables['tdp'] + 1)
        x = jnp.asarray(1000., jnp.float64)

    @jax.jit
    def fn(x, nl):
        if family == 'qnse':
            pm, _, dzeta, _ = surface._tables(mode)
            return surface._interp_psi(jnp.asarray(pm), x, np.dtype(mode).type(-5), dzeta)
        return shallow._funcs(mode)[0](x)

    def snapshot():
        key = ck.cheap_key(fn, (x, nl), {}, nl)
        assert key is not None
        lowered = fn.lower(x, nl)
        value = np.asarray(lowered.compile()(x, nl))
        assert np.isfinite(value).all()
        return key, ax.hlo_sha256_from_lowered(lowered), value

    before = snapshot()
    try:
        with monkeypatch.context() as patch:
            module = surface if family == 'qnse' else shallow
            patch.setattr(module, '_tables', lambda selected:
                          changed if selected == mode else original(selected))
            fn.clear_cache()
            after = snapshot()
    finally:
        fn.clear_cache()
    assert before[0] != after[0], (family, mode, 'full cheap key')
    assert before[1] != after[1], (family, mode, 'lowered HLO')
    assert not np.array_equal(before[2], after[2]), (family, mode, 'actual provider output')


@pytest.mark.parametrize('cu', [10, 11])
def test_cup_mskf_all_bound_table_fields_change_full_key_hlo_and_output(monkeypatch, cu):
    from gpuwrf.physics import cumulus_kfcup_tables as tables
    nl = replace(_namelist(), cu_physics=cu)
    x = jnp.asarray(1, jnp.float32)
    for field in ('TTAB_R4', 'QSTAB_R4', 'THE0K_R4', 'ALU', 'RDPR', 'RDTHK', 'PLUTOP'):
        @jax.jit
        def fn(x, nl):
            return x + jnp.asarray(getattr(tables, field)).reshape(-1)[0]

        def snapshot():
            key = ck.cheap_key(fn, (x, nl), {}, nl)
            assert key is not None
            lowered = fn.lower(x, nl)
            value = np.asarray(lowered.compile()(x, nl))
            assert np.isfinite(value).all()
            return key, ax.hlo_sha256_from_lowered(lowered), value

        before = snapshot()
        changed = np.array(getattr(tables, field), copy=True)
        changed.flat[0] += 1
        try:
            with monkeypatch.context() as patch:
                patch.setattr(tables, field, changed)
                fn.clear_cache()
                after = snapshot()
        finally:
            fn.clear_cache()
        assert before[0] != after[0], (cu, field, 'full cheap key')
        assert before[1] != after[1], (cu, field, 'lowered HLO')
        assert not np.array_equal(before[2], after[2]), (cu, field, 'bound table output')
