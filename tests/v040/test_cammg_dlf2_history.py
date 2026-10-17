"""CAM-MG (mp_physics=11) DLF2 history: WRF's Registry zero in every admitted configuration.

WRF (pristine 4.7.1): Registry/registry.cam:73 `state real dlf2 ikj misc 1 - rh "DLF2" ... "kg/kg/s"`,
allocated by cammgmpscheme (Registry.EM_COMMON:3031) with initial_data_value=0. The only writer is
module_shcu_camuwshcu_driver.F:400, reached only under shcu_physics==CAMUWSHCUSCHEME
(module_shallowcu_driver.F:545); CAMMGMP reads it intent(in) (module_mp_cammgmp_driver.F:184).
The port refuses shcu_physics!=0, so DLF2 is +0 for the whole run.
Original CPU proof: c89 (mp11, bl9, shcu0), every cell of all 25 frames +0 bits.
"""
import json
from pathlib import Path
import re

import numpy as np
import pytest

from gpuwrf.io.producer_history import inactive_names

ROOT = Path(__file__).resolve().parents[2]
PROOF = json.loads((ROOT / 'proofs/t_cammgws/DLF2_CPU_ALL_FRAMES.json').read_text())
AUDIT = json.loads((ROOT / 'proofs/t_cammgws/DLF2_SOURCE_AUDIT.json').read_text())


def _lookup(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else default


def test_original_cpu_all_frames_hold_positive_zero():
    assert PROOF['frame_count'] == 25 and PROOF['all_frames_positive_zero']
    assert PROOF['global_attributes']['MP_PHYSICS'] == '11' and PROOF['global_attributes']['SHCU_PHYSICS'] == '0'
    for frame in PROOF['frames']:
        assert frame['bits'] == 0 and frame['all_equal_bits'] and frame['nonzero_bits'] == 0
        assert frame['shape'] == [1, 44, 42, 42]


def test_pristine_has_a_single_grid_writer_behind_the_shallow_cu_branch():
    writes = [s for s in AUDIT['sites'] if re.match(r'dlf2\s*\(\s*i\s*,\s*k\s*,\s*j\s*\)\s*=', s['text'], re.I)]
    assert [(s['file'], s['line']) for s in writes] == [('phys/module_shcu_camuwshcu_driver.F', 400)]
    reader = [s for s in AUDIT['sites'] if s['file'] == 'phys/module_mp_cammgmp_driver.F' and 'intent(in)' in s['text']]
    assert reader and 'dlf2_3d' in reader[0]['text']
    pristine = ROOT / 'data/wrf_pristine/WRF'
    if pristine.is_dir():  # bind the audit to the actual pristine sources
        import hashlib
        for name, sha in AUDIT['files'].items():
            assert hashlib.sha256((pristine / name).read_bytes()).hexdigest() == sha, name


@pytest.mark.parametrize('full', (False, True))
def test_writer_emits_original_bits_and_metadata(tmp_path, full):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    state, grid, nl = synthetic_case()
    nl.update(mp_physics=11, shcu_physics=0)
    auth = writer_authority(grid, 'd01')
    names = ('DLF2',)
    p = prepare_wrfout_payload(state, grid, nl, tmp_path / 'c89', domain='d01', domain_authority=auth,
                               valid_time='2023-01-15T01:00:00', run_start='2023-01-15T00:00:00', lead_hours=1.,
                               full_variable_set=full, variable_subset=names)
    write_prepared_wrfout(p, expected_domain='d01', expected_domain_authority=auth, variable_subset=names)
    ref = PROOF['field']
    with Dataset(p.target) as ds:
        v = ds['DLF2']
        assert list(v.dimensions) == ref['dimensions'] and str(v.dtype) == ref['dtype']
        for key in ('description', 'units', 'stagger', 'MemoryOrder', 'FieldType', 'coordinates'):
            assert str(v.getncattr(key)) == ref[key], key
        assert np.all(np.asarray(v[:], np.float32).view(np.uint32) == 0)


def test_only_mp11_without_uw_shallow_convection_receives_the_constant():
    assert 'DLF2' in inactive_names({'mp_physics': 11, 'shcu_physics': 0}, _lookup)
    assert 'DLF2' in inactive_names({'mp_physics': 11}, _lookup)
    assert 'DLF2' not in inactive_names({'mp_physics': 11, 'shcu_physics': 2}, _lookup)
    for mp in (8, 10, 28, 38, 40):
        assert 'DLF2' not in inactive_names({'mp_physics': mp, 'shcu_physics': 0}, _lookup)


def test_contradicting_supplied_value_is_refused(tmp_path):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    state, grid, nl = synthetic_case()
    nl.update(mp_physics=11)
    with pytest.raises(ValueError, match='DLF2'):
        prepare_wrfout_payload(state, grid, nl, tmp_path / 'bad', domain='d01',
                               domain_authority=writer_authority(grid, 'd01'),
                               valid_time='2023-01-15T01:00:00', run_start='2023-01-15T00:00:00', lead_hours=1.,
                               variable_subset=('DLF2',),
                               diagnostics={'DLF2': np.full((3, 4, 5), 1e-9, np.float32)})


def test_uw_shallow_convection_writer_branch_is_refused_before_load():
    """The constant is exact only because the DLF2 producer branch can never run."""
    from gpuwrf.io.namelist_binding import NamelistNotHonouredError, require_namelist_honoured
    control = {'physics': {'mp_physics': [11], 'shcu_physics': [0]}}
    require_namelist_honoured(control, ('d01',))
    with pytest.raises((NamelistNotHonouredError, ValueError, NotImplementedError)) as refused:
        require_namelist_honoured({'physics': {'mp_physics': [11], 'shcu_physics': [2]}}, ('d01',))
    assert 'shcu_physics' in str(refused.value)


@pytest.mark.parametrize('full', (False, True))
def test_nested_history_writer_emits_dlf2_without_a_subset(tmp_path, full):
    # Production _PerDomainWrfoutWriter path (no explicit variable subset), both stream modes.
    from types import SimpleNamespace
    from gpuwrf.io.netcdf_lock import Dataset
    from test_initial_history_wrf_t0 import _writer
    writer, state = _writer(tmp_path, lambda *a, **kw: {})
    writer._full_variable_set = full
    nl = writer.bundles['d01'].namelist
    writer.bundles['d01'].namelist = {**nl, 'mp_physics': 11, 'shcu_physics': 0,
                                      'physics': {'mp_physics': 11, 'shcu_physics': 0}}
    result = writer('d01', 200, SimpleNamespace(state=state, nwp_diagnostics=None))
    with Dataset(result['wrfout']) as ds:
        v = ds['DLF2']
        assert list(v.dimensions) == PROOF['field']['dimensions']
        assert str(v.getncattr('units')) == PROOF['field']['units']
        assert np.all(np.asarray(v[:], np.float32).view(np.uint32) == 0)
