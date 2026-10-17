"""WRF-assigned constant history of the admitted branches: QNBCA/TAOD5502D (mp28/38), MSKF_REFL_10CM (mp40).

WRF branch conditions (pristine 4.7.1):
* QNBCA: mp_gt_driver sets nbca=0 on every microphysics column unless wif_input_opt=2
  (module_mp_thompson.F:1336-1338); zero QNBCA input keeps the specified zone at 0.
* TAOD5502D: assigned only under aer_opt=3 with RRTMG SW (module_radiation_driver.F:1653-1689).
* MSKF_REFL_10CM: assigned only under diagflag .and. do_radar_ref==1 (module_mp_morr_two_moment_aero.F:1183-1203);
  do_radar_ref=1 itself is computed (tests/v040/test_classd_morrison_radar.py).
Original CPU proof: c19 (mp28), c65 (mp38), c20 (mp40), every cell of all 25 frames +0 bits.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.io.producer_history import inactive_names

ROOT = Path(__file__).resolve().parents[2]
PROOF = json.loads((ROOT / 'proofs/t_classd/CONSTANT_CPU_ALL_FRAMES.json').read_text())['cases']
CASES = {'c19': (28, ('QNBCA', 'TAOD5502D')), 'c65': (38, ('QNBCA', 'TAOD5502D')), 'c20': (40, ('MSKF_REFL_10CM',))}


def _lookup(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else default


def test_original_cpu_all_frames_hold_positive_zero():
    for case, (mp, names) in CASES.items():
        proof = PROOF[case]
        assert proof['frame_count'] == 25 and proof['all_frames_zero']
        assert int(proof['global_attributes']['MP_PHYSICS']) == mp
        for frame in proof['frames']:
            for name in names:
                assert frame['fields'][name]['bits'] == 0 and frame['fields'][name]['nonzero_bits'] == 0


@pytest.mark.parametrize('case', tuple(CASES))
@pytest.mark.parametrize('full', (False, True))
def test_writer_emits_original_bits_and_metadata(tmp_path, case, full):
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    mp, names = CASES[case]
    state, grid, nl = synthetic_case()
    nl.update(mp_physics=mp)
    auth = writer_authority(grid, 'd01')
    p = prepare_wrfout_payload(state, grid, nl, tmp_path / case, domain='d01', domain_authority=auth,
                               valid_time='2023-01-15T01:00:00', run_start='2023-01-15T00:00:00', lead_hours=1.,
                               full_variable_set=full, variable_subset=names)
    write_prepared_wrfout(p, expected_domain='d01', expected_domain_authority=auth, variable_subset=names)
    with Dataset(p.target) as ds:
        for name in names:
            v, ref = ds[name], PROOF[case]['fields'][name]
            assert list(v.dimensions) == ref['dimensions'] and str(v.dtype) == ref['dtype']
            for key in ('description', 'units', 'stagger', 'MemoryOrder'):
                assert str(v.getncattr(key)) == ref[key], (name, key)
            assert np.all(np.asarray(v[:], np.float32).view(np.uint32) == 0)


def test_active_branches_never_receive_the_constant():
    assert 'QNBCA' not in inactive_names({'mp_physics': 28, 'wif_input_opt': 2}, _lookup)
    assert 'TAOD5502D' not in inactive_names({'mp_physics': 38, 'aer_opt': 3}, _lookup)
    assert 'MSKF_REFL_10CM' not in inactive_names({'mp_physics': 40, 'do_radar_ref': 1}, _lookup)
    assert {'QNBCA', 'TAOD5502D'} <= set(inactive_names({'mp_physics': 28, 'wif_input_opt': 1}, _lookup))
    for mp in (8, 10, 18):
        assert not {'QNBCA', 'TAOD5502D', 'MSKF_REFL_10CM'} & set(inactive_names({'mp_physics': mp}, _lookup))


def test_contradicting_supplied_value_is_refused(tmp_path):
    from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
    from test_m7_netcdf_writer import synthetic_case, writer_authority
    state, grid, nl = synthetic_case()
    nl.update(mp_physics=28)
    with pytest.raises(ValueError, match='QNBCA'):
        prepare_wrfout_payload(state, grid, nl, tmp_path / 'bad', domain='d01',
                               domain_authority=writer_authority(grid, 'd01'),
                               valid_time='2023-01-15T01:00:00', run_start='2023-01-15T00:00:00', lead_hours=1.,
                               variable_subset=('QNBCA',),
                               diagnostics={'QNBCA': np.ones((3, 4, 5), np.float32)})


@pytest.mark.parametrize('mp', (28, 38))
def test_nonzero_black_carbon_input_is_refused(mp):
    """Only zero QNBCA input keeps WRF's unported specified-zone transport identically zero."""
    from gpuwrf.nesting.scalar_initialization import initialize_extra_scalar_fields
    from test_v016_thompson_aero_threading import _tiny_state
    state = _tiny_state()

    class Run:
        namelist = {'physics': {'mp_physics': [mp]}}

        def load_wrfinput(self, domain, name, **kwargs):
            if name == 'QNBCA':
                value = np.zeros(state.qv.shape, np.float32)
                value[0, 0, 0] = 1e6
                return value
            raise KeyError(name)

    with pytest.raises(ValueError, match=f'mp_physics={mp} qualifies only zero QNBCA'):
        initialize_extra_scalar_fields(Run(), 'd01', state, None)


@pytest.mark.parametrize('mp,group,key,value', [
    (28, 'physics', 'aer_opt', 3), (38, 'physics', 'aer_opt', 3),
    (28, 'domains', 'wif_input_opt', 2), (38, 'domains', 'wif_input_opt', 2)])
def test_varying_wrf_branches_are_refused_before_load(mp, group, key, value):
    """The constants are exact only because these branches can never run."""
    from gpuwrf.io.namelist_binding import NamelistNotHonouredError, require_namelist_honoured
    from gpuwrf.io.namelist_check import validate_operational_namelist
    control = {'physics': {'mp_physics': [mp], 'cu_physics': [0]}}
    validate_operational_namelist(control)
    require_namelist_honoured(control, ('d01',))
    raw = {'physics': {'mp_physics': [mp], 'cu_physics': [0]}}
    raw.setdefault(group, {})[key] = [value]
    with pytest.raises((NamelistNotHonouredError, ValueError, NotImplementedError)) as refused:
        validate_operational_namelist(raw)
        require_namelist_honoured(raw, ('d01',))
    assert key in str(refused.value) or 'radar' in str(refused.value)
