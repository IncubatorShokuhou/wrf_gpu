"""Shin-Hong scale-aware path vs pristine WRF 4.7.1 at dx = 1 km and 9 km.

t-shinhong's oracle (tests/v040/test_shinhong_wrf_oracle.py) has real columns only at
3 km. The G2/WN3 nests run Shin-Hong at 9/3/1 km, where pu/pq/pthl/pthnl/ptke see other
dx/PBLH ratios (pthl/pthnl pmax clamps active). Same 8 real c07 CPU-WRF columns, modes
0 / 1 cold / 1 warm, both dx (proofs/w41_shinhongtke/make_dx_fixture.py).
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'proofs/w41_shinhongtke/fixtures/shinhong_dx_oracle.npz'
META = ROOT / 'proofs/w41_shinhongtke/fixtures/shinhong_dx_oracle.json'


def _base():
    spec = importlib.util.spec_from_file_location('shin_oracle_v040', ROOT / 'tests/v040/test_shinhong_wrf_oracle.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.KNOWN_COMPILED_EXCESS = set()  # strict: no disclosed cell at these resolutions
    return module


def _load():
    meta = json.loads(META.read_text())
    assert hashlib.sha256(FIX.read_bytes()).hexdigest() == meta['fixture_sha256']
    assert all(v['init_sensitivity_zero_vs_inf_byte'] for v in meta['per_dx'].values())
    assert meta['reference_build']['fma_instructions'] == 0
    d = np.load(FIX)
    assert len(d['mode']) == 48 and set(d['dx'].tolist()) == {1000, 9000}
    assert np.all(d['surface'][:, 13] == d['dx']) and np.all(d['surface'][:, 14] == d['dx'])
    return d


def _groups(d):
    groups = {}
    for i, key in enumerate(zip(d['nz'].tolist(), d['dx'].tolist())):
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def test_literal_real4_is_byte_identical_at_1_and_9_km():
    base, d = _base(), _load()
    checked = 0
    for idxs in _groups(d):
        n, out = base._run(d, idxs, eager=True)
        for pos, i in enumerate(idxs):
            for name, j in base.FIELDS:
                wrf = d['wrf'][i][:n, j]
                act = out[name][pos].astype(np.float32)
                assert np.isfinite(act).all(), (i, name)
                if name in ('qc', 'qi'):
                    tiny = np.abs(wrf) < base.Q_SUBNORMAL_FLOOR
                    assert act[~tiny].tobytes() == wrf[~tiny].tobytes(), (i, name)
                    assert np.all(np.abs(act[tiny].astype(np.float64) - wrf[tiny]) <= base.Q_SUBNORMAL_FLOOR), (i, name)
                else:
                    assert act.tobytes() == wrf.tobytes(), (
                        f'row {i} dx {d["dx"][i]} {name}: max abs {np.max(np.abs(act.astype(np.float64) - wrf)):.3e}')
                checked += 1
            assert int(out['kpbl'][pos]) == int(round(float(d['wrf'][i][1, 9]))), i
            assert out['pblh'][pos].tobytes() == d['wrf'][i][0, 9].tobytes(), i
    assert checked == 48 * len(base.FIELDS)


@pytest.fixture(scope='module')
def compiled():
    base, d = _base(), _load()
    return base, d, [(idxs, *base._run(d, idxs, eager=False)) for idxs in _groups(d)]


def test_compiled_inside_wrf_input_ulp_envelope_at_1_and_9_km(compiled):
    base, d, runs = compiled
    for idxs, n, out in runs:
        base._assert_compiled(d, n, idxs, out)


def test_compiled_gate_rejects_erased_driving_outputs(compiled):
    base, d, runs = compiled
    edited = [(idxs, n, {k: np.array(v, copy=True) for k, v in out.items()}) for idxs, n, out in runs]
    for _, _, out in edited:
        for name in base.DRIVING:
            out[name][:] = 0.0
    with pytest.raises(AssertionError, match='outside'):
        for idxs, n, out in edited:
            base._assert_compiled(d, n, idxs, out)


def test_resolutions_differ_in_wrf_truth():
    """The two dx sets are not the same oracle twice: WRF's own rates change with dx."""
    d = _load()
    same_input = [i for i in range(24) if d['mode'][i] == 0 or i % 3 == 1]  # mode0 + 1-cold (warm TKE is dx-own)
    for i in same_input:
        assert d['profiles'][i].tobytes() == d['profiles'][i + 24].tobytes()
    assert any(d['wrf'][i].tobytes() != d['wrf'][i + 24].tobytes() for i in same_input)
