"""Validate the reference adapter before comparing new runtime coefficients."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('w44_phim_oracle', ROOT / 'proofs/w44_params/phim_oracle.py')
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
ROWS = json.loads((ROOT / 'tests/v025/b_phys/fixtures/mynn_predict_saw_wrf.json').read_text())['rows']


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    return ORACLE.build(tmp_path_factory.mktemp('phim_wrf'))


def real_zet(row):
    flux, state = row['flux'], row['state']
    ust = np.asarray(flux['ustar'], np.float32)
    fltv = np.asarray(flux['fltv'], np.float32)
    dz0 = np.asarray(state['dz'], np.float32)[:, 0]
    # The original MYNN driver supplies z/L at the first half layer.
    rmol = -np.float32(.4) * (np.float32(9.81) / np.float32(300.)) * fltv / np.maximum(ust**3, np.float32(1e-6))
    return np.clip(np.float32(.5) * dz0 * rmol, -20., 20.)


@pytest.mark.parametrize('row', ROWS, ids=lambda row: row['label'])
def test_reference_default_bytes_and_independent_stability_branches(row, libraries):
    zet = real_zet(row)
    original = ORACLE.evaluate(libraries['original'], zet)
    default = ORACLE.evaluate(libraries['runtime'], zet)
    np.testing.assert_array_equal(original.view(np.uint32), default.view(np.uint32))
    unstable = ORACLE.evaluate(libraries['runtime'], zet, cphm_unst=17.6)
    stable = ORACLE.evaluate(libraries['runtime'], zet, am_st=6.71)
    np.testing.assert_array_equal(unstable[zet >= 0], default[zet >= 0])
    np.testing.assert_array_equal(stable[zet < 0], default[zet < 0])
    for active, changed in ((zet < 0, unstable), (zet > 0, stable)):
        if np.any(active):
            assert np.any(changed[active] != default[active])
