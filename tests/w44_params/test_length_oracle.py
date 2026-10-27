"""Reference adapter control on the existing active-plume real-column fixture."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('w44_length_oracle', ROOT / 'proofs/w44_params/length_oracle.py')
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
ROWS = json.loads((ROOT / 'tests/v025/b_phys/fixtures/mynn_elb_mf_wrf.json').read_text())['rows']


@pytest.fixture(scope='module')
def libraries(tmp_path_factory):
    return ORACLE.build(tmp_path_factory.mktemp('length_wrf'))


@pytest.mark.parametrize('row', ROWS, ids=lambda row: str(row['tau']))
def test_original_default_control(row, libraries):
    original = ORACLE.evaluate(libraries['original'], row)
    default = ORACLE.evaluate(libraries['runtime'], row)
    np.testing.assert_array_equal(original.view(np.uint32), default.view(np.uint32))


def test_changed_length_has_a_real_active_witness(libraries):
    changed_cells = 0
    for row in ROWS:
        default = ORACLE.evaluate(libraries['runtime'], row)
        changed = ORACLE.evaluate(libraries['runtime'], row, alp1=.253)
        np.testing.assert_array_equal(changed[:, 0], default[:, 0])  # qkw is independent of ALP1.
        changed_cells += np.count_nonzero(changed[:, 1] != default[:, 1])
    assert changed_cells > 0
