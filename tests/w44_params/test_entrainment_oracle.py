"""Unchanged full-source rebuild control before varying the WRF entrainment literal."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('w44_entrainment_oracle', ROOT / 'proofs/w44_params/entrainment_oracle.py')
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    return ORACLE.build_and_run(tmp_path_factory.mktemp('entrainment_wrf'))


def test_rebuilt_default_matches_original_object(reference):
    for name in ORACLE.DMP_FIELDS:
        np.testing.assert_array_equal(reference['default'][name].view(np.uint32),
                                      reference['original'][name].view(np.uint32))


def test_changed_entrainment_has_real_plume_response(reference):
    assert np.any(reference['original']['edmf_s_aw'] > 0)
    assert np.any(reference['changed']['edmf_s_awqv'] != reference['default']['edmf_s_awqv'])
