"""Root-only harness selection must preserve the shared loader's default."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('prod_selection', Path(__file__).with_name('prod_inputs.py'))
PROD_INPUTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROD_INPUTS)


def test_domain_selection_and_missing_child(monkeypatch, tmp_path):
    monkeypatch.setattr(PROD_INPUTS, 'PROD', tmp_path)
    (tmp_path / 'wrfinput_d01').touch()
    monkeypatch.setattr(PROD_INPUTS, '_load', lambda names: names)
    assert PROD_INPUTS.prod_domains(root_only=True) == ('d01',)
    with pytest.raises(pytest.skip.Exception, match='PROD inputs unavailable'):
        PROD_INPUTS.prod_domains()
    (tmp_path / 'wrfinput_d02').touch()
    assert PROD_INPUTS.prod_domains() == ('d01', 'd02')


@pytest.mark.parametrize('names', [('d01',), ('d01', 'd02')])
def test_root_carry_abi_and_default_loader_unchanged(monkeypatch, names):
    from gpuwrf.contracts import state
    from gpuwrf.integration import nested_pipeline
    from gpuwrf.nesting import nest_o3
    old_guard = state._gpu_device
    carry = SimpleNamespace(radiation_diagnostics=object(), state=SimpleNamespace(theta=np.zeros((3, 4, 5))),
                            o3rad=None)
    carry.replace = lambda **kw: SimpleNamespace(**{**vars(carry), **kw})
    loaded = (None, {}, {}, None, {}, {'d01': carry})
    calls = []
    def load(config, selected):
        calls.append(selected)
        assert state._gpu_device is not old_guard
        return loaded
    monkeypatch.setattr(nested_pipeline, '_load_domains', load)
    monkeypatch.setattr(nest_o3, 'nest_o3_from_parent_enabled', lambda: True)
    got = PROD_INPUTS._load.__wrapped__(names)
    assert calls == [names]
    assert state._gpu_device is old_guard
    if len(names) == 1:
        assert got[5]['d01'].o3rad.shape == (3, 4, 5)
        assert got[5]['d01'].o3rad.dtype == np.float32
        np.testing.assert_array_equal(got[5]['d01'].o3rad, 0)
    else:
        assert got is loaded and got[5]['d01'] is carry
