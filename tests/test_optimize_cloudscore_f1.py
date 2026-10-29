"""Exact independent cr-v45 F1 counterexample from the 4d3f22a29f review."""
import importlib.util
from pathlib import Path
import pytest
from gpuwrf.optimize import continuous
from gpuwrf.surface_params import eki
def test_native_requires_alisios_correlation_contract(tmp_path):
 p=Path(__file__).with_name('test_optimize_cloudscore.py');s=importlib.util.spec_from_file_location('critic_cloud_fixture',p);f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
 assert eki.ENSEMBLE_SPACE_UPDATE is True
 cfg=f.facade(tmp_path);raw={k:cfg[k] for k in ('format','truth','daylight','regions')}
 with pytest.raises(ValueError,match='correlation|contract|ALISIOS|admitted'):
  continuous._scorer({'kind':'cloud','sigma':.2,'cloud':raw,'observation_operator':'native'},[],lambda p:p)
