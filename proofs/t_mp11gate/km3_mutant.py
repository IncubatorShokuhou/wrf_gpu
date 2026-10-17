"""Delete only the new km3 guard; require genuine entry-order failures."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'scripts/v025'))
import cpu_guard
import pytest
from gpuwrf.io import km3_config
km3_config.selection_errors=lambda *a,**k: []
raise SystemExit(pytest.main([str(root/'tests/test_km3_gpu_admission.py'),'-q',
    '-k','not cli_pre_jax','--basetemp',sys.argv[1],'--junitxml',sys.argv[2]]))
