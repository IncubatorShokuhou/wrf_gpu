"""Delete only the activity predicate; expect genuine activity assertion kills."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'scripts/v025'))
import cpu_guard  # CPU guard must precede JAX imports.
import gpuwrf.io.restart_schema as schema
import pytest

schema.history_field_required = lambda field, namelist: field != 'hfx'
sys.exit(pytest.main(['tests/test_restart_history_activity.py::test_history_field_activity',
                     '--basetemp', '<USER_HOME>/wrf_gpu2_lanes/t-u10erestart/pytest',
                     '--junitxml', sys.argv[1], '-q']))
