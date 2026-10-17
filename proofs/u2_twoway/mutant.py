"""Genuine assertion-failure check: replace WRF option 2 by the old option 1."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/v025"))
import cpu_guard  # Must precede every JAX/gpuwrf import, including pytest plugins.
import pytest
from gpuwrf.coupling import boundary_feedback as feedback

feedback.smdsm_smooth=feedback.sm121_smooth
sys.exit(pytest.main(["tests/test_u2_twoway_wrf_oracle.py", "-q", "-k", "2-case2",
                      "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/u2-twoway/pytest-mutant"]))
