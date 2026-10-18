"""ALISIOS availability checks must not hide synthetic tests during collection."""
from pathlib import Path
import runpy

import pytest


MODULES = (
    "a1_da/test_a1_ensemble.py",
    "a1_da/test_a1_perturb.py",
    "a1_da/test_member_cycle.py",
    "a1_da/test_operators.py",
    "a1_da/test_state_io.py",
    "test_disabled_surface_history.py",
    "test_fidq2_lead_zero_history_t.py",
    "test_full_history_default.py",
    "test_initial_history_wrf_t0.py",
    "test_online_diagnostics.py",
    "test_online_diagnostics_restart_store.py",
    "test_online_radar_caller.py",
    "test_solar_caller.py",
    "test_solar_history.py",
    "test_solar_native_restart.py",
    "test_solar_restart.py",
    "test_tracer_restart.py",
    "test_u2_feedback_restart.py",
    "test_writer_clat_feedback.py",
    "test_writer_held_volume_fields.py",
    "v040/milbrandt/test_restart.py",
)


@pytest.mark.parametrize("module", MODULES)
def test_alisios_modules_import_without_skipping_synthetic_tests(module, monkeypatch):
    root = Path(__file__).parent
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "a1_da"))
    try:
        runpy.run_path(str(root / module), run_name="_alisios_import_probe")
    except pytest.skip.Exception as exc:
        pytest.fail(f"{module} skipped at import instead of its real-data test: {exc}")
