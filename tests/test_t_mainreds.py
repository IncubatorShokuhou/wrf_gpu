"""Product regressions from the zero-red release sweep (CPU/stub telemetry only)."""
from importlib import import_module
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_coupling_configuration_import_is_backend_dark():
    code = '''
import importlib.abc
import sys
sys.argv = ['gpuwrf', 'doctor']  # Offline bootstrap, as in the actual CLI guards.
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jax' or fullname.startswith(('jax.', 'gpuwrf.physics', 'gpuwrf.dynamics')):
            raise AssertionError('heavy import during configuration admission: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
import gpuwrf.coupling as coupling
from gpuwrf.coupling.cuplus_config import selection_errors
assert selection_errors({'physics': {'cu_physics': 0}}) == []
assert set(coupling.__all__).issubset(dir(coupling))
try:
    coupling.not_an_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown export accepted')
'''
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_coupling_exports_preserve_original_objects():
    import gpuwrf.coupling as coupling
    original = {
        "boundary_apply": ("BoundaryConfig", "apply_lateral_boundaries"),
        "driver": ("build_initial_state", "run_forecast_segment", "run_to_output_leads"),
        "physics_couplers": ("mynn_adapter", "rrtmg_adapter", "surface_adapter", "thompson_adapter"),
    }
    assert set(coupling.__all__) == {name for names in original.values() for name in names}
    for module, names in original.items():
        source = import_module(f"gpuwrf.coupling.{module}")
        for name in names:
            assert getattr(coupling, name) is getattr(source, name)
            assert coupling.__dict__[name] is getattr(source, name)


def test_audit_optional_geometry_and_tracer_scaling():
    from scripts.m7_1km_memory_audit import field_shapes_for, build_static_memory_model
    from gpuwrf.contracts.state import State
    assert set(field_shapes_for(3, 4, 6)) == set(State.__slots__)
    shapes = field_shapes_for(3, 4, 6, tracer_count=2)
    assert set(shapes) == set(State.__slots__)
    for name in ("re_cloud", "re_ice", "re_snow", "tke", "epsilon_pbl", "epsilon_exch_m", "epsilon_wq"):
        assert shapes[name] == (3, 4, 6)
    for name in ("sms_br", "sms_gamu", "sms_ustm", "sfc_wspd"):
        assert shapes[name] == (4, 6)
    assert shapes["epsilon_pbl_step"] == ()
    # CAM-MG mp=11 Registry leaves (mp11-only since 958ce52667) keep their real staggering.
    for name in ("cldfra_mp", "lradius", "wsedl3d"):
        assert shapes[name] == (3, 4, 6)
    for name in ("cam_kvh3d", "cam_tke_pbl", "cam_turbtype3d", "cam_smaw3d"):
        assert shapes[name] == (4, 4, 6)
    for name in ("cam_rainnc", "cam_sr", "cam_fatal", "cam_snowh"):
        assert shapes[name] == (4, 6)
    for name in ("qh_bdy", "Nc_bdy", "pek_adv_bdy", "qt_bdy", "qrimef_bdy"):
        assert shapes[name] == (1, 4, 5, 3, 7)
    assert shapes["tracers"] == (2, 3, 4, 6)
    assert shapes["tracer_bdy"] == (2, 2, 4, 5, 3, 6)
    assert shapes["tracer_budget"] == (2, 4)
    grid = {"derived_full_1km": {"mass_shape": [3, 4, 6]}}
    one = build_static_memory_model(grid, tracer_count=1)
    two = build_static_memory_model(grid, tracer_count=2)
    # Independent counts: one REAL tracer volume, two boundary times, four budget scalars.
    assert two["total_state_bytes"] - one["total_state_bytes"] == 4 * (3*4*6 + 2*4*5*3*6 + 4)
    scalar = next(f for f in two["fields"] if f["name"] == "epsilon_pbl_step")
    assert (scalar["elements"], scalar["bytes"], scalar["dtype"]) == (1, 4, "int32")
    fatal = next(f for f in two["fields"] if f["name"] == "cam_fatal")
    assert (fatal["elements"], fatal["bytes"], fatal["dtype"]) == (24, 96, "int32")


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_audit_refuses_invalid_tracer_count(count):
    from scripts.m7_1km_memory_audit import field_shapes_for
    with pytest.raises(ValueError, match="tracer_count"):
        field_shapes_for(3, 4, 6, tracer_count=count)


@pytest.mark.parametrize("telemetry,query_rc", [
    ("Failed to initialize NVML", 0), ("Failed to initialize NVML", 1),
    ("N/A", 0), ("", 0), ("25000", 1),
])
@pytest.mark.parametrize("measured_rc", [0, 7, 124])
def test_run_arm_unknown_telemetry_preserves_measured_rc(tmp_path, telemetry, query_rc, measured_rc):
    result = _run_arm_stub(tmp_path, telemetry, query_rc, measured_rc)
    assert result.returncode == measured_rc, result.stderr
    assert "drain UNKNOWN" in result.stderr
    assert f"measured rc={measured_rc}" in result.stderr
    assert "unbound variable" not in result.stderr


def _run_arm_stub(tmp_path, telemetry, query_rc, measured_rc, threshold="24000"):
    vendor = tmp_path / "nvidia-smi"
    vendor.write_text('#!/bin/bash\nprintf "%s\\n" "$STUB_MEMORY"\nexit "$STUB_QUERY_RC"\n')
    vendor.chmod(0o755)
    envfile = tmp_path / "env.sh"
    envfile.write_text("s0_guard() { :; }\n")
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               STUB_MEMORY=telemetry, STUB_QUERY_RC=str(query_rc), BENCH_SAMPLERS="0",
               GPUWRF_LOCK_MIN_FREE_MIB=threshold)
    return subprocess.run(["bash", str(ROOT / "scripts/bench/run_arm.sh"), str(envfile),
                           str(ROOT), str(tmp_path), "", "", "", "--", "bash", "-c",
                           f"exit {measured_rc}"], env=env, capture_output=True, text=True, timeout=10)


def test_run_arm_valid_decimal_telemetry(tmp_path):
    result = _run_arm_stub(tmp_path, "00025000", 0, 7, threshold="024000")
    assert result.returncode == 7, result.stderr
    assert "UNKNOWN" not in result.stderr


def test_run_arm_valid_low_memory_still_waits_for_drain(tmp_path):
    vendor = tmp_path / "nvidia-smi"
    vendor.write_text('''#!/bin/bash
if [[ "$1" == --query-gpu=memory.free ]]; then
    if [[ -e "$STUB_DRAIN_QUERY" ]]; then echo 25000
    else touch "$STUB_DRAIN_QUERY"; echo 20000
    fi
else echo 'none'
fi
''')
    vendor.chmod(0o755)
    sleeper = tmp_path / "sleep"
    sleeper.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> "$STUB_DRAIN_SLEEPS"\n')
    sleeper.chmod(0o755)
    envfile = tmp_path / "env.sh"
    envfile.write_text("s0_guard() { :; }\n")
    sleeps = tmp_path / "sleeps"
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               STUB_DRAIN_QUERY=str(tmp_path / "queried"), STUB_DRAIN_SLEEPS=str(sleeps),
               BENCH_SAMPLERS="0", GPUWRF_LOCK_MIN_FREE_MIB="24000")
    result = subprocess.run(["bash", str(ROOT / "scripts/bench/run_arm.sh"), str(envfile),
                             str(ROOT), str(tmp_path), "", "", "", "--", "bash", "-c", "exit 7"],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 7, result.stderr
    assert "arm draining: free 20000 MiB < 24000" in result.stderr
    assert sleeps.read_text() == "2\n"
    assert "UNKNOWN" not in result.stderr


def test_run_arm_refuses_invalid_drain_threshold_before_measured_command(tmp_path):
    result = _run_arm_stub(tmp_path, "25000", 0, 7, threshold="bad_threshold")
    assert result.returncode == 2
    assert "must be a nonnegative integer" in result.stderr
