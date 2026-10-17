"""CUDA architecture gate: GPUs below sm_80 are refused before model work (mocked devices, CPU-only)."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwrf.runtime import gpu_preflight as gp

REPO = Path(__file__).resolve().parents[1]
SWISS = REPO / "examples" / "switzerland_d01"


def _cuda_device(kind: str, cc: str, device_id: int = 0):
    return SimpleNamespace(platform="gpu", device_kind=kind, compute_capability=cc, id=device_id,
                           client=SimpleNamespace(platform_version="cuda 12090"))


@pytest.mark.parametrize("value,expected", [
    ("6.1", (6, 1)), ("7.5", (7, 5)), ("8.0", (8, 0)), ("12.0", (12, 0)), ("9", (9, 0)), ((8, 6), (8, 6)),
    ("", None), (None, None), ("sm_61", None), ("6.x", None),
])
def test_parse_compute_capability(value, expected):
    assert gp.parse_compute_capability(value) == expected


@pytest.mark.parametrize("cc", ["6.1", "7.0", "7.5"])
def test_pre_ampere_is_refused_with_actionable_message(cc):
    with pytest.raises(gp.UnsupportedGpuError) as info:
        gp.check_cuda_architecture(name="NVIDIA GeForce GTX 1080 Ti", compute_capability=cc, environ={})
    text = str(info.value)
    assert f"compute capability {cc}" in text and ">= 8.0" in text and "sm_80" in text
    assert "Triton" in text and "FAILED_PRECONDITION" in text
    assert "JAX_PLATFORMS=cpu gpuwrf run" in text
    assert f"{gp.UNSUPPORTED_GPU_ENV}=1" in text and "unsupported, slow" in text
    assert info.value.payload["status"] == "unsupported"
    assert isinstance(info.value, gp.GpuPreflightError)
    assert gp.UNSUPPORTED_GPU_EXIT_CODE == 69


@pytest.mark.parametrize("cc", ["8.0", "8.6", "8.9", "9.0", "10.0", "12.0"])
def test_ampere_and_newer_pass_silently(cc):
    err = io.StringIO()
    payload = gp.check_cuda_architecture(name="NVIDIA GeForce RTX 5090", compute_capability=cc,
                                         environ={gp.UNSUPPORTED_GPU_ENV: "1"}, stderr=err)
    assert payload["status"] == "supported" and err.getvalue() == ""


def test_explicit_opt_in_admits_with_unsupported_slow_label():
    err = io.StringIO()
    payload = gp.check_cuda_architecture(name="NVIDIA GeForce GTX 1080 Ti", compute_capability="6.1",
                                         environ={gp.UNSUPPORTED_GPU_ENV: "1"}, stderr=err)
    assert payload["status"] == "unsupported_opt_in"
    assert "UNSUPPORTED, SLOW" in err.getvalue() and "6.1" in err.getvalue()
    for value in ("0", "false", "", "off"):
        with pytest.raises(gp.UnsupportedGpuError):
            gp.check_cuda_architecture(name="x", compute_capability="6.1", environ={gp.UNSUPPORTED_GPU_ENV: value})


def test_unreadable_capability_is_unknown_not_refused():
    assert gp.check_cuda_architecture(name="x", compute_capability=None, environ={})["status"] == "unknown"


def test_nvidia_smi_selection_follows_cuda_visible_devices():
    rows = ["0, GPU-aaa, 12.0, NVIDIA GeForce RTX 5090", "1, GPU-bbb, 6.1, NVIDIA GeForce GTX 1080 Ti"]
    assert gp.select_nvidia_smi_compute_capability(rows, {}) == ("NVIDIA GeForce RTX 5090", "12.0")
    assert gp.select_nvidia_smi_compute_capability(rows, {"CUDA_VISIBLE_DEVICES": "1"})[1] == "6.1"
    assert gp.select_nvidia_smi_compute_capability(rows, {"CUDA_VISIBLE_DEVICES": "GPU-bbb"})[1] == "6.1"
    with pytest.raises(RuntimeError, match="does not list"):
        gp.select_nvidia_smi_compute_capability(rows, {"CUDA_VISIBLE_DEVICES": "2"})
    with pytest.raises(RuntimeError, match="no GPU rows"):
        gp.select_nvidia_smi_compute_capability(["garbage"], {})


def test_cli_preflight_is_backend_dark_and_skips_non_cuda():
    def boom(_env):
        raise AssertionError("nvidia-smi must not be queried for a non-CUDA target")

    assert gp.run_gpu_architecture_preflight(environ={"JAX_PLATFORMS": "cpu"}, query=boom)["status"] == "skipped"
    assert gp.run_gpu_architecture_preflight(environ={"JAX_PLATFORMS": "rocm"}, query=boom)["status"] == "skipped"


def test_cli_preflight_refuses_old_cuda_gpu_and_records_query_failures():
    env = {"JAX_PLATFORMS": "cuda"}
    with pytest.raises(gp.UnsupportedGpuError):
        gp.run_gpu_architecture_preflight(environ=env, query=lambda _e: ("NVIDIA GeForce GTX 1080 Ti", "6.1"))
    ok = gp.run_gpu_architecture_preflight(environ=env, query=lambda _e: ("NVIDIA GeForce RTX 5090", "12.0"))
    assert ok["status"] == "supported" and ok["source"] == "nvidia-smi"

    def failing(_env):
        raise RuntimeError("nvidia-smi not found on PATH")

    unknown = gp.run_gpu_architecture_preflight(environ=env, query=failing)
    assert unknown["status"] == "unknown" and "nvidia-smi not found" in unknown["error"]


def test_nvidia_smi_architecture_query_has_a_finite_timeout_and_timeout_is_unknown(monkeypatch):
    """A hung nvidia-smi must not hang `gpuwrf run`: finite timeout, TimeoutExpired -> status unknown."""

    import math
    import subprocess

    seen = {}

    def hung(cmd, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(gp.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gp.subprocess, "run", hung)
    payload = gp.run_gpu_architecture_preflight(environ={"JAX_PLATFORMS": "cuda"})
    assert payload["status"] == "unknown" and payload["source"] == "nvidia-smi"
    assert "timed out" in payload["error"]
    assert isinstance(seen.get("timeout"), (int, float)) and math.isfinite(seen["timeout"]) and 0 < seen["timeout"] <= 60
    with pytest.raises(RuntimeError, match="timed out"):
        gp.query_nvidia_smi_compute_capability({})


def test_jax_device_gate_checks_cuda_devices_only():
    with pytest.raises(gp.UnsupportedGpuError):
        gp.check_jax_device_architecture(_cuda_device("NVIDIA GeForce GTX 1080 Ti", "6.1"), environ={})
    assert gp.check_jax_device_architecture(_cuda_device("NVIDIA GeForce RTX 5090", "12.0"),
                                            environ={})["status"] == "supported"
    rocm = SimpleNamespace(platform="gpu", device_kind="AMD Instinct MI300X", compute_capability="9.4",
                           client=SimpleNamespace(platform_version="rocm 6.2"))
    assert gp.check_jax_device_architecture(rocm, environ={})["status"] == "skipped"
    cpu = SimpleNamespace(platform="cpu", device_kind="cpu", client=SimpleNamespace(platform_version="cpu"))
    assert gp.check_jax_device_architecture(cpu, environ={})["status"] == "skipped"


def test_state_allocation_refuses_pre_ampere_device(monkeypatch):
    """Programmatic runs (no CLI) are refused when the first state field is placed."""

    from gpuwrf.contracts import state

    old = _cuda_device("NVIDIA GeForce GTX 1080 Ti", "6.1", device_id=7)
    monkeypatch.setattr(state.jax, "devices", lambda *a, **k: [old])
    monkeypatch.setattr(state, "_GPU_ARCHITECTURE_CHECKED", {})
    monkeypatch.delenv(gp.UNSUPPORTED_GPU_ENV, raising=False)
    with pytest.raises(gp.UnsupportedGpuError, match="compute capability 6.1"):
        state._gpu_device()
    monkeypatch.setenv(gp.UNSUPPORTED_GPU_ENV, "1")
    assert state._gpu_device() is old  # explicit opt-in: admitted (warning printed once)
    new = _cuda_device("NVIDIA GeForce RTX 5090", "12.0", device_id=0)
    monkeypatch.delenv(gp.UNSUPPORTED_GPU_ENV)
    monkeypatch.setattr(state.jax, "devices", lambda *a, **k: [new])
    assert state._gpu_device() is new


def test_architecture_opt_in_is_not_an_aot_key_input():
    from gpuwrf.runtime.aot_cheap_key import trace_env_is_inert

    assert trace_env_is_inert(gp.UNSUPPORTED_GPU_ENV)


@pytest.mark.skipif(not (SWISS / "wrfinput_d01").exists(), reason="bundled Swiss case missing")
def test_cli_run_fails_early_on_pre_ampere_gpu(monkeypatch, tmp_path, capsys):
    """`gpuwrf run` exits 69 with the message before any forecast work on a cc 6.1 GPU."""

    from gpuwrf import cli
    import gpuwrf.integration.nested_pipeline as nested

    def no_model(*_a, **_k):
        raise AssertionError("forecast must not start on an unsupported GPU")

    monkeypatch.setattr(nested, "execute_nested_pipeline", no_model)
    monkeypatch.setattr(gp, "query_nvidia_smi_compute_capability",
                        lambda _env: ("NVIDIA GeForce GTX 1080 Ti", "6.1"))
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")             # backend-dark CUDA target
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")  # explicit allocator: no re-exec
    monkeypatch.delenv(gp.UNSUPPORTED_GPU_ENV, raising=False)
    rc = cli.main(["run", "--input-dir", str(SWISS), "--output-dir", str(tmp_path / "out"),
                   "--domain", "d01", "--hours", "1", "--scratch-dir", str(tmp_path / "scratch")])
    err = capsys.readouterr().err
    assert rc == gp.UNSUPPORTED_GPU_EXIT_CODE
    assert "unsupported GPU: NVIDIA GeForce GTX 1080 Ti has CUDA compute capability 6.1" in err
    assert "JAX_PLATFORMS=cpu gpuwrf run" in err
    assert not (tmp_path / "out" / "wrfout_d01_2023-01-15_00:00:00").exists()
