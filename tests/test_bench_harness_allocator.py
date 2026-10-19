"""CPU tests for the s0_nested_harness launch phase (w42-harness): no device touch before the CLI C-auto
pool step, the pool attested in the receipt. No GPU: the CUDA target and the NVML probe are stubbed."""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "bench" / "s0_nested_harness.py"


@pytest.fixture()
def harness(monkeypatch):
    monkeypatch.setenv("BENCH_SRC_REPO", str(ROOT))  # never the live main checkout (E159)
    spec = importlib.util.spec_from_file_location("s0_harness_under_test", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setenv(mod.LAUNCH_ENV, "")  # restored (unset) at teardown; _pre_exec writes it
    return mod


def _fake_cli(body: str) -> types.ModuleType:
    """A module whose main() reaches the allocator step through a module-global lookup, like gpuwrf.cli."""
    mod = types.ModuleType("fake_cli")
    exec(body, mod.__dict__)
    return mod


@pytest.mark.parametrize("env,mode", [
    ({"_GPUWRF_C_AUTO": "1", "_GPUWRF_C_AUTO_BUDGET": "6442450944", "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
      "XLA_CLIENT_MEM_FRACTION": "0.205000", "_GPUWRF_NESTED_ALLOC_REEXEC": "1"}, "c_auto"),
    ({"_GPUWRF_C_AUTO": "1", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "_GPUWRF_NESTED_ALLOC_REEXEC": "1"}, "demand"),
    ({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, "explicit"),
    ({}, "explicit"),
])
def test_allocator_record_modes(harness, env, mode):
    rec = harness._allocator_record(env)
    assert rec["mode"] == mode
    if mode == "c_auto":
        assert rec["c_auto_budget_bytes"] == 6442450944 and rec["mem_fraction"] == "0.205000" and rec["cli_reexec"]


def test_pre_exec_no_reexec_keeps_process(harness, tmp_path):
    cli = _fake_cli(
        "calls = []\n"
        "def _maybe_reexec_for_nested_allocator(args):\n"
        "    calls.append(args)\n"
        "def main(argv):\n"
        "    _maybe_reexec_for_nested_allocator(argv)\n"
        "    raise AssertionError('the harness must stop the launch-phase CLI at its allocator step')\n")
    assert harness._pre_exec(cli, ["run"], tmp_path) is None
    assert cli.calls == [["run"]]
    assert cli._maybe_reexec_for_nested_allocator.__name__ == "_maybe_reexec_for_nested_allocator"  # restored
    rec = json.loads((tmp_path / harness.PRE_EXEC_FILE).read_text())
    assert rec["reexec"] is False and "backend_dark" in rec["allocator_step"]
    launch = json.loads(os.environ[harness.LAUNCH_ENV])
    assert launch["nonce"] == rec["nonce"] and launch["pid"] == os.getpid()
    # this same process is then the measured one: 0 s launch phase, not a re-exec
    meas = harness._launch_record(tmp_path)
    assert meas["reexec"] is False and meas["pre_exec_s"] == 0.0


def test_pre_exec_exec_replaces_process(harness, tmp_path, monkeypatch):
    class Exec(BaseException):
        pass

    from gpuwrf.runtime import gpu_allocator
    monkeypatch.setattr(gpu_allocator, "_pool_device_memory", lambda: (32 << 30, 28 << 30))
    cli = _fake_cli(
        "from gpuwrf.runtime import gpu_allocator\n"
        "def _maybe_reexec_for_nested_allocator(args):\n"
        "    gpu_allocator._pool_device_memory()\n"
        "    raise EXEC\n"
        "def main(argv):\n"
        "    _maybe_reexec_for_nested_allocator(argv)\n")
    cli.EXEC = Exec
    with pytest.raises(Exec):
        harness._pre_exec(cli, ["run"], tmp_path)
    rec = json.loads((tmp_path / harness.PRE_EXEC_FILE).read_text())
    assert "reexec" not in rec  # the step never returned
    assert rec["nvml_probe"]["free_bytes"] == 28 << 30 and rec["nvml_probe"]["total_bytes"] == 32 << 30
    assert gpu_allocator._pool_device_memory() == (32 << 30, 28 << 30)  # the monkeypatched stub, wrapper removed


def test_pre_exec_returns_cli_rc_before_allocator_step(harness, tmp_path):
    cli = _fake_cli("def _maybe_reexec_for_nested_allocator(args):\n    raise AssertionError\n"
                    "def main(argv):\n    return 2\n")
    assert harness._pre_exec(cli, ["run"], tmp_path) == 2


def test_launch_record_reexec_and_nonce(harness, tmp_path, monkeypatch):
    nonce = f"{os.getpid()}-1"
    monkeypatch.setenv(harness.LAUNCH_ENV, json.dumps({"nonce": nonce, "pid": os.getpid(),
                                                       "wall_utc": harness.T0_WALL - 4.5}))
    monkeypatch.setenv("_GPUWRF_NESTED_ALLOC_REEXEC", "1")
    (tmp_path / harness.PRE_EXEC_FILE).write_text(json.dumps({"nonce": nonce, "pid": os.getpid(),
                                                              "launch_wall_utc": harness.T0_WALL - 4.5}))
    rec = harness._launch_record(tmp_path)
    assert rec["reexec"] is True and rec["pre_exec_s"] == 4.5
    (tmp_path / harness.PRE_EXEC_FILE).write_text(json.dumps({"nonce": "stale", "pid": 1}))
    assert "nonce" in harness._launch_record(tmp_path)["error"]


_REAL_CLI_DRIVER = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1] + "/src")
import importlib.util
spec = importlib.util.spec_from_file_location("tool", sys.argv[4])
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
h = getattr(tool, "harness", tool)   # wn3_forecast wraps the harness module
import gpuwrf.backends.platform as platform
from gpuwrf.runtime import gpu_allocator
from pathlib import Path
real_target = platform.target_backend
platform.target_backend = lambda environ=None: "cuda"          # pretend a CUDA host; JAX itself stays on CPU
gpu_allocator._pool_device_memory = lambda: (32 << 30, 30 << 30)
gpu_allocator.read_plan = lambda path, key, domains: {
    "budget_bytes": 6 << 30, "programs": {"d01": [{"generated_code_size_in_bytes": 1 << 20}]}}
class Exec(BaseException):
    pass
def fake_exec(file, argv, env):
    raise Exec(env)
os.execvpe = fake_exec
out = Path(sys.argv[3])
sys.argv = [h.__file__, "--input-dir", sys.argv[2], "--out", str(out), "--hours", "3", "--tag", "t"]
try:
    rc = h.main()   # the whole launch phase as run under with_gpu_lock
    print(json.dumps({"exec": None, "rc": rc}))
except Exec as e:
    env = e.args[0]
    keep = ("XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION",
            "_GPUWRF_C_AUTO", "_GPUWRF_C_AUTO_BUDGET", "_GPUWRF_NESTED_ALLOC_REEXEC", h.LAUNCH_ENV)
    print(json.dumps({"exec": {k: env.get(k) for k in keep},
                      "measured": h._allocator_record(env), "cli_argv": h.REC.get("cli_argv")}))
'''


def _synthetic_case(case: Path) -> None:
    netCDF4 = pytest.importorskip("netCDF4")
    case.mkdir()
    (case / "namelist.input").write_text(" &domains\n max_dom = 3,\n e_we = 9, 7, 7,\n /\n")
    for name, (nx, ny, dx) in {"d01": (8, 6, 9000.0), "d02": (6, 5, 3000.0), "d03": (6, 5, 1000.0)}.items():
        with netCDF4.Dataset(case / f"wrfinput_{name}", "w") as ds:
            ds.createDimension("west_east", nx)
            ds.createDimension("south_north", ny)
            ds.DX = dx
            ds.DY = dx


@pytest.mark.parametrize("tool", ["scripts/bench/s0_nested_harness.py", "scripts/wn3_forecast.py"])
def test_real_cli_pool_step_is_backend_dark_and_reexecs_with_c_auto(tmp_path, tool):
    """Harness main() → real gpuwrf.cli → configure_cli_pool on CPU: the NVML probe sees a dark backend
    (any jax device touch before it, the v0.4.1 harness bug, initializes the CPU backend here and fails) and
    the re-exec env carries the C-auto pool the measured process will initialize JAX with."""
    case, out = tmp_path / "case", tmp_path / "out"
    _synthetic_case(case)
    out.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("_GPUWRF_", "XLA_", "_S0_"))}
    env.update(JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", GPUWRF_JAX_CACHE_DIR=str(tmp_path / "jax"),
               XLA_PYTHON_CLIENT_ALLOCATOR="cuda_async", PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-c", _REAL_CLI_DRIVER, str(ROOT), str(case), str(out), str(ROOT / tool)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(r.stdout.strip().splitlines()[-1])
    ex = res["exec"]
    assert ex is not None, f"the CLI pool step did not re-exec: {res} {r.stderr[-2000:]}"
    assert ex["_GPUWRF_C_AUTO"] == "1" and ex["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert ex["_GPUWRF_C_AUTO_BUDGET"] == str(6 << 30) and ex["_GPUWRF_NESTED_ALLOC_REEXEC"] == "1"
    assert float(ex["XLA_CLIENT_MEM_FRACTION"]) == pytest.approx(6 / 32, abs=1e-6)  # budget / CUDA total
    assert res["measured"]["mode"] == "c_auto"
    assert json.loads(ex["_S0_HARNESS_LAUNCH"])["pid"] > 0
    pre = json.loads((out / "pre_exec.json").read_text())
    assert pre["nvml_probe"]["backend_dark"] is True and pre["allocator_step"]["backend_dark"] is True
    assert pre["nvml_probe"]["free_bytes"] == 30 << 30
    assert "C-auto case pool 6.00 GiB" in r.stderr
    if tool.endswith("wn3_forecast.py"):   # the plan key is computed for the 3-nest CLI argv
        assert res["cli_argv"][res["cli_argv"].index("--max-dom") + 1] == "3"
