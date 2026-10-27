"""Queue admission, correct namelist binding, and conservative cache reuse."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gpuwrf.cli import build_parser
from gpuwrf.runtime import precompile_cli as pc


@pytest.fixture
def queue(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    (case / "wrfinput_d01").write_bytes(b"input")
    (case / "wrfbdy_d01").write_bytes(b"boundary")
    (case / "namelist.input").write_text("&domains\n max_dom=1,\n time_step=60,\n/\n")
    other = tmp_path / "ysu.input"
    other.write_text("&domains\n max_dom=1,\n time_step=54,\n/\n&physics\n bl_pbl_physics=1,\n sf_sfclay_physics=1,\n/\n")
    manifest = tmp_path / "variants.json"
    manifest.write_text(json.dumps({"variants": [{"name": "ysu", "namelist": other.name}]}))
    args = build_parser().parse_args(["precompile", "--namelist-variants", str(manifest),
        "--input-dir", str(case), "--cache-dir", str(tmp_path / "cache"), "--work-dir", str(tmp_path / "work")])
    return args, case, other


def test_queue_is_backend_dark(queue, monkeypatch):
    args, case, _ = queue
    monkeypatch.setattr(pc, "source_identity", lambda: {"test": "source"})
    monkeypatch.setattr(pc, "target_identity", lambda: pytest.fail("queue queried GPU"))
    monkeypatch.setattr(pc, "run_worker", lambda *a: pytest.fail("queue executed a model"))
    assert pc.command(args) == 0
    report = json.loads((args.work_dir / "report.json").read_text())
    assert report["variants"][0]["status"] == "QUEUED"
    assert report["deviceless"] == "NOT_QUALIFIED"
    assert report["execute_argv"][-1] == "--execute"
    assert report["execute_argv"][report["execute_argv"].index("--input-dir")+1] == str(case)
    assert not args.cache_dir.exists()
    assert not list(case.glob("*.log"))


@pytest.mark.parametrize("data", [{"variants": []}, {"variants": [{"name": "a", "namelist": "x", "physics": {}}]},
                                  {"variants": [{"name": "../a", "namelist": "x"}]}, {"variant": []}])
def test_manifest_refuses_unknown_or_empty_variants(tmp_path, data):
    path = tmp_path / "variants.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        pc.variants(path)


def test_duplicate_names_refuse(queue):
    args, _, _ = queue
    data = json.loads(args.namelist_variants.read_text())
    data["variants"] *= 2
    args.namelist_variants.write_text(json.dumps(data))
    assert pc.command(args) == 2
    assert not args.work_dir.exists()


def test_namelist_is_bound_to_actual_case(queue, tmp_path):
    _, case, other = queue
    (case / "wrfout_d01_cpu").write_bytes(b"do not turn native priming into replay")
    with pc.case_view(case, other, tmp_path) as view:
        assert (view / "namelist.input").read_bytes() == other.read_bytes()
        assert not (view / "namelist.input").is_symlink()
        assert (view / "wrfinput_d01").resolve() == (case / "wrfinput_d01").resolve()
        assert not list(view.glob("wrfout*"))
    assert "time_step=60" in (case / "namelist.input").read_text()


def test_wn3_hour_alignment_refuses_before_work(queue):
    args, _, _ = queue
    args.hours = 1
    assert pc.command(args) == 2
    assert not args.work_dir.exists()


def test_cache_requires_complete_identical_native_artifacts(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    native, meta = cache / "a.xlaexec", cache / "a.meta"
    native.write_bytes(b"native")
    meta.write_bytes(b"metadata")
    entry = {"identity": "id", "status": "BUILT", "files": pc.cache_files(cache)}
    assert pc.cached(entry, "id", cache)
    assert not pc.cached(entry, "new-source-or-input-or-target-or-flags", cache)
    meta.unlink()
    assert not pc.cached(entry, "id", cache)
    meta.write_bytes(b"metadata")
    native.write_bytes(b"corrupt")
    assert not pc.cached(entry, "id", cache)
    assert not pc.cached({**entry, "files": {}}, "id", cache)


def test_idempotent_same_scheme_deduplicates_and_partial_failure_is_reported(queue, monkeypatch):
    args, _, other = queue
    args.execute = True
    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    monkeypatch.setenv("GPUWRF_QUEUE_TEST_SECRET", "private-value-must-not-be-persisted")
    monkeypatch.setattr(pc, "lock_fd", lambda: 9)
    monkeypatch.setattr(pc, "source_identity", lambda: {"test": "source"})
    monkeypatch.setattr(pc, "target_identity", lambda: {"test": "gpu"})
    data = json.loads(args.namelist_variants.read_text())
    data["variants"].append({**data["variants"][0], "name": "same-scheme"})
    args.namelist_variants.write_text(json.dumps(data))
    calls = []
    def fake_worker(request, log, timeout, fd, env):
        data = json.loads(request.read_text())
        assert env["JAX_COMPILATION_CACHE_DIR"] == str(args.cache_dir)
        assert Path(data["argv"][2], "namelist.input").read_bytes() == other.read_bytes()
        args.cache_dir.mkdir(exist_ok=True)
        (args.cache_dir / "compiled.xlaexec").write_bytes(b"executable")
        (args.cache_dir / "compiled.meta").write_bytes(b"contract")
        pc._write(Path(data["result"]), {"rc": 0, "platform": "gpu", "aot_programs": {"d01": "fixture"}})
        calls.append(data)
        return 0
    monkeypatch.setattr(pc, "run_worker", fake_worker)
    assert pc.command(args) == 0
    assert len(calls) == 1
    assert "private-value-must-not-be-persisted" not in (args.work_dir / "index.json").read_text()
    (args.input_dir / "wrfbdy_d01").write_bytes(b"new boundary")
    assert pc.command(args) == 0
    assert len(calls) == 2
    assert [v["status"] for v in json.loads((args.work_dir / "report.json").read_text())["variants"]] == ["BUILT", "CACHED"]
    assert pc.command(args) == 0
    assert len(calls) == 2
    (args.cache_dir / "compiled.meta").unlink()
    monkeypatch.setattr(pc, "run_worker", lambda *a: 1)
    assert pc.command(args) == 1
    assert all(v["status"] == "FAILED" for v in json.loads((args.work_dir / "report.json").read_text())["variants"])


def test_execute_refuses_without_official_lock(queue, monkeypatch):
    args, _, _ = queue
    args.execute = True
    monkeypatch.delenv("GPUWRF_GPU_LOCK_HELD", raising=False)
    assert pc.command(args) == 2
    assert not args.work_dir.exists()


def test_auxiliary_input_changes_invalidate_admission(queue):
    _, case, _ = queue
    aux = case / "forcing"
    aux.mkdir()
    (aux / "auxinput4").write_bytes(b"old SST")
    old = pc.input_files(case)
    (aux / "auxinput4").write_bytes(b"new SST")
    assert old != pc.input_files(case)


def test_timeout_reaps_own_process_group(tmp_path, monkeypatch):
    original = subprocess.Popen
    pidfile = tmp_path / "pids"
    code = f"""
import os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
p = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
Path({str(pidfile)!r}).write_text(str(os.getpid())+' '+str(p.pid))
time.sleep(60)
"""
    def innocent_child(*args, **kwargs):
        assert kwargs.get("pass_fds") == (9,)
        return original([sys.executable, "-c", code], stdout=kwargs["stdout"], stderr=kwargs["stderr"],
                        start_new_session=True)
    monkeypatch.setattr(pc.subprocess, "Popen", innocent_child)
    with pytest.raises(subprocess.TimeoutExpired):
        pc.run_worker(tmp_path / "unused", tmp_path / "log", .3, 9, os.environ)
    assert pidfile.exists()
    for pid in pidfile.read_text().split():
        stat = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 2
        while stat.exists() and stat.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(.01)
        assert not stat.exists() or stat.read_text().split()[2] == "Z"


def test_external_runtime_tables_invalidate_receipts(tmp_path, monkeypatch):
    from gpuwrf import assets
    package = tmp_path / "gpuwrf"
    (package / "runtime").mkdir(parents=True)
    (package / "__init__.py").write_text("# unchanged source")
    table = tmp_path / "external.npz"
    table.write_bytes(b"old table")
    monkeypatch.setattr(pc, "__file__", str(package / "runtime" / "precompile_cli.py"))
    monkeypatch.setattr(assets, "manifest", lambda: {"assets": {"external.npz": {"sha256": pc.sha_file(table)}}})
    monkeypatch.setattr(assets, "asset_path", lambda rel: table)
    before = pc.source_identity()
    table.write_bytes(b"new table")
    assert pc.source_identity() != before


def test_help_does_not_initialize_backend():
    code = """
import jax._src.xla_bridge as xb
def refuse(*a, **k): raise AssertionError('backend initialized')
xb.get_backend = refuse
from gpuwrf.cli import main
main(['precompile', '--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "JAX_PLATFORMS": "cpu", "GPUWRF_JAX_CACHE": "0",
                                 "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--namelist-variants" in result.stdout
