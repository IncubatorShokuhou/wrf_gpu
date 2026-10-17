"""scripts/run_parallel_cases.sh --gpus (ensemble mode, lane o2-multigpu M1): per-GPU binding, sizing and admission,
failure isolation, per-member checkpoint stores and retries. CPU only: fake devices, fake CLI, fake nvidia-smi."""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GIB = 1024**3


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name + "_mg", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = _load("parallel_cases")

# The fake case process records the GPU binding it got and can fail once / write a checkpoint generation.
FAKE_CLI = r'''
import json, os, sys, time, pathlib
a = sys.argv
out = pathlib.Path(a[a.index("--output-dir") + 1]); out.mkdir(parents=True, exist_ok=True)
name = pathlib.Path(a[a.index("--input-dir") + 1]).name
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
(out / ("binding_" + str(os.getpid()) + ".json")).write_text(json.dumps(
    {"cvd": cvd, "order": os.environ.get("CUDA_DEVICE_ORDER", "<unset>"), "argv": a[1:]}))
flag = os.environ.get("FAKE_PLAN_FLAG")
if flag:
    pathlib.Path(flag + "." + cvd).write_text("plan")
resume = a[a.index("--resume-checkpoint") + 1] if "--resume-checkpoint" in a else None
if "--checkpoint-dir" in a and not resume:
    gen = pathlib.Path(a[a.index("--checkpoint-dir") + 1]) / "generation-000001"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "manifest.json").write_text(json.dumps({"created_ns": time.time_ns()}))
time.sleep(float(os.environ.get("FAKE_SLEEP_" + name, os.environ.get("FAKE_SLEEP", "1.0"))))
if os.environ.get("FAKE_RC_" + name) and not resume and not (out.parent / "failed-attempt-0").exists():
    sys.exit(int(os.environ["FAKE_RC_" + name]))  # first attempt fails (a resumed or fresh second attempt succeeds)
(out / "wrfout_d01_2026-01-01_00:00:00").write_bytes(b"x")
print(json.dumps({"effective_hours": 2, "all_domains_finite": True, "all_outputs_present": True}))
'''


def _devs(n=2, free_gib=20.0, caps=None):
    caps = caps or ["8.9"] * n
    return [{"index": str(i), "uuid": f"GPU-{i:04d}", "name": "NVIDIA GeForce RTX 4090", "compute_cap": caps[i],
             "total_bytes": 24 * GIB, "free_bytes": int(free_gib * GIB)} for i in range(n)]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    cli = tmp_path / "fake_cli.py"
    cli.write_text(FAKE_CLI)
    cases = []
    for name in ("c1", "c2", "c3", "c4"):
        (tmp_path / "cases" / name).mkdir(parents=True)
        cases.append(str(tmp_path / "cases" / name))
    monkeypatch.setenv("GPUWRF_GPU_LOCK_HELD", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setattr(pc, "vram_by_pid", lambda: {})
    for key in pc.EXPLICIT_ALLOCATOR_ENV:
        monkeypatch.delenv(key, raising=False)
    base = ["--out-root", str(tmp_path / "out"), "--gpu-log-interval", "0", "--min-mem-available-gb", "0",
            "--host-gb-per-case", "0.001", "--cli-json", json.dumps([sys.executable, str(cli)])]
    return tmp_path, cases, base


def _bindings(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out.glob("binding_*.json"))]


def _receipt(tmp: Path, name: str) -> dict:
    return json.loads((tmp / "out" / name / "receipt.json").read_text())


def test_select_devices_refuses_unknown_duplicate_and_mixed_capabilities():
    devs = _devs(3)
    assert [d["index"] for d in pc.select_devices("all", devs)] == ["0", "1", "2"]
    assert [d["index"] for d in pc.select_devices("2, 0", devs)] == ["2", "0"]
    for bad in ("5", "0,0", ""):
        with pytest.raises(ValueError):
            pc.select_devices(bad, devs)
    with pytest.raises(ValueError, match="compute capabilities"):
        pc.select_devices("all", _devs(2, caps=["8.9", "12.0"]))


def test_gpu_devices_parses_nvidia_smi(tmp_path, monkeypatch):
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.parent.mkdir()
    smi.write_text("#!/bin/sh\necho '0, GPU-aaa, 24564, 330, 24000, 8.9, NVIDIA GeForce RTX 4090'\n"
                   "echo '1, GPU-bbb, 24564, 330, 1000, 8.9, NVIDIA GeForce RTX 4090'\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{smi.parent}:{os.environ['PATH']}")
    devs = pc.gpu_devices()
    assert [(d["index"], d["uuid"], d["compute_cap"]) for d in devs] == [("0", "GPU-aaa", "8.9"), ("1", "GPU-bbb", "8.9")]
    assert devs[0]["total_bytes"] == (24564 - 330) * 1024**2 and devs[1]["free_bytes"] == 1000 * 1024**2
    assert devs[0]["name"] == "NVIDIA GeForce RTX 4090"


def test_fits_device_is_fits_for_one_gpu():
    """Without --gpus the launcher admits through fits_device with one device key: it must equal the v0.3 fits()."""
    rng = random.Random(7)

    def case(need, host, cold):
        c = pc.Case(Path("/x"), "x", Path("/o"))
        c.sizing, c.host_kb, c.host_cold_kb = {"need_bytes": need}, host, cold
        return c
    for _ in range(3000):
        pick = lambda: case(rng.choice([None, GIB, 3 * GIB, 7 * GIB]), rng.randint(1, 9), rng.randint(0, 12))  # noqa: E731
        running = [pick() for _ in range(rng.randint(0, 3))]
        cand, cap, host, mx = pick(), rng.choice([0, 4, 8, 20]) * GIB, rng.randint(0, 40), rng.randint(1, 4)
        assert pc.fits_device(running, cand, None, cap, host, mx, 1 << 30) == pc.fits(running, cand, cap, host, mx)


def test_two_gpus_bind_each_case_and_balance_placement(fake):
    tmp, cases, base = fake
    calls = []

    def sizer(ds, a, e):
        calls.append((e.get("CUDA_VISIBLE_DEVICES"), e.get("CUDA_DEVICE_ORDER")))
        return [{"source": "c-auto-plan", "need_bytes": 8 * GIB} for _ in ds]
    rc = pc.main([*base, "--gpus", "all", *cases, "--", "--hours", "2"], sizer=sizer, devices=lambda: _devs(2, free_gib=17))
    assert rc == 0
    assert sorted(set(calls)) == [("0", "PCI_BUS_ID"), ("1", "PCI_BUS_ID")]  # per-GPU plan keys
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    assert [(e["start"], e["gpu"]) for e in run["events"]] == [("c1", "0"), ("c2", "1"), ("c3", "0"), ("c4", "1")]
    assert run["peak_concurrency"] == 4 and [d["index"] for d in run["devices"]] == ["0", "1"]
    for name in ("c1", "c2", "c3", "c4"):
        rec = _receipt(tmp, name)
        gpu = rec["gpu"]["index"]
        assert rec["rc"] == 0 and run["cases"][name]["gpu"] == gpu
        (binding,) = _bindings(tmp / "out" / name / "wrfout")
        assert binding == {"cvd": gpu, "order": "PCI_BUS_ID", "argv": binding["argv"]}  # the process saw exactly its GPU
        assert f"CUDA_VISIBLE_DEVICES={gpu}" in rec["env"] and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in rec["env"]


def test_vram_is_admitted_per_gpu(fake):
    tmp, cases, base = fake
    sizer = lambda ds, a, e: [{"need_bytes": 8 * GIB} for _ in ds]  # noqa: E731
    rc = pc.main([*base, "--gpus", "0,1", *cases[:3]], sizer=sizer, devices=lambda: _devs(2, free_gib=9))
    assert rc == 0
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    # one 8 GiB case per 9 GiB GPU: c3 waits for a GPU to free up
    assert [e["gpu"] for e in run["events"]][:2] == ["0", "1"] and run["peak_concurrency"] == 2
    assert run["events"][2]["t_s"] >= 0.9


def test_sizing_run_is_exclusive_on_its_gpu_only(fake, monkeypatch):
    tmp, cases, base = fake
    flag = tmp / "plan"
    monkeypatch.setenv("FAKE_PLAN_FLAG", str(flag))

    def sizer(ds, a, e):
        have = Path(f"{flag}.{e['CUDA_VISIBLE_DEVICES']}").exists()
        return [{"source": "c-auto-plan", "need_bytes": GIB} if have else {"source": "sizing-run", "need_bytes": None}
                for _ in ds]
    rc = pc.main([*base, "--gpus", "all", *cases], sizer=sizer, devices=lambda: _devs(2))
    assert rc == 0
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    # both GPUs size concurrently (one case each), then the rest run next to each other
    assert [(e["start"], e["gpu"], e["running"]) for e in run["events"]][:2] == [("c1", "0", 1), ("c2", "1", 2)]
    assert run["events"][2]["t_s"] >= 0.9
    assert _receipt(tmp, "c1")["sizing"]["source"] == "sizing-run" and _receipt(tmp, "c4")["sizing"]["source"] == "c-auto-plan"


def test_failed_member_does_not_stop_the_others_and_resumes_from_its_own_checkpoint(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setenv("FAKE_RC_c1", "1")
    monkeypatch.setenv("FAKE_SLEEP_c2", "4.0")
    sizer = lambda ds, a, e: [{"need_bytes": GIB} for _ in ds]  # noqa: E731
    rc = pc.main([*base, "--gpus", "all", "--checkpoint-interval-steps", "50", "--retry-failed", "1", *cases[:2]],
                 sizer=sizer, devices=lambda: _devs(2))
    assert rc == 0
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    retry = [e for e in run["events"] if "retry" in e]
    own_gen = str(tmp / "out" / "c1" / "checkpoint" / "generation-000001")
    assert retry == [{"t_s": retry[0]["t_s"], "retry": "c1", "attempt": 1, "resume_from": own_gen}]
    assert retry[0]["t_s"] < 3.9  # c2 (4 s) was still running: the failure stopped nothing
    first = json.loads((tmp / "out" / "c1" / "attempt-0.receipt.json").read_text())
    final = _receipt(tmp, "c1")
    assert first["rc"] == 1 and first["attempt"] == 0 and final["rc"] == 0 and final["attempt"] == 1
    assert final["command"][-2:] == ["--resume-checkpoint", own_gen] and final["resumed_from"] == own_gen
    assert run["cases"]["c1"]["attempts"] == 2 and run["cases"]["c2"] == run["cases"]["c2"] | {"rc": 0, "attempts": 1}
    for name in ("c1", "c2"):  # every member checkpoints into its own store
        rec = _receipt(tmp, name)
        i = rec["command"].index("--checkpoint-dir")
        assert rec["command"][i:i + 4] == ["--checkpoint-dir", str(tmp / "out" / name / "checkpoint"),
                                           "--checkpoint-interval-steps", "50"]


def test_retry_without_checkpoint_moves_the_failed_history_aside(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setenv("FAKE_RC_c1", "3")
    rc = pc.main([*base, "--retry-failed", "2", cases[0]], sizer=lambda ds, a, e: [{"need_bytes": GIB}], gpu=lambda: (32 * GIB, 30 * GIB))
    assert rc == 0
    out = tmp / "out" / "c1"
    assert len(_bindings(out / "failed-attempt-0" / "wrfout")) == 1 and len(_bindings(out / "wrfout")) == 1
    assert (out / "wrfout" / "wrfout_d01_2026-01-01_00:00:00").exists()
    assert _receipt(tmp, "c1")["resumed_from"] is None and "--resume-checkpoint" not in _receipt(tmp, "c1")["command"]


def test_failures_are_isolated_and_refusals_are_not_retried(fake, monkeypatch):
    tmp, cases, base = fake
    monkeypatch.setenv("FAKE_RC_c1", "2")  # CLI refusal: deterministic, never retried
    monkeypatch.setenv("FAKE_RC_c2", "1")  # runtime failure, no retries requested
    rc = pc.main([*base, "--gpus", "all", "--retry-failed", "1", *cases[:3]],
                 sizer=lambda ds, a, e: [{"need_bytes": GIB} for _ in ds], devices=lambda: _devs(2))
    assert rc == 1
    run = json.loads((tmp / "out" / "parallel_run.json").read_text())
    assert run["cases"]["c1"]["rc"] == 2 and run["cases"]["c1"]["attempts"] == 1
    assert run["cases"]["c2"]["rc"] == 0 and run["cases"]["c2"]["attempts"] == 2  # retried from scratch
    assert run["cases"]["c3"]["rc"] == 0 and run["stop_reason"] is None


def test_gpus_refusals(fake, monkeypatch):
    tmp, cases, base = fake
    sizer = lambda ds, a, e: [{"need_bytes": GIB} for _ in ds]  # noqa: E731
    assert pc.main([*base, "--gpus", "all", cases[0]], sizer=sizer, devices=lambda: _devs(2, caps=["8.9", "9.0"])) == 2
    assert pc.main([*base, "--gpus", "7", cases[0]], sizer=sizer, devices=lambda: _devs(2)) == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert pc.main([*base, "--gpus", "all", cases[0]], sizer=sizer, devices=lambda: _devs(2)) == 2
    assert not (tmp / "out").exists()


def test_without_gpus_the_case_env_is_the_callers(fake):
    tmp, cases, base = fake
    rc = pc.main([*base, cases[0]], sizer=lambda ds, a, e: [{"need_bytes": GIB}], gpu=lambda: (32 * GIB, 30 * GIB),
                 devices=lambda: pytest.fail("no device discovery without --gpus"))
    assert rc == 0
    (binding,) = _bindings(tmp / "out" / "c1" / "wrfout")
    assert binding["cvd"] == "<unset>" and binding["order"] == "<unset>"
    rec = _receipt(tmp, "c1")
    assert "gpu" not in rec and "--checkpoint-dir" not in rec["command"]
    assert "devices" not in json.loads((tmp / "out" / "parallel_run.json").read_text())


def test_dry_run_reports_the_per_gpu_plan(fake, capsys):
    tmp, cases, base = fake
    rc = pc.main([*base, "--gpus", "all", "--dry-run", *cases], sizer=lambda ds, a, e: [{"need_bytes": 8 * GIB} for _ in ds],
                 devices=lambda: _devs(2, free_gib=17))
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["would_start_now"] == 4 and [c["gpu"] for c in plan["cases"]] == ["0", "1", "0", "1"]
    assert [d["capacity_bytes"] for d in plan["devices"]] == [17 * GIB, 17 * GIB]
