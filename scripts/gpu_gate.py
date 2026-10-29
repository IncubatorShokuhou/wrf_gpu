"""CPU-only control plane for admitted GPU windows (stdlib; never imports JAX).

Launchers own the official GPU flock and scientific gates. This controller only
admits work, supervises its own process groups, and checks completion receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

SECRET = r"TOKEN|PASSW|SECRET|API_?KEY|EARTHDATA|CREDENTIAL|_COOKIE$"
STALE = r"^(W44R_|GPUWRF_|XLA_|JAX_|TF_)"
SELECTORS = {"SEQ_SMOKE", "PREFLIGHT_ONLY", "WRF_GPU2_SEQ_SMOKE",
             "W44R_LAUNCH_SMOKE", "LD_PRELOAD", "CUDA_ROOT", "CUDA_VISIBLE_DEVICES"}
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


class Refusal(Exception):
    pass


def require(ok, message):
    if not ok:
        raise Refusal(message)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    p = Path(path)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(p)


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "timestamp needs explicit timezone")
    return int(parsed.timestamp())


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now():
    # External commands deliberately permit private date/df/lslocks test doubles.
    return int(subprocess.check_output(["date", "-u", "+%s"], text=True))


def sleep(seconds, deadline=None):
    # Even admission sleeps are owned/reaped on a signal. In particular a lease
    # must not linger through a foreground 60s sample after TERM or hard expiry.
    require(run(["sleep", str(seconds)], {**clean_env(), "JAX_PLATFORMS": "cpu"},
                os.devnull, deadline) == 0, "admission wait expired")


def clean_env(extra=None, patterns=()):
    env = {**os.environ, **(extra or {})}
    for key in list(env):
        if key in SELECTORS or re.search(SECRET, key, re.I) or re.search(STALE, key) or any(
            re.search(p, key) for p in patterns
        ):
            del env[key]
    return env


def secret_scrub():
    # Before subprocesses, not just before profiling children. Never log values.
    for key in list(os.environ):
        if re.search(SECRET, key, re.I):
            del os.environ[key]


def lock_free(path):
    text = subprocess.check_output(["lslocks", "--noheadings", "--output", "PATH"], text=True)
    return all(row.strip() != str(path) for row in text.splitlines())


def free_bytes(path):
    lines = subprocess.check_output(["df", "--output=avail", "-B1", str(path)], text=True).splitlines()
    return int(lines[-1].strip())


def stop_group(proc):
    # Only the group created by this controller, never global GPU/lock PIDs.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run(command, env, output, deadline=None, tick=None, cancelled=None):
    proc = None
    try:
        with Path(output).open("a") as out:
            proc = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                if cancelled is not None and cancelled():
                    return 130
                if deadline is not None and now() >= deadline:
                    stop_group(proc)
                    return 124
                if tick:
                    tick()
                time.sleep(.05)
            if tick:
                tick()
            # A child can exit between polls after expiry; rc0 is insufficient.
            if deadline is not None and now() >= deadline:
                return 124
            return proc.returncode
    finally:
        if proc is not None:
            stop_group(proc)


@contextmanager
def terminal_signals():
    """Record termination without unwinding owned cleanup/result publication."""
    pending = [False]
    def defer(_sig, _frame):
        pending[0] = True
    previous = {sig: signal.signal(sig, defer)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield pending
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def plan_steps(path):
    steps = load(path)["steps"]
    require(isinstance(steps, list) and steps, "empty planned step set")
    names = []
    for s in steps:
        require(SAFE_NAME.fullmatch(s["lane"]), "invalid lane")
        names.append(s["lane"])
        require(type(s["cap_min"]) is int and s["cap_min"] >= 1, "cap_min must be integer >=1")
        require(isinstance(s.get("args", []), list) and all(isinstance(a, str) for a in s.get("args", [])), "args must be strings")
        require(isinstance(s.get("env", {}), dict), "env must be an object")
    require(len(set(names)) == len(names), "duplicate planned step")
    return steps


def child_env(step, smoke, run_dir):
    env = clean_env(step.get("env"))
    # The environment declared by the sealed plan is intentional, unlike stale
    # model selectors inherited from an interactive manager shell.
    env.update(step.get("env", {}))
    for key in list(env):
        if re.search(SECRET, key, re.I) or key in SELECTORS:
            del env[key]
    if smoke:
        env.update(JAX_PLATFORMS="cpu", WRF_GPU2_SEQ_SMOKE="1",
                   WRF_GPU2_SMOKE_DIR=str(run_dir),
                   WRF_GPU2_SMOKE_RECEIPT=str(run_dir / (step["lane"] + ".smoke.json")),
                   WRF_GPU2_REGISTRY_SMOKE="1")
    return env


def preflight(steps, smoke, run_dir):
    for s in steps:
        p = Path(s["path"])
        require(hashlib.sha256(p.read_bytes()).hexdigest() == s["sha256"], f"{s['lane']}: sha mismatch")
        require(subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0, "bash -n failed")
        code = "\n".join(l for l in p.read_text().splitlines() if not l.lstrip().startswith("#"))
        require("with_ram_slot" not in code, "GPU launcher must use own memory scope")
        require(s.get("args") or not re.search(r"\$\{1:\?|\$1\b", code), "launcher needs args")
        env = child_env(s, smoke, run_dir)
        require(shutil.which("python", path=env.get("PATH", "")), "child PATH lacks python")
        require(env.get("PYTHONPATH") and all(Path(p).is_dir() for p in env["PYTHONPATH"].split(os.pathsep)), "child PYTHONPATH missing/invalid")
        probe = ("import importlib.util,os,pathlib,sys; "
                 "roots=[pathlib.Path(p).resolve() for p in os.environ['PYTHONPATH'].split(os.pathsep)]; "
                 "specs=[importlib.util.find_spec(m) for m in sys.argv[1:]]; "
                 "sys.exit(0 if all(s and s.origin and any(pathlib.Path(s.origin).resolve().is_relative_to(r) for r in roots) for s in specs) else 2)")
        # Resolve imports with the actual child interpreter/env, without importing
        # model code or initializing CUDA. This also catches broken interpreters.
        require(run(["python", "-c", probe, *s.get("required_modules", ["gpuwrf"])],
                    {**env, "JAX_PLATFORMS": "cpu"}, run_dir / "environment.log", now() + 60) == 0,
                "child interpreter/module resolution failed")
        if smoke:
            require(isinstance(s.get("registry_smoke"), list) and s["registry_smoke"], "registry schema smoke required")
            require(isinstance(s.get("phases"), list) and s["phases"] and len(set(s["phases"])) == len(s["phases"]), "smoke phase inventory required")


def completion(text, steps):
    ends = re.findall(r"\bEND ([A-Za-z0-9_.-]+) rc=([0-9]+)\b", text)
    expected = Counter(s["lane"] for s in steps)
    return (Counter(l for l, _ in ends) == expected and all(r == "0" for _, r in ends)
            and not re.search(r"\bSKIP ", text))


def sequence(plan, run_dir, smoke=False, preflight_only=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = plan_steps(plan)
    preflight(steps, smoke, run_dir)
    print("PREFLIGHT OK", flush=True)
    if preflight_only:
        return 0
    end = epoch(os.environ.get("HARD_END_UTC", iso(now() + 7200)))
    start = now()
    os.environ.update(GA_START_UTC=iso(start), GA_END_UTC=iso(end))
    results = []
    for s in steps:
        lane = s["lane"]
        rc = 1
        if now() + (s["cap_min"] + 2) * 60 > end:
            print(f"SKIP {lane}: insufficient time", flush=True)
            results.append(dict(lane=lane, rc=1, status="SKIPPED"))
            continue
        env = child_env(s, smoke, run_dir)
        output = run_dir / (lane + ".log")
        if smoke:
            Path(env["WRF_GPU2_SMOKE_RECEIPT"]).unlink(missing_ok=True)
        # Recheck the actual sealed script immediately before dispatch.
        if hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() != s["sha256"]:
            print(f"SKIP {lane}: sha mismatch", flush=True)
            results.append(dict(lane=lane, rc=1, status="SKIPPED"))
            continue
        print(f"RUN {lane} {s['path']}", flush=True)
        deadline = min(end, now() + (s["cap_min"] + 1) * 60)
        if smoke:
            schema_command = [a.replace("{smoke_dir}", str(run_dir)) for a in s["registry_smoke"]]
            rc = run(schema_command, env, output, deadline)
        else:
            rc = 0
        if rc == 0:
            rc = run(["bash", s["path"], *s.get("args", [])], env, output, deadline)
        if smoke and rc == 0:
            try:
                receipt = load(env["WRF_GPU2_SMOKE_RECEIPT"])
                require(receipt.get("device") == "cpu" and receipt.get("real_lock_used") is False,
                        "smoke did not prove CPU/private lock")
                require(Counter(receipt.get("completed_phases", [])) == Counter(s["phases"]), "smoke omitted/duplicated phase")
                require(receipt.get("registry_schema_valid") is True and receipt.get("production_registry_writes") == 0,
                        "smoke registry schema/storage proof missing")
            except (Refusal, OSError, ValueError) as exc:
                print(f"SMOKE REFUSE {lane}: {exc}", flush=True)
                rc = 1
        rc = rc if rc >= 0 else 128 - rc
        print(f"END {lane} rc={rc}", flush=True)
        results.append(dict(lane=lane, rc=rc))
    save(run_dir / "sequence.json", dict(mode="smoke" if smoke else "run", steps=results,
                                       native_gpu_gates="NOT-APPLICABLE" if smoke else "launcher-owned"))
    return 0 if len(results) == len(steps) and all(r["rc"] == 0 for r in results) else 1


def quiet_cores(config, deadline=None):
    def read():
        rows = {}
        for line in Path(config.get("stat_path", "/proc/stat")).read_text().splitlines():
            f = line.split()
            if re.fullmatch(r"cpu\d+", f[0]):
                v = list(map(int, f[1:]))[:8]  # guest times already included
                rows[int(f[0][3:])] = (sum(v), v[3] + v[4])
        return rows
    before = read()
    sleep(60, deadline)
    after = read()
    for cpu in map(int, config["cpus"].split(",")):
        total = after[cpu][0] - before[cpu][0]
        require(total > 0 and 1 - (after[cpu][1] - before[cpu][1]) / total <= .5,
                f"CPU {cpu} not quiet over 60s")


def gate(config):
    c = load(config)
    for field in ("not_before", "not_after", "hard_end", "go_glob", "cross_dir", "run_dir", "plan", "lease", "lock_path", "disk_path", "disk_floor_bytes", "disk_need_bytes", "cpu_lease"):
        env_key = "GATE_" + field.upper()
        if env_key in os.environ:
            c[field] = json.loads(os.environ[env_key]) if field in ("disk_floor_bytes", "disk_need_bytes", "cpu_lease") else os.environ[env_key]
        require(field in c, f"missing config {field}")
    rd = Path(c["run_dir"])
    rd.mkdir(parents=True, exist_ok=False)  # one immutable invocation directory
    cross = Path(c["cross_dir"])
    nb, latest, he = (epoch(c[k]) for k in ("not_before", "not_after", "hard_end"))
    require(nb <= latest < he, "invalid window")
    require(all(type(c[k]) is int and c[k] >= 0 for k in ("disk_floor_bytes", "disk_need_bytes")),
            "disk floor/need must be nonnegative integer bytes")
    steps = plan_steps(c["plan"])
    frozen = rd / "plan.json"
    save(frozen, {"steps": steps})
    need_time = sum((s["cap_min"] + 2) * 60 for s in steps)
    gid = "gate_" + uuid.uuid4().hex
    env = clean_env(patterns=c.get("scrub_patterns", []))
    env.update(GA_GATE_ID=gid, SEQ_RUN_DIR=str(rd / "sequence"))
    seq = c.get("sequencer", ["bash", str(Path(__file__).with_name("night_seq_v10.sh"))])
    output = rd / "invocation.log"
    output.touch()
    acquired = False
    launched = False
    status, rc, detail = "NOSTART", 4, "not admitted"
    cpu = c["cpu_lease"]
    env["CPU_LEASE_DIR"] = str(Path(cpu["directory"]).parent)
    lease_path = Path(cpu["directory"]) / (gid + "-cpu.lease")
    require(not lease_path.exists(), "CPU lease name collision")
    notices = set()

    def bounds():
        t = now()
        require(nb <= t <= latest, "outside start window")
        require(not any(epoch(a) <= t < epoch(b) for a, b in c.get("blocked_intervals", [])), "blocked interval")
        require(t + need_time <= he, "whole plan does not fit hard end")
        require(lock_free(c["lock_path"]), "GPU lock occupied")

    def resources():
        text = Path(c["lease"]).read_text()
        exp = re.findall(r"^expires-utc:\s*(\S+)", text, re.M)
        require(len(exp) == 1 and epoch(exp[0]) >= he, "missing/short GPU lease")
        free = free_bytes(c["disk_path"])
        require(free - c["disk_need_bytes"] >= c["disk_floor_bytes"], "disk byte floor")
        return free

    def notice_tick():
        text = output.read_text()
        for i, hook in enumerate(c.get("notice_hooks", [])):
            if i in notices or initial_free >= hook["min_free_bytes"]:
                continue
            lane = re.escape(hook["lane"])
            if not re.search(rf"\bRUN {lane}(?:\s|$)", text) or re.search(rf"\bEND {lane} rc=", text):
                continue
            for name in sorted(glob.glob(hook["entry_glob"], recursive=True)):
                try:
                    entry = load(name)
                    if (entry.get("entry") != hook["entry"] or entry.get("mode") != "gpu"
                            or entry.get("env", {}).get("GA_GATE_ID") != gid):
                        continue
                    session = Path(name).with_name("session.log")
                    if re.search(hook["pattern"], session.read_text()):
                        result = run([*hook["command"], name, gid], env, rd / "notice.log", min(he, now() + 10))
                        require(result == 0, "notice hook failed")
                        notices.add(i)
                        break
                except (OSError, ValueError):
                    continue

    def release(cancelled=None):
        nonlocal acquired
        if acquired:
            if not lease_path.exists():
                acquired = False
                return
            release_rc = run([*cpu["command"], "release", gid, "gate end/abort/expiry"],
                             env, rd / "cpu_lease.log", now() + 20, cancelled=cancelled)
            if release_rc == 0 and not lease_path.exists():
                acquired = False
            else:
                raise Refusal("CPU lease release failed (ownership retained)")

    try:
        while True:
            t = now()
            require(t <= latest and t < he, "start window expired")
            triggers = sorted(glob.glob(str(cross / c["go_glob"])))
            if t >= nb and triggers and lock_free(c["lock_path"]) and not any(
                epoch(a) <= t < epoch(b) for a, b in c.get("blocked_intervals", [])
            ):
                # Every matching trigger may only shorten the granted endpoint.
                for trigger in triggers:
                    for override in re.findall(r"HARD_END_UTC=(\S+)", Path(trigger).read_text()):
                        he = min(he, epoch(override))
                break
            sleep(c.get("poll_seconds", 20), min(latest + 1, he))
        bounds()
        initial_free = resources()
        env.update(HARD_END_UTC=iso(he))
        pfenv = {**env, "PREFLIGHT_ONLY": "1"}
        require(run([*seq, str(frozen)], pfenv, rd / "preflight.log", he) == 0, "sequencer preflight failed")
        bounds()
        # Mark before acquire so cancellation during the helper's 30-second wait
        # also releases THIS unique lease. No foreign lease name is ever used.
        acquired = True
        require(run([*cpu["command"], "acquire", gid, cpu["cpus"], iso(he),
                     "admitted timed GPU window"], env, rd / "cpu_lease.log", he) == 0,
                "CPU lease acquisition failed")
        require(lease_path.is_file(), "CPU helper did not publish lease")
        lease_start = re.findall(r"^start-utc:\s*(\S+)", lease_path.read_text(), re.M)
        require(len(lease_start) == 1, "CPU lease start missing")
        remaining = 30 - (now() - epoch(lease_start[0]))
        if remaining > 0:
            sleep(remaining, he)
        bounds()  # do not spend another minute sampling after a late acquire
        quiet_cores(cpu, he)
        bounds()
        resources()
        launched = True
        rc = run([*seq, str(frozen)], env, output, he, notice_tick)
        require(rc == 0 and completion(output.read_text(), steps), "planned completion mismatch/nonzero/skipped")
        detail = "every planned step exactly once rc=0"
        status = "DONE"
    except (Refusal, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        detail = str(exc)
        status = "STOPPED" if launched else "NOSTART"
        rc = 1 if launched else 4
    except KeyboardInterrupt:
        status, rc, detail = "STOPPED" if launched else "NOSTART", 130, "interrupted"
    finally:
        with terminal_signals() as pending:
            # Interrupt/reap the first helper, then protect its bounded retry
            # and terminal receipt from further SIGINT/TERM/HUP unwinding.
            for attempt in range(2):
                try:
                    release(cancelled=(lambda: pending[0]) if attempt == 0 else None)
                    break
                except (Refusal, OSError, subprocess.SubprocessError) as exc:
                    status, rc, detail = "STOPPED", 1, str(exc)
            if pending[0]:
                status, rc, detail = "STOPPED", 130, "interrupted during terminal release; " + detail
            result = dict(status=status, rc=rc, detail=detail, gate_id=gid,
                          cpu_lease_retained=acquired, notices_sent=len(notices))
            save(rd / "result.json", result)
            name = c.get("cross_names", {}).get(status, f"{gid}_{status}.md")
            require(Path(name).name == name, "cross name must be a basename")
            with (cross / name).open("x") as f:
                f.write(f"# {status} {gid}\n{detail}\n")
            print(json.dumps(result), flush=True)
    return rc


def main(argv=None):
    secret_scrub()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gate", "sequence"))
    parser.add_argument("config_or_plan")
    args = parser.parse_args(argv)
    def interrupted(_sig, _frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        if args.mode == "gate":
            return gate(args.config_or_plan)
        return sequence(args.config_or_plan, Path(os.environ["SEQ_RUN_DIR"]),
                        os.environ.get("SEQ_SMOKE") == "1", os.environ.get("PREFLIGHT_ONLY") == "1")
    except (Refusal, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
