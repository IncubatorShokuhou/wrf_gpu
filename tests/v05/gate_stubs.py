"""Private command doubles for test_gpu_gate.py. No operational paths."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(os.environ["PRIVATE_CASE"])
    c = json.loads((root / "case.json").read_text())
    cmd, *args = sys.argv[1:]
    def clock():
        return int((root / "clock").read_text())
    def stamp(t):
        return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    def record(event):
        with (root / "calls.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
    if cmd == "date":
        print(clock())
    elif cmd == "df":
        print("Avail\n" + str(int(c.get("free_gib", 430) * 2**30)))
    elif cmd == "lslocks":
        if (root / "locked").read_text() == "1":
            print(str(root / "gpu.lock"))
    elif cmd == "sleep":
        record(dict(mode="sleep", seconds=float(args[0])))
        if c.get("signal_quiet") and float(args[0]) == 60:
            (root / "sampling").touch()
            time.sleep(30)
        # /proc/stat double advances idle ticks, or busy ticks for refusal.
        stat = root / "stat"
        fields = stat.read_text().split()
        fields[1 if c.get("busy_cpu") else 4] = str(int(fields[1 if c.get("busy_cpu") else 4]) + 100)
        stat.write_text(" ".join(fields) + "\n")
        q = c.get("clock_queue", [])
        counter = root / "sleep_count"
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n + 1))
        if q:
            (root / "clock").write_text(str(q[min(n, len(q)-1)]))
        if c.get("unlock_after_sleep"):
            (root / "locked").write_text("0")
        if c.get("trigger_after_sleep"):
            (root / "cross/GO.md").write_text("GO\n")
        if c.get("expire_quiet") and float(args[0]) == 60:
            (root / "clock").write_text(str(c["end_epoch"]))
        time.sleep(.002)
    elif cmd == "lease":
        action, name, *rest = args
        path = root / "leases" / (name + "-cpu.lease")
        record(dict(mode=action, name=name, time=clock()))
        if action == "acquire":
            if c.get("acquire_fail"):
                return 3
            path.write_text("start-utc: " + stamp(clock() if c.get("fresh_lease") else clock()-30) + "\nexpires-utc: " + rest[1] + "\n")
            if c.get("signal_acquire"):
                (root / "acquiring").touch()
                time.sleep(30)
            if c.get("acquire_advance"):
                (root / "clock").write_text(str(c["acquire_advance"]))
        elif action == "release":
            if c.get("signal_release") and not (root / "releasing").exists():
                (root / "releasing").touch()
                time.sleep(30)
            if c.get("release_fail"):
                return 42
            path.unlink(missing_ok=True)
    elif cmd == "notice":
        entry, gid = args
        record(dict(mode="notice", entry=entry, gid=gid))
        (root / "notice").write_text(entry)
        if c.get("hang_notice"):
            (root / "clock").write_text(str(c["end_epoch"]))
            time.sleep(30)
    elif cmd == "seq":
        if os.environ.get("PREFLIGHT_ONLY"):
            record(dict(mode="preflight", time=clock()))
            if c.get("preflight_advance"):
                (root / "clock").write_text(str(c["preflight_advance"]))
            if c.get("lock_after_preflight"):
                (root / "locked").write_text("1")
            return c.get("preflight_rc", 0)
        keys = ("SEQ_SMOKE", "WRF_GPU2_SEQ_SMOKE", "PREFLIGHT_ONLY", "GPUWRF_STALE", "JAX_STALE", "W44R_STALE", "XLA_STALE", "TF_STALE", "CUDA_ROOT", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR", "HOME", "LD_LIBRARY_PATH", "HARD_END_UTC", "PATH", "PYTHONPATH", "GA_GATE_ID", "RELEASE_APIKEY")
        record(dict(mode="actual", time=clock(), env={k: os.environ.get(k) for k in keys}))
        plan = json.loads(Path(args[0]).read_text())["steps"]
        if c.get("hang"):
            (root / "clock").write_text(str(c["end_epoch"]))
            time.sleep(30)
        for s in plan:
            print("RUN " + s["lane"] + " private", flush=True)
        if c.get("watch"):
            gid = os.environ["GA_GATE_ID"]
            # These must NEVER cause premature delivery.
            if c.get("foreign_ENTRY_ID"):
                f = root / "entries/foreign"; f.mkdir()
                (f / "ENTRY.json").write_text(json.dumps(dict(entry="rw3", mode="gpu", env={"GA_GATE_ID": "wrong"})))
                (f / "session.log").write_text("ARM O3 rc=0 try 1\n")
                time.sleep(.15)
                assert not (root / "notice").exists(), "foreign entry fired notice"
            if c.get("unrelated_after_RUN") or c.get("unrelated_O3"):
                f = root / "entries/unrelated"; f.mkdir()
                (f / "session.log").write_text("ARM O3 rc=0 try 1\n")
                time.sleep(.15)
                assert not (root / "notice").exists(), "unbound session fired notice"
            f = root / "entries/current"; f.mkdir()
            (f / "ENTRY.json").write_text(json.dumps(dict(entry="rw3", mode="gpu", env={"GA_GATE_ID": gid})))
            if c.get("old_ENTRY"):
                os.utime(f / "ENTRY.json", (1, 1))
            (f / "session.log").write_text("ARM O3 rc=0 try 1\n")
            until = time.monotonic() + 3
            while not (root / "notice").exists() and time.monotonic() < until:
                time.sleep(.01)
            assert (root / "notice").read_text() == str(f / "ENTRY.json")
        count = c.get("end_count", len(plan))
        for i, s in enumerate(plan[:count]):
            lane = plan[0]["lane"] if c.get("duplicate_steps") else s["lane"]
            rc = 7 if (c.get("one_rc7") or c.get("watch")) and i == len(plan)-1 else 0
            print(f"END {lane} rc={rc}", flush=True)
        if count < len(plan):
            print("SKIP " + plan[-1]["lane"] + " RAM", flush=True)
        if c.get("unexpected_step"):
            print("END foreign rc=0", flush=True)
        if c.get("foreign_log"):
            (root / "unrelated_newer.log").write_text("\n".join("END " + s["lane"] + " rc=0" for s in plan))
        if c.get("finish_past_end"):
            (root / "clock").write_text(str(c["end_epoch"] + 1))
        return c.get("seq_rc", 0)
    elif cmd == "registry":
        record(dict(mode="registry", smoke=os.environ.get("WRF_GPU2_REGISTRY_SMOKE")))
        # Schema double for an unavailable registry helper; '+' enum segments.
        forcing = args[0]
        return 0 if all(x in ("WN3", "WN2", "AIFS", "ERA5", "GFS", "IFS", "HARMONIE", "OTHER") for x in forcing.split("+")) else 2
    elif cmd == "launcher":
        record(dict(mode="launcher", args=args))
        if c.get("launch_rc"):
            return c["launch_rc"]
        if os.environ.get("WRF_GPU2_SEQ_SMOKE") and not c.get("omit_receipt"):
            receipt = dict(device="cpu", real_lock_used=False, completed_phases=c.get("phases", ["model", "e41", "check", "registry_end"]), registry_schema_valid=True, production_registry_writes=0)
            receipt.update(c.get("receipt_override", {}))
            Path(os.environ["WRF_GPU2_SMOKE_RECEIPT"]).write_text(json.dumps(receipt))
        if c.get("finish_past_end"):
            (root / "clock").write_text(str(c["end_epoch"] + 1))
    else:
        raise AssertionError(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
