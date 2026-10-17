"""Bound an existing original-oracle cold setup at its original 60m deadline."""
from pathlib import Path
import argparse
import json
import os
import signal
import time


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19])  # ppid, starttime ticks


def main(parent, oracle, timer, receipt, resume=None):
    if resume:
        previous = json.loads(resume.read_text())
        timer_id, parent_id, oracle_id = previous["timer_identity"], previous["parent_identity"], previous["oracle_identity"]
        assert identity(oracle)[1] == oracle_id[1]
        deadline = previous["deadline_monotonic"]
    else:
        timer_id, parent_id, oracle_id = identity(timer), identity(parent), identity(oracle)
        assert parent_id[0] == timer and oracle_id[0] == parent
        deadline = timer_id[1] / os.sysconf("SC_CLK_TCK") + 3600.
    # /proc task starttimes and /proc/uptime share the kernel time origin.
    # Python's namespaced CLOCK_MONOTONIC can use a different offset here.
    clock = lambda: float(Path("/proc/uptime").read_text().split()[0])
    data = {"watchdog_pid": os.getpid(), "timer": timer, "timer_identity": timer_id, "parent": parent, "parent_identity": parent_id,
            "initial_oracle": oracle, "oracle_identity": oracle_id, "deadline_monotonic": deadline,
            "remaining_s": deadline - clock(), "status": "armed"}
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data), flush=True)
    while clock() < deadline:
        alive = False
        for pid, ident in ((parent, parent_id), (oracle, oracle_id)):
            try:
                alive |= identity(pid)[1] == ident[1]
            except FileNotFoundError:
                pass
        if not alive:
            break
        time.sleep(1.)
    else:
        victims = {}
        for pid, expected in ((parent, parent_id), (oracle, oracle_id)):
            try:
                if identity(pid)[1] == expected[1]:
                    victims[pid] = identity(pid)
                    children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                    victims.update({int(child): identity(int(child)) for child in children})
            except FileNotFoundError:
                pass
        for pid, ident in victims.items():
            try:
                if identity(pid) == ident:
                    os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(2.)
        for pid, ident in victims.items():
            try:
                if identity(pid) == ident:
                    os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
        data["status"] = "deadline_terminated"
        data["victims"] = victims
        receipt.write_text(json.dumps(data, indent=2) + "\n")
        return
    data["status"] = "parent_completed"
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=int)
    parser.add_argument("--oracle", required=True, type=int)
    parser.add_argument("--timer", required=True, type=int)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    main(args.parent, args.oracle, args.timer, args.receipt, args.resume)
