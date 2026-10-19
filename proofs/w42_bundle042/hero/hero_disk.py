"""w42-bundle042 HERO: <DATA_ROOT> (nvme3n1) await of ONE arm window from the session-wide disk_watch.py JSONL.

usage: python3 hero_disk.py DISK_JSONL T0_UNIX T1_UNIX [LIMIT_MS=50]
Interval-weighted await = sum d(ms) / sum d(ios) over the 1 s records fully inside [T0, T1] (same formula as the w42-harness
S1H summarize.py disk_await and rw3t's post-arm check), plus seconds with a 1 s await > LIMIT_MS and the max 1 s await.
Prints one JSON line; "disturbed" = await_ms > LIMIT_MS (manager 2026-10-19T05:59:13Z: that arm's whole-run is DISTURBED,
not a regression). No records in the window -> await_ms null, disturbed null (unknown, never clean).
"""
import json
import sys


def arm_await(path, t0, t1, limit=50.0):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    win = [r for r in rows if t0 <= r["t"] - r["dt"] and r["t"] <= t1 + 1e-6]
    ios = sum(r["nvme3n1"]["ios"] for r in win)
    ms = sum(r["nvme3n1"]["ms"] for r in win)
    one = [r["nvme3n1"]["await_ms"] for r in win]
    aw = round(ms / ios, 2) if ios else (0.0 if win else None)
    return {"await_ms": aw, "seconds": len(win), "seconds_over_limit": sum(1 for a in one if a > limit),
            "max_1s_ms": max(one) if one else None, "limit_ms": limit,
            "disturbed": None if aw is None else aw > limit}


if __name__ == "__main__":
    lim = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
    print(json.dumps(arm_await(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), lim)))
