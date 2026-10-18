"""Post-hoc acceptance of ATTRIB2 slots (critic cr-w41-releaseev 14:32:58Z / 16:18:26Z rule; the session rc is NOT authority).

usage: python3 accept.py ATTRIB_DIR OUT_JSON [DISK_SUMMARY_JSON]
A slot (N1 O1 N2 O2 N3 O3) COUNTS only if its accepted dir runs/<slot> passes every check:
  rc0            rc.txt == 0 and device.txt names a CUDA device
  consumer       scope log BUNDLE_CONSUMER rc=0 compiled_new=0; check_before/check_after BUNDLE_OK missing=0 changed=0 new=0
  watch_present  core_watch_summary.json and core_watch.jsonl exist; core_watch.log has no Traceback
  watch_final    final summary within_rule == true, aborted == null, and NO ABORT file in the dir (late breach)
  watch_cover    jsonl samples cover t0..t_end: first sample <= t0 + 3 s, last >= t_end - 3 s, no gap > 3 s,
                 samples >= (t_end - t0) - 5; summary utc_end >= t_end (written after the arm)
Retained rejected attempts runs/<slot>.abortedK are listed with their ABORT reason. Disk disposition (await > 50 ms)
is attached from disk_watch.py summarize when given (disclosure, not a rejection).
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
disk = json.loads(Path(sys.argv[3]).read_text())["arms"] if len(sys.argv) > 3 else {}
R = A / "runs"


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def check(slot):
    d = R / slot
    c, info = {}, {}
    if not d.is_dir():
        return {"present": False}, {}
    rd = lambda n: (d / n).read_text() if (d / n).exists() else ""
    c["rc0"] = rd("rc.txt").strip() == "0" and bool(re.search(r"cuda|gpu", rd("device.txt"), re.I))
    scope = (R / f"{slot}.scope.log").read_text() if (R / f"{slot}.scope.log").exists() else ""
    cons = [l for l in scope.splitlines() if l.startswith("BUNDLE_CONSUMER rc=")]
    ok = lambda n: rd(n).splitlines()[-1:] and rd(n).splitlines()[-1].startswith("BUNDLE_OK missing=0 changed=0 new=0 ")
    c["consumer"] = bool(cons) and " rc=0 " in cons[-1] + " " and "compiled_new=0" in cons[-1] and bool(ok("check_before.txt")) and bool(ok("check_after.txt"))
    info["consumer_line"] = cons[-1][:160] if cons else None
    summ = json.loads(rd("core_watch_summary.json")) if rd("core_watch_summary.json") else None
    lines = [json.loads(x) for x in rd("core_watch.jsonl").splitlines() if x.strip()]
    c["watch_present"] = summ is not None and bool(lines) and "Traceback" not in rd("core_watch.log")
    c["watch_final"] = bool(summ) and summ.get("within_rule") is True and summ.get("aborted") is None and not rd("ABORT").strip()
    try:
        t0, t1 = float(rd("t0_unix")), float(rd("t_end_unix"))
    except ValueError:
        t0 = t1 = None
    if lines and t0:
        t = [ts(x["utc"]) for x in lines]
        gaps = [b - a for a, b in zip(t, t[1:])]
        info.update(first_minus_t0=round(t[0] - t0, 1), t_end_minus_last=round(t1 - t[-1], 1), max_gap_s=max(gaps, default=0),
                    samples=len(lines), wall_s=round(t1 - t0, 1),
                    summary_end_minus_t_end=round(ts(summ["utc_end"]) - t1, 1) if summ else None)
        c["watch_cover"] = (t[0] <= t0 + 3 and t[-1] >= t1 - 3 and max(gaps, default=0) <= 3 and len(lines) >= (t1 - t0) - 5
                            and summ is not None and ts(summ["utc_end"]) >= t1)
    else:
        c["watch_cover"] = False
    if summ:
        info["watch"] = {k: summ.get(k) for k in ("clean", "within_rule", "max_proc_seconds", "agg_seconds", "agg_max_pct",
                                                  "kthread_seconds_over_proc_pct", "kthread_max_pct", "core_busy_pct")}
        info["watch_offenders"] = summ.get("offenders", [])[:3]
    return c, info


res = {"rule": "critic 14:32:58Z: rc0 + consumer/bundle OK + watcher present/healthy + continuous coverage + final within_rule, no ABORT",
       "slots": {}, "rejected_attempts": {}}
for slot in ("N1", "O1", "N2", "O2", "N3", "O3"):
    c, info = check(slot)
    res["slots"][slot] = dict(counts=bool(c) and all(c.values()) and c.get("present", True) is not False, checks=c, info=info,
                              disk=disk.get(slot))
for d in sorted(R.glob("*.aborted*")):
    if d.is_dir():
        res["rejected_attempts"][d.name] = dict(abort=(d / "ABORT").read_text().strip() if (d / "ABORT").exists() else None,
                                                disk=disk.get(d.name))
res["counted"] = [s for s, v in res["slots"].items() if v["counts"]]
OUT.write_text(json.dumps(res, indent=1) + "\n")
for s, v in res["slots"].items():
    print(s, "COUNTS" if v["counts"] else "REJECT", {k: x for k, x in v["checks"].items() if not x} or "", (v.get("disk") or {}).get("disposition"),
          (v["info"].get("watch") or {}).get("clean"), v["info"].get("max_gap_s"), v["info"].get("samples"), v["info"].get("wall_s"))
print("rejected attempts:", {k: (v["abort"] or "")[:120] for k, v in res["rejected_attempts"].items()})
