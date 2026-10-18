"""S1AB summary (CPU, JSON only): matched PROD 24 h warm whole-run A (v0.4.0 bundle) vs B (v0.4.1 B1 pin).

usage: python summary.py OUT P_CHECK_BEFORE P_CHECK_AFTER       -> OUT/AB.json, last stdout line = compact verdict JSON
Per timed arm (A1 B1 A2 B2): E162 whole-run s/fc-h (last original wrfout mtime - receipt t0, / 24), the receipt anatomy of
b1/s1_attrib.py (pre-step, segment 1, median/mean seg 2-24, output, tail; s/fc-h = period * 3600/3618), the measured main
thread's run-queue wait (pidstat), the core gate state, the program identity (receipt git_head/src_tree must be the arm's
side), and for A the bundle consumer's BUNDLE_CONSUMER line. B/A = mean(B) / mean(A); A/A, B/B = |x1 - x2| / mean.
"""
import importlib.util, json, re, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s1_attrib", HERE.parent / "b1/s1_attrib.py")
attrib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attrib)
SIDES = {"A": ("7bb5ae669b2e4b863d6bafdd659a03fea026004f", "df562316"),
         "B": ("bcd156d19a0babb376f0f2976833e76f0f4c470a", "fb10553c9c67")}
ARMS = {"A1": "A", "B1": "B", "A2": "A", "B2": "B"}


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


out, pre, post = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
gates = {}
for line in (out / "core_gate.tsv").read_text().splitlines() if (out / "core_gate.tsv").exists() else []:
    name, utc, state, tries = line.split("\t")
    gates[name] = {"utc": utc, "state": state, "checks": int(tries)}
session = (out / "session.log").read_text() if (out / "session.log").exists() else ""
consumer = {m.group(1): m.group(2) for m in re.finditer(r"arm (\S+) consumer: (BUNDLE_CONSUMER .*)", session)}
result = {"out": str(out), "P_check_before": pre, "P_check_after": post, "consumer": consumer, "arms": {},
          "session_arms": re.findall(r"arm \S+ rc=\d+|SKIP \S+", session)}
for name, side in ARMS.items():
    receipt, end = load(out / name / "run/receipt.json"), load(out / name / "endpoint.json")
    if not receipt or not end:
        result["arms"][name] = "MISSING"
        continue
    a = attrib.anatomy(receipt)
    pid = attrib.pidstat_main(out / name / "pidstat.log", float(receipt["t0_wall_utc"]), receipt["pid"]) \
        if (out / name / "pidstat.log").exists() else []
    head, tree = SIDES[side]
    result["arms"][name] = {
        "side": side, "git_head": receipt.get("git_head"), "src_tree": receipt.get("src_tree"),
        "identity_ok": receipt.get("git_head") == head and str(receipt.get("src_tree", "")).startswith(tree),
        "end_reason": receipt.get("end_reason"), "endpoint_valid": end.get("valid"),
        "whole_s_per_fc_h": round(end.get("s_per_fc_h") or end["wall_s"] / end["hours"], 4) if end.get("wall_s") else None,
        "pre_step_s": a["pre_step_s"], "seg1_s": a["segments"][0]["period_s"], "median_seg2_24": a["median_seg2_24"],
        "mean_seg2_24": a["mean_seg2_24"], "output_s_total": a["output_s_total"], "tail_s": a["tail_after_last_output_s"],
        "t_end": a["t_end"], "main_wait_pct_max": max((s[2] for s in pid), default=None),
        "core_gate": gates.get(name), "segments_s_per_fc_h": [r["s_per_fc_h"] for r in a["segments"]]}


def pair(side, key):
    vals = [v[key] for v in result["arms"].values() if isinstance(v, dict) and v["side"] == side and v.get(key) is not None]
    return vals if len(vals) == 2 else None


summary = {}
for key in ("whole_s_per_fc_h", "median_seg2_24", "pre_step_s"):
    a, b = pair("A", key), pair("B", key)
    summary[key] = {"A": a, "B": b}
    if a and b:
        summary[key].update({"B_over_A": round(statistics.mean(b) / statistics.mean(a), 4),
                             "AA_pct": round(100 * abs(a[0] - a[1]) / statistics.mean(a), 2),
                             "BB_pct": round(100 * abs(b[0] - b[1]) / statistics.mean(b), 2)})
result["summary"] = summary
ok = [v for v in result["arms"].values() if isinstance(v, dict)]
result["valid"] = (len(ok) == 4 and all(v["identity_ok"] and v["endpoint_valid"] and v["end_reason"] == "completed" for v in ok)
                   and pre.startswith("BUNDLE_OK") and "new=0" in pre and post.startswith("BUNDLE_OK") and "new=0" in post)
(out / "AB.json").write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps({"valid": result["valid"], **{k: v for k, v in summary.items()},
                  "gates": {k: v["state"] for k, v in gates.items()}, "P": [pre[:40], post[:40]],
                  "consumer_A1": consumer.get("A1", "")[:120]}))
