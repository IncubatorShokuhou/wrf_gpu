"""B1 PROD whole-run S1 attribution, CPU only (manager 2026-10-18T11:47:21Z: +16.6 % vs RC2 -> explain, program vs host).

usage: python s1_attrib.py OURS_RECEIPT RC2_RECEIPT OURS_PIDSTAT HOST_LOAD OUT_JSON
OURS_RECEIPT = B1 S1/whole24/run/receipt.json (pin bcd156d19a); RC2_RECEIPT = docs/release/evidence/v031/prod24_receipt_selected.json
(RC2 addb342c5, the 3.604 reference); OURS_PIDSTAT = the arm's pidstat -u -t of the measured process (5 s samples, local time);
HOST_LOAD = session HOST_LOAD.txt (top-25 snapshots before/after every arm). Per receipt: startup (jax import, wrappers, first
step start), each 1 h segment (period, call, compile in call, output s, advance host_s), tail (t_end - last output end);
s/fc-h = period * 3600 / 3618 like bench_summary. Host side: per segment the measured main thread's mean %CPU / %wait (time
runnable but not running = CPU contention) / cores seen, plus the foreign processes the snapshots saw on 3,13,19,29.
"""
import json, re, statistics, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCALE = 3600.0 / 3618.0
GPU_ARM = {3, 13, 19, 29}


def anatomy(receipt):
    segs, outs, adv = receipt["derived"]["segments"], receipt["outputs"], receipt["advance"]
    rows = []
    for i, seg in enumerate(segs):
        lo = seg["t_start"]
        hi = segs[i + 1]["t_start"] if i + 1 < len(segs) else receipt["t_end"]
        period = seg.get("period_to_next_s") or (hi - lo)
        rows.append({"k": seg["k"], "t_start": round(lo, 3), "period_s": round(period, 4),
                     "s_per_fc_h": round(period * SCALE, 4), "call_s": seg.get("call_s"),
                     "compile_s_in_call": seg.get("compile_s_in_call"),
                     "output_s": round(sum(o["s"] for o in outs if lo <= o["t_start"] < hi), 4),
                     "advance_host_s": round(sum(a.get("host_s") or 0 for a in adv if lo <= a["t"] < hi), 4)})
    last_out = max(o["t_start"] + o["s"] for o in outs)
    return {"t_jax_import": receipt.get("t_jax_import"), "t_wrappers": receipt.get("t_wrappers"),
            "pre_step_s": round(segs[0]["t_start"], 3), "first_advance_t": round(min(a["t"] for a in adv), 3),
            "first_output_end_t": round(min(o["t_start"] + o["s"] for o in outs), 3), "last_output_end_t": round(last_out, 3),
            "t_end": receipt["t_end"], "tail_after_last_output_s": round(receipt["t_end"] - last_out, 3),
            "segments": rows, "median_seg2_24": round(statistics.median(r["s_per_fc_h"] for r in rows[1:]), 4),
            "mean_seg2_24": round(statistics.mean(r["s_per_fc_h"] for r in rows[1:]), 4),
            "output_s_total": round(sum(o["s"] for o in outs), 3), "src_tree": receipt.get("src_tree"),
            "git_head": receipt.get("git_head")}


def pidstat_main(path, t0, tgid):
    """Main-thread samples of the measured process: (t relative to t0, %CPU, %wait, core). pidstat prints local time."""
    lines = Path(path).read_text().splitlines()
    day = datetime.strptime(re.search(r"(\d\d/\d\d/\d\d)", lines[0]).group(1), "%m/%d/%y").date()
    offset = datetime.now().astimezone().utcoffset() or timedelta(0)
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) < 10 or not re.fullmatch(r"\d\d:\d\d:\d\d", parts[0]) or parts[3] != str(tgid):
            continue  # main thread rows: TGID '-' and TID == measured pid
        local = datetime.combine(day, datetime.strptime(parts[0], "%H:%M:%S").time())
        utc = (local - offset).replace(tzinfo=timezone.utc).timestamp()
        out.append((utc - t0, float(parts[8]), float(parts[7]), int(parts[9])))
    return out


def host_snapshots(path, arm):
    """Foreign processes in the HOST_LOAD snapshots around ARM that sat on a GPU-arm core (psr in 3,13,19,29)."""
    text, keep, rows = Path(path).read_text().splitlines(), False, {}
    for line in text:
        if line.startswith("== "):
            keep = arm in line
            label = line[3:].split(" loadavg")[0]
            if keep:
                rows[label] = {"loadavg": line.split("loadavg ")[1], "on_gpu_arm_cores": []}
            continue
        parts = line.split(None, 4)
        if keep and len(parts) == 5 and parts[0].isdigit() and int(parts[0]) in GPU_ARM and float(parts[1]) > 5:
            rows[label]["on_gpu_arm_cores"].append(" ".join(parts)[:140])
    return rows


def main(ours_path, rc2_path, pidstat_path, host_path, out_path):
    ours, rc2 = json.loads(Path(ours_path).read_text()), json.loads(Path(rc2_path).read_text())
    a, b = anatomy(ours), anatomy(rc2)
    samples = pidstat_main(pidstat_path, float(ours["t0_wall_utc"]), ours["pid"])
    for row in a["segments"]:
        lo, hi = row["t_start"], row["t_start"] + row["period_s"]
        inside = [s for s in samples if lo <= s[0] < hi]
        row["main_thread"] = {"n": len(inside), "cpu_pct": round(statistics.mean(s[1] for s in inside), 1) if inside else None,
                              "wait_pct": round(statistics.mean(s[2] for s in inside), 1) if inside else None,
                              "cores": sorted({s[3] for s in inside})}
    pairs = [{"k": x["k"], "ours": x["s_per_fc_h"], "rc2": y["s_per_fc_h"], "ratio": round(x["s_per_fc_h"] / y["s_per_fc_h"], 4),
              "ours_wait_pct": x["main_thread"]["wait_pct"], "ours_output_s": x["output_s"], "rc2_output_s": y["output_s"]}
             for x, y in zip(a["segments"], b["segments"])]
    ratios = [p["ratio"] for p in pairs[1:]]
    whole = lambda r: r["t_end"] / 24.0  # noqa: E731  (receipt t_end; the E162 last-file endpoint is separate)
    result = {"ours": a, "rc2": b, "per_segment": pairs,
              "decomposition_s": {"pre_step": [a["pre_step_s"], b["pre_step_s"]],
                                  "seg1": [a["segments"][0]["period_s"], b["segments"][0]["period_s"]],
                                  "seg2_24_sum": [round(sum(r["period_s"] for r in a["segments"][1:]), 3),
                                                  round(sum(r["period_s"] for r in b["segments"][1:]), 3)],
                                  "t_end": [a["t_end"], b["t_end"]], "t_end_per_fc_h": [round(whole(a), 4), round(whole(b), 4)]},
              "seg2_24_ratio": {"median": round(statistics.median(ratios), 4), "min": min(ratios), "max": max(ratios),
                                "stdev": round(statistics.pstdev(ratios), 4)},
              "main_thread_wait_pct": {"max": max((s[2] for s in samples), default=None),
                                       "mean": round(statistics.mean(s[2] for s in samples), 2) if samples else None,
                                       "n_samples": len(samples), "cores": sorted({s[3] for s in samples})},
              "host_snapshots": host_snapshots(host_path, "S1/whole24")}
    Path(out_path).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("decomposition_s", "seg2_24_ratio", "main_thread_wait_pct")}, indent=1))
    for p in pairs:
        print(p)


if __name__ == "__main__":
    main(*sys.argv[1:6])
