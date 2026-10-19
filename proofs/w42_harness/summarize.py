"""w42-harness S1H: CPU summary of one session root (after session.sh).

usage: python3 summarize.py SESSION_ROOT OUT_DIR
Per accepted PROD arm (runs/<slot>, aborted tries are listed, never counted): receipt segment rates (bench_summary.arm_summary,
same tools dir), whole-run wall to the last original wrfout mtime (E162) from three bases -- session launch (before the
consumer: common basis of O and N), harness launch (N only: receipt launch_wall_utc), measured process t0 (receipt t0;
the basis of every earlier harness number) -- the measured pool (N: receipt allocator; O: the CLI's 'gpuwrf: C-auto ...'
line), JAX peak / bytes_limit, consumer compiled_new, watcher disclosures (core_watch sweep + 20/50 shadow, sched_watch CLI
runqueue wait + cgroup PSI, <DATA_ROOT> await), wrfout SHA-256. Ratios N/O per kind with A/A (O1 vs O2) and B/B (N1 vs N2)
spreads = |x1 - x2| / mean. WN3 par3: batch wall to the last wrfout of the three cases / 72 case-hours vs CPU-WRF 3x4-core
123.385 s per case-hour (B1/D28 denominator). Writes OUT_DIR/S1H_RESULT.json + S1H_TABLE.md; bench_summary headline JSONs
bench_new_proxy.json (N proxies, C-auto required) and bench_old_proxy.json (O proxies, --allow-demand-alloc).
"""
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench_summary", HERE / "bench_summary.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)
CPU_WRF_CASE_H = 123.385
B6_SHA = Path("<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.1/receipts/B6_prod_3h/wrfout_sha.txt")


def sha_list(files):
    out = {}
    for p in sorted(files):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out[p.name] = h.hexdigest()
    return out


def disk_await(t0, t1):
    try:
        rows = [json.loads(x) for x in (A / "disk.jsonl").read_text().splitlines() if x.strip()]
    except OSError:
        return None
    win = [r for r in rows if t0 <= r["t"] - r["dt"] and r["t"] <= t1 + 1e-6]
    ios = sum(r["nvme3n1"]["ios"] for r in win)
    ms = sum(r["nvme3n1"]["ms"] for r in win)
    return {"await_ms": round(ms / ios, 2) if ios else 0.0, "seconds": len(win)}


def watch(d):
    def rd(name):
        try:
            return json.loads((d / name).read_text())
        except (OSError, ValueError):
            return None
    cw, st, sw = rd("core_watch_summary.json"), rd("core_watch_strict_summary.json"), rd("sched_watch_summary.json")
    pick = lambda s: None if s is None else {k: s.get(k) for k in ("clean", "within_rule", "max_proc_seconds", "agg_seconds")}  # noqa: E731
    return {"sweep_rule": pick(cw), "strict_20_50_shadow": pick(st),
            "cli_main_wait_share": (sw or {}).get("cli_main_wait_share"),
            "cgroup_psi_some_s": ((sw or {}).get("last") or {}).get("pressure_some_total_s")}


def prod_arm(d):
    r = json.loads((d / "run" / "receipt.json").read_text())
    ep = json.loads((d / "endpoint.json").read_text())
    s = bs.arm_summary(d / "run" / "receipt.json")
    hours = int(r["cli_argv"][r["cli_argv"].index("--hours") + 1])
    segs = [x["s_per_fc_h"] for x in s["segments"] if x["k"] >= 2]
    last = max(ep["file_write_unix"].values())
    err = (d / "stderr.log").read_text(errors="replace")
    cli_pool = re.findall(r"gpuwrf: (C-auto[^\n]{0,160})", err)
    cons = [l for l in (d / "stdout.log").read_text(errors="replace").splitlines() if l.startswith("BUNDLE_CONSUMER ")]
    wrf = sorted(Path(r["cli_argv"][r["cli_argv"].index("--output-dir") + 1]).glob("wrfout_d0*"))
    t1 = float((d / "t_end_unix").read_text())
    alloc = r.get("allocator") or {}
    return {
        "slot": d.name, "hours": hours, "rc": r.get("rc"), "valid": ep["valid"], "src_tree": r.get("src_tree"),
        "git_head": r.get("git_head"), "device": r.get("device"),
        "allocator_mode": alloc.get("mode") or ("demand" if any("unavailable" in c for c in cli_pool) else
                                                 "c_auto" if any("case pool" in c for c in cli_pool) else "unknown"),
        "allocator": alloc or None, "cli_pool_line": cli_pool[:1], "pre_exec": r.get("pre_exec"),
        "seg_rates_s_per_fc_h": segs,
        "proxy_seg2_3": round(statistics.mean(segs[:2]), 3) if len(segs) >= 2 else None,
        "seg2plus_mean": s["s_per_fc_h_seg2plus"], "seg2plus_median": round(statistics.median(segs), 3) if segs else None,
        "stepping_s_per_fc_h": s["stepping_s_per_fc_h"], "output_s_per_fc_h": s["output_s_per_fc_h"],
        "whole_s_per_fc_h": {
            "session_launch": round((last - ep["session_launch_unix"]) / hours, 3),
            "harness_launch": (round((last - float(ep["harness_launch_wall_utc"])) / hours, 3)
                               if ep.get("harness_launch_wall_utc") else None),
            "measured_process_t0": round((last - ep["t0_wall_utc"]) / hours, 3)},
        "first_segment_s": s["segments"][0]["s_per_fc_h"] if s["segments"] else None,
        "jax_memory_stats": r.get("jax_memory_stats"), "vram_peak_mib_pid": (r.get("derived") or {}).get("vram_peak_mib"),
        "consumer": cons[-1] if cons else None, "watch": watch(d),
        "disk": disk_await(float((d / "launch_unix").read_text()), t1),
        "wrfout_sha256": sha_list(wrf), "n_wrfout": len(wrf),
    }


def spread(a, b):
    return round(abs(a - b) / ((a + b) / 2) * 100, 2)


runs = A / "runs"
arms, aborted = {}, sorted(p.name for p in runs.glob("*.aborted*"))
for d in sorted(runs.iterdir()):
    if d.is_dir() and ".aborted" not in d.name and (d / "run" / "receipt.json").is_file():
        arms[d.name] = prod_arm(d)

ratios = {}
for kind, metric in (("S", ("whole_s_per_fc_h", "session_launch")), ("S", ("whole_s_per_fc_h", "measured_process_t0")),
                     ("S", ("seg2plus_median", None)), ("P", ("proxy_seg2_3", None)), ("P", ("whole_s_per_fc_h", "session_launch"))):
    def val(name):
        a = arms.get(name)
        if not a:
            return None
        v = a[metric[0]]
        return v[metric[1]] if metric[1] else v
    o = [v for v in (val(f"{kind}_O1"), val(f"{kind}_O2")) if v is not None]
    n = [v for v in (val(f"{kind}_N1"), val(f"{kind}_N2")) if v is not None]
    key = f"{kind}:{metric[0]}" + (f":{metric[1]}" if metric[1] else "")
    ratios[key] = {"O": o, "N": n,
                   "N_over_O": round(statistics.mean(n) / statistics.mean(o), 4) if o and n else None,
                   "AA_pct": spread(*o) if len(o) == 2 else None, "BB_pct": spread(*n) if len(n) == 2 else None}

ident = {}
for hours in (3, 24):
    shas = {k: a["wrfout_sha256"] for k, a in arms.items() if a["hours"] == hours}
    sets = {json.dumps(v, sort_keys=True) for v in shas.values()}
    ident[f"{hours}h_all_arms_byte_identical"] = len(sets) == 1 if shas else None
if B6_SHA.is_file():
    b6 = dict(reversed(l.split()) for l in B6_SHA.read_text().splitlines() if l.strip())
    ident["3h_equal_bundle_B6_prod_cli"] = {k: a["wrfout_sha256"] == b6 for k, a in arms.items() if a["hours"] == 3}

wn3 = None
w = runs / "WN3"
if (w / "mtimes.json").is_file():
    mt = json.loads((w / "mtimes.json").read_text())
    launch = float((w / "launch_unix").read_text())
    per = {c: {"n_wrfout": len(v), "wall_s": round(max(v.values()) - launch, 2) if v else None} for c, v in mt.items()}
    batch = max(x["wall_s"] for x in per.values() if x["wall_s"] is not None)
    case_h = 24 * len(per)
    wn3 = {"cases": per, "batch_wall_s": batch, "s_per_case_sim_hour": round(batch / case_h, 3),
           "speedup_vs_cpu_wrf_3x4core": round(CPU_WRF_CASE_H / (batch / case_h), 2),
           "consumer": next((l for l in (w / "stdout.log").read_text(errors="replace").splitlines()
                             if l.startswith("BUNDLE_CONSUMER ")), None), "watch": watch(w),
           "disk": disk_await(launch, float((w / "t_end_unix").read_text()))}

benches = {}
for label, names, extra in (("new_proxy", ["P_N1", "P_N2"], []), ("old_proxy", ["P_O1", "P_O2"], ["--allow-demand-alloc"])):
    got = [n for n in names if n in arms]
    if len(got) == 2:
        rc = bs.main(["--out", str(OUT / f"bench_{label}.json"), "--tag", f"w42h_{label}", *extra,
                      *[f"warm{i + 1}={runs / n / 'run' / 'receipt.json'}" for i, n in enumerate(got)]])
        benches[label] = {"rc": rc, "json": str(OUT / f"bench_{label}.json")}

res = {"schema": "w42-harness.s1h.v1", "session_root": str(A), "arms": arms, "aborted_tries": aborted, "ratios": ratios,
       "identity": ident, "wn3_par3": wn3, "bench_summary": benches,
       "core_gate": (A / "core_gate.tsv").read_text().splitlines() if (A / "core_gate.tsv").is_file() else []}
(OUT / "S1H_RESULT.json").write_text(json.dumps(res, indent=1) + "\n")
lines = ["| slot | pool | proxy seg2-3 | seg2+ median | whole (session launch) | whole (t0) | first seg | JAX peak GB | "
         "compiled_new | sweep watch | CLI wait | PSI s |", "|---" * 12 + "|"]
for k, a in arms.items():
    cn = re.search(r"compiled_new=(\d+)", a["consumer"] or "")
    lines.append(f"| {k} | {a['allocator_mode']} | {a['proxy_seg2_3']} | {a['seg2plus_median']} | "
                 f"{a['whole_s_per_fc_h']['session_launch']} | {a['whole_s_per_fc_h']['measured_process_t0']} | "
                 f"{a['first_segment_s']} | {round(((a['jax_memory_stats'] or {}).get('peak_bytes_in_use') or 0) / 1e9, 2)} | "
                 f"{cn.group(1) if cn else '?'} | {(a['watch']['sweep_rule'] or {}).get('within_rule')} | "
                 f"{a['watch']['cli_main_wait_share']} | {a['watch']['cgroup_psi_some_s']} |")
lines += ["", "| ratio | O | N | N/O | A/A % | B/B % |", "|---|---|---|---|---|---|"]
lines += [f"| {k} | {v['O']} | {v['N']} | {v['N_over_O']} | {v['AA_pct']} | {v['BB_pct']} |" for k, v in ratios.items()]
lines += ["", f"identity: {json.dumps(ident)}", f"WN3 par3: {json.dumps(wn3)}", f"aborted tries: {aborted}"]
(OUT / "S1H_TABLE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
