"""w42-bundle042 HERO: CPU summary of one hero session root (after hero_session.sh) -> HERO_RESULT.json + HERO_TABLE.md.

usage: python3 hero_summary.py SESSION_ROOT OUT_DIR [B6_SHA_TXT]
Derived from the w42-harness S1H summarize.py (pin 8c92c3c5d7:proofs/w42_harness/summarize.py). Per accepted PROD arm
(runs/<slot>; aborted tries are listed, never counted): receipt segment rates (bench_summary.arm_summary of the PIN,
P/src/scripts/bench_summary.py), whole-run wall to the last ORIGINAL wrfout mtime (E162) from three bases -- harness launch
(receipt launch_wall_utc = start of the measured harness process: PRIMARY, the basis of the released 4.600), session launch
(before the consumer) and measured process t0 --, the pool attestation (receipt allocator.mode == c_auto, the NVML free-memory
probe backend-dark, JAX bytes_limit == the C-auto budget within 0.5 %), JAX peak, consumer compiled_new, watcher disclosures,
arm-window <DATA_ROOT> await (hero_disk.py; > 50 ms = DISTURBED), per-PID VRAM peak, host VmHWM, wrfout SHA-256.
Pre-registered hero rule (hero_session.sh header):
  HERO_PROD_S_PER_FCH = mean primary whole-run s/fc-h over the CLEAN PROD 24 h arms (H1/H2/H3: accepted, rc 0, endpoint
  valid, C-auto attested, compiled_new 0, await <= 50 ms); none clean -> mean over the accepted attested 24 h arms, labelled
  DISTURBED. HERO_PROD_SECONDS = the same mean in seconds. Ratio = CPU-WRF 12-rank 92.3 s/fc-h / HERO_PROD_S_PER_FCH.
  HERO_PAR3_S_PER_CASE_H = WN3 batch wall (session launch -> last wrfout of the 3 cases) / 72 case-hours; ratio = CPU-WRF
  3x4-core 123.385 s per case-hour / it; valid iff the 3 cases started together (start events <= 5 s, peak concurrency 3),
  rc 0 x 3, 75 wrfout each, C-auto plan + pool line in every case receipt, AOT-warm each, consumer compiled_new=0;
  DISTURBED iff its await > 50 ms (disclosed, the number is still reported).
Identity (disclosure): all PROD 24 h arms byte-identical; W0 (3 h harness) == the bundle session's B6 PROD 3 h CLI replay.
"""
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
B6_SHA = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("<DATA_ROOT>/wrf_gpu2/scratch/w42-bundle042/gpu/runs/B6_prod_3h/wrfout_sha.txt")
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hero_disk import arm_await  # noqa: E402

spec = importlib.util.spec_from_file_location("bench_summary", "<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.2/src/scripts/bench_summary.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)
CPU_WRF_PROD = 92.3
CPU_WRF_CASE_H = 123.385
LIMIT_MS = 50.0


def sha_list(files):
    out = {}
    for p in sorted(files):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        out[p.name] = h.hexdigest()
    return out


def rd(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def disk(d):
    got = rd(d / "disk_await.json")
    if got is not None:
        return got
    try:
        return arm_await(A / "disk.jsonl", float((d / "launch_unix").read_text()), float((d / "t_end_unix").read_text()), LIMIT_MS)
    except (OSError, ValueError) as exc:
        return {"await_ms": None, "disturbed": None, "error": repr(exc)}


def watch(d):
    cw, st, sw = rd(d / "core_watch_summary.json"), rd(d / "core_watch_strict_summary.json"), rd(d / "sched_watch_summary.json")
    pick = lambda s: None if s is None else {k: s.get(k) for k in ("clean", "within_rule", "max_proc_seconds", "agg_seconds")}  # noqa: E731
    return {"sweep_rule": pick(cw), "strict_20_50_shadow": pick(st),
            "cli_main_wait_share": (sw or {}).get("cli_main_wait_share"),
            "cgroup_psi_some_s": ((sw or {}).get("last") or {}).get("pressure_some_total_s")}


def consumer(d):
    cons = [l for l in (d / "stdout.log").read_text(errors="replace").splitlines() if l.startswith("BUNDLE_CONSUMER ")]
    line = cons[-1] if cons else None
    m = re.search(r"compiled_new=(\d+)", line or "")
    return line, (int(m.group(1)) if m else None)


def vram_peak(d, pid):
    """Max per-PID used_memory (MiB) of the measured PID from vram_apps.csv (nvidia-smi 0.5 s), else None."""
    try:
        rows = [l.split(",") for l in (d / "vram_apps.csv").read_text().splitlines() if l.count(",") >= 2]
        vals = [int(r[2]) for r in rows if pid is not None and r[1].strip() == str(pid) and r[2].strip().isdigit()]
        return max(vals) if vals else None
    except OSError:
        return None


def prod_arm(d):
    r = json.loads((d / "run" / "receipt.json").read_text())
    ep = json.loads((d / "endpoint.json").read_text())
    s = bs.arm_summary(d / "run" / "receipt.json")
    hours = int(r["cli_argv"][r["cli_argv"].index("--hours") + 1])
    segs = [x["s_per_fc_h"] for x in s["segments"] if x["k"] >= 2]
    steps = [x["stepping_s_per_fc_h"] for x in s["segments"] if x["k"] >= 2]
    last = max(ep["file_write_unix"].values())
    err = (d / "stderr.log").read_text(errors="replace")
    cli_pool = re.findall(r"gpuwrf: (C-auto[^\n]{0,160})", err)
    line, compiled_new = consumer(d)
    wrf = sorted(Path(r["cli_argv"][r["cli_argv"].index("--output-dir") + 1]).glob("wrfout_d0*"))
    alloc = r.get("allocator") or {}
    jms = r.get("jax_memory_stats") or {}
    budget, limit = alloc.get("c_auto_budget_bytes"), jms.get("bytes_limit")
    probe = ((r.get("pre_exec") or {}).get("nvml_probe") or {})
    attested = bool(alloc.get("mode") == "c_auto" and probe.get("backend_dark") is True and budget and limit
                    and abs(int(limit) - int(budget)) <= 0.005 * int(budget))
    hl = float(ep["harness_launch_wall_utc"]) if ep.get("harness_launch_wall_utc") else None
    dk = disk(d)
    return {
        "slot": d.name, "hours": hours, "rc": r.get("rc"), "end_reason": r.get("end_reason"), "valid": ep["valid"],
        "src_tree": r.get("src_tree"), "git_head": r.get("git_head"), "device": r.get("device"),
        "c_auto_attested": attested, "allocator": alloc or None, "cli_pool_line": cli_pool[:1],
        "nvml_probe_backend_dark": probe.get("backend_dark"), "jax_bytes_limit": limit, "c_auto_budget_bytes": budget,
        "seg_rates_s_per_fc_h": segs, "seg2plus_median": round(statistics.median(segs), 3) if segs else None,
        "seg2plus_mean": s["s_per_fc_h_seg2plus"], "stepping_seg2plus_median": round(statistics.median(steps), 3) if steps else None,
        "stepping_s_per_fc_h": s["stepping_s_per_fc_h"], "output_s_per_fc_h": s["output_s_per_fc_h"],
        "first_segment_s_per_fc_h": s["segments"][0]["s_per_fc_h"] if s["segments"] else None,
        "whole_s": {"harness_launch": round(last - hl, 3) if hl else None,
                    "session_launch": round(last - ep["session_launch_unix"], 3),
                    "measured_process_t0": round(last - ep["t0_wall_utc"], 3)},
        "whole_s_per_fc_h": {"harness_launch": round((last - hl) / hours, 3) if hl else None,
                             "session_launch": round((last - ep["session_launch_unix"]) / hours, 3),
                             "measured_process_t0": round((last - ep["t0_wall_utc"]) / hours, 3)},
        "jax_peak_bytes": jms.get("peak_bytes_in_use"), "vram_peak_mib_receipt": (r.get("derived") or {}).get("vram_peak_mib"),
        "vram_peak_mib_pid_csv": vram_peak(d, r.get("pid") or (r.get("pre_exec") or {}).get("pid")),
        "host_vmhwm_kb": r.get("host_vmhwm_kb"), "consumer": line, "compiled_new": compiled_new, "watch": watch(d),
        "disk": dk, "disturbed": dk.get("disturbed"), "wrfout_sha256": sha_list(wrf), "n_wrfout": len(wrf),
    }


def spread(a, b):
    return round(abs(a - b) / ((a + b) / 2) * 100, 2)


runs = A / "runs"
arms, aborted, errors = {}, sorted(p.name for p in runs.glob("*.aborted*")), {}
for d in sorted(runs.iterdir()) if runs.is_dir() else []:
    if d.is_dir() and ".aborted" not in d.name and d.name != "WN3":
        try:
            arms[d.name] = prod_arm(d)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors[d.name] = repr(exc)[:300]

timed = [a for k, a in arms.items() if k in ("H1", "H2", "H3") and a["hours"] == 24 and a["rc"] == 0 and a["valid"]
         and a["c_auto_attested"] and a["compiled_new"] == 0]
clean = [a for a in timed if a["disturbed"] is False]
basis, label = (clean, "CLEAN") if clean else (timed, "DISTURBED")
hero_prod = None
if basis:
    mean = lambda key, sub: round(statistics.mean(a[key][sub] for a in basis), 3)  # noqa: E731
    v = mean("whole_s_per_fc_h", "harness_launch")
    hero_prod = {
        "HERO_PROD_S_PER_FCH": v, "HERO_PROD_SECONDS": mean("whole_s", "harness_launch"),
        "ratio_vs_cpu_wrf_92_3": round(CPU_WRF_PROD / v, 2), "label": label, "arms": [a["slot"] for a in basis],
        "basis": "process start (harness launch) -> last original wrfout mtime (E162), whole 24 h run incl. init, stepping, output",
        "session_clock_s_per_fch": mean("whole_s_per_fc_h", "session_launch"),
        "measured_process_t0_s_per_fch": mean("whole_s_per_fc_h", "measured_process_t0"),
        "stepping_seg2plus_median_s_per_fch": round(statistics.mean(a["stepping_seg2plus_median"] for a in basis), 3),
        "seg2plus_median_s_per_fch": round(statistics.mean(a["seg2plus_median"] for a in basis), 3),
        "all_timed_arms": {a["slot"]: {"whole_s_per_fc_h": a["whole_s_per_fc_h"], "await_ms": a["disk"].get("await_ms"),
                                       "disturbed": a["disturbed"]} for a in timed},
        "spread_pct_first_two": (spread(timed[0]["whole_s_per_fc_h"]["harness_launch"], timed[1]["whole_s_per_fc_h"]["harness_launch"])
                                 if len(timed) >= 2 else None),
    }

ident = {}
shas = {k: a["wrfout_sha256"] for k, a in arms.items() if a["hours"] == 24}
ident["24h_all_arms_byte_identical"] = (len({json.dumps(v, sort_keys=True) for v in shas.values()}) == 1) if shas else None
if B6_SHA.is_file() and "W0" in arms:
    b6 = dict(reversed(l.split()) for l in B6_SHA.read_text().splitlines() if l.strip())
    ident["W0_3h_equal_bundle_B6_prod_cli"] = arms["W0"]["wrfout_sha256"] == b6

wn3 = None
w = runs / "WN3"
if (w / "mtimes.json").is_file():
    mt = json.loads((w / "mtimes.json").read_text())
    launch = float((w / "launch_unix").read_text())
    pr = rd(w / "out" / "parallel_run.json") or {}
    starts = [e["t_s"] for e in pr.get("events", []) if "start" in e]
    per = {}
    for c, v in mt.items():
        rec = rd(w / "out" / c / "receipt.json") or {}
        sz = rec.get("sizing") or {}
        per[c] = {"n_wrfout": len(v), "wall_s": round(max(v.values()) - launch, 2) if v else None, "rc": rec.get("rc"),
                  "sizing_source": sz.get("source"), "plan_key": sz.get("plan_key"), "budget_bytes": sz.get("budget_bytes"),
                  "allocator": rec.get("allocator"), "aot_cache_warm": rec.get("aot_cache_warm"),
                  "aot_cold_executables": rec.get("aot_cold_executables"), "vram_peak_mib": rec.get("vram_peak_mib"),
                  "vmhwm_kb": rec.get("vmhwm_kb"), "start_utc": rec.get("start_utc"),
                  "c_auto_attested": bool(sz.get("source") == "c-auto-plan"
                                          and any("C-auto case pool" in l for l in (rec.get("allocator") or [])))}
    walls = [x["wall_s"] for x in per.values() if x["wall_s"] is not None]
    batch = max(walls) if walls else None
    case_h = 24 * len(per)
    line, compiled_new = consumer(w)
    dk = disk(w)
    together = bool(len(starts) == 3 and max(starts) <= 5.0 and pr.get("peak_concurrency") == 3)
    valid = bool(len(per) == 3 and together and all(x["rc"] == 0 and x["n_wrfout"] == 75 and x["c_auto_attested"]
                                                     and x["aot_cache_warm"] is True and not x["aot_cold_executables"]
                                                     for x in per.values()) and compiled_new == 0)
    v = round(batch / case_h, 3) if batch else None
    wn3 = {"cases": per, "batch_wall_s": batch, "case_hours": case_h, "HERO_PAR3_S_PER_CASE_H": v,
           "HERO_PAR3_BATCH_SECONDS": batch, "ratio_vs_cpu_wrf_3x4core_123_385": round(CPU_WRF_CASE_H / v, 2) if v else None,
           "valid_par3": valid, "started_together": together, "start_events_t_s": starts,
           "peak_concurrency": pr.get("peak_concurrency"), "launcher_wall_s": pr.get("wall_s"),
           "disturbed": dk.get("disturbed"), "label": ("DISTURBED" if dk.get("disturbed") else "CLEAN") if valid else "INVALID",
           "consumer": line, "compiled_new": compiled_new, "watch": watch(w), "disk": dk}

res = {"schema": "w42-bundle042.hero.v1", "session_root": str(A), "hero_prod": hero_prod, "hero_par3": wn3, "arms": arms,
       "aborted_tries": aborted, "arm_errors": errors, "identity": ident,
       "core_gate": (A / "core_gate.tsv").read_text().splitlines() if (A / "core_gate.tsv").is_file() else []}
(OUT / "HERO_RESULT.json").write_text(json.dumps(res, indent=1) + "\n")
lines = ["| slot | pool attested | whole s/fc-h (process start) | session clock | t0 | seg2+ median | stepping seg2+ median | "
         "first seg | await ms | JAX peak GB | VRAM PID MiB | compiled_new | sweep watch | CLI wait | PSI s |", "|---" * 15 + "|"]
for k, a in arms.items():
    lines.append(f"| {k} | {a['c_auto_attested']} | {a['whole_s_per_fc_h']['harness_launch']} | {a['whole_s_per_fc_h']['session_launch']} | "
                 f"{a['whole_s_per_fc_h']['measured_process_t0']} | {a['seg2plus_median']} | {a['stepping_seg2plus_median']} | "
                 f"{a['first_segment_s_per_fc_h']} | {a['disk'].get('await_ms')} | {round((a['jax_peak_bytes'] or 0) / 1e9, 2)} | "
                 f"{a['vram_peak_mib_pid_csv']} | {a['compiled_new']} | {(a['watch']['sweep_rule'] or {}).get('within_rule')} | "
                 f"{a['watch']['cli_main_wait_share']} | {a['watch']['cgroup_psi_some_s']} |")
lines += ["", f"HERO PROD: {json.dumps(hero_prod)}", f"HERO PAR3: {json.dumps({k: v for k, v in (wn3 or {}).items() if k != 'cases'})}",
          f"WN3 cases: {json.dumps((wn3 or {}).get('cases'))}", f"identity: {json.dumps(ident)}", f"aborted tries: {aborted}",
          f"arm errors: {errors}"]
(OUT / "HERO_TABLE.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
hp, hw = hero_prod or {}, wn3 or {}
print("HERO_LINE " + " ".join(f"{k}={v}" for k, v in (
    ("HERO_PROD_S_PER_FCH", hp.get("HERO_PROD_S_PER_FCH")), ("HERO_PROD_SECONDS", hp.get("HERO_PROD_SECONDS")),
    ("x_vs_CPU_92.3", hp.get("ratio_vs_cpu_wrf_92_3")), ("prod_label", hp.get("label")), ("prod_arms", ",".join(hp.get("arms", []))),
    ("HERO_PAR3_S_PER_CASE_H", hw.get("HERO_PAR3_S_PER_CASE_H")), ("HERO_PAR3_BATCH_SECONDS", hw.get("HERO_PAR3_BATCH_SECONDS")),
    ("x_vs_CPU_123.385", hw.get("ratio_vs_cpu_wrf_3x4core_123_385")), ("par3_label", hw.get("label")))))
