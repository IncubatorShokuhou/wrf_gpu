"""B1 v0.4.1 release benchmark, CPU follow-up after the lock (E41 before any cache cleanup, E166/E177).

usage: JAX_PLATFORMS=cpu python followup.py ROOT [--skip-e41] [--skip-d6] [--skip-ids]
Order: E41 (before any cache cleanup) -> par24 nested D6 -> identities -> summaries; every heavy step waits while a
foreign QUIET flag exists. Writes ROOT/RESULT.json; missing evidence is reported as missing, never as PASS:
  S1_proxy   = scripts/bench_summary.py of the program over the harness receipts (prime as cold, warm1, warm2): seg 2-3
               mean of the warm arms, A/A, vs RC2 3.088 / LW9 5.633 s/fc-h;
  S1_whole   = PROD 24 h warm harness arm: (last original wrfout mtime - receipt t0_wall_utc) / 24 from the E162 endpoint
               receipt written by session.sh right after the arm; plus its bench_summary segments (median seg 2-24,
               pre-step, output); vs RC2 3.604 / LW9 6.457 s/fc-h;
  WN3        = per launcher arm (cold3, warm3, par24): parallel_run.json wall_s_per_case_hour (launcher start -> last case
               exit / case-hours, the D28 whole-run definition), last-file variant from the captured mtimes, per-case
               rc/warm/VmHWM/VRAM; par24 steady stepping throughput [I] = 1 / sum_i(1 / r_i), r_i = d01 publication
               cadence hours 2..24 of case i (all cases concurrent); vs CPU-WRF 3x4-core 123.385 and D28 13.49;
  E41        = scripts/wave_cubin_gate.py of the program on cache_prod and cache_wn3 (every rich JIT + AOT program);
  D6         = nested D6 gate of the pinned fix (manager 2026-10-18T08:23:19Z): the SHA-pinned F20 scorer (frozen
               manifest 9df4fab8, full CPU inventory, every frame's header/schema, nonfinite) of every par24 case and
               domain d01-d03 vs the original CPU-WRF frames (guarded staged copies, setup.py); a case whose par24 run did
               not finish rc 0 with 25 frames per domain is NOT_RUN, never PASS;
  identities = proofs/s1_port/release_identity.py of the snapshot (Swiss RD11, PROD s0_case_20260725) -> arms.release.sha,
               compared with the v0.4.1 default identities (Swiss b977712f / PROD 321aa651); core gate + host load.
"""
import hashlib, json, os, re, statistics, subprocess, sys, time
from datetime import datetime
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
EXPECTED_IDS = {"swiss": "b977712fe325c953f3ebbf8f83ea1ca0b9260833ef70981e48bd8a49410e142a",
                "prod": "321aa651067f1d074b25ebe164536bf27fea5dca151a80c2861dcbab8b58f96b"}


def quiet_wait(max_s=7200):
    """Wait while a foreign QUIET flag exists (heavy CPU/disk work is not allowed under it); returns seconds waited."""
    t0 = time.time()
    while Path("/tmp/wrf_gpu2_quiet").exists() and time.time() - t0 < max_s:
        time.sleep(30)
    return round(time.time() - t0, 1)


def identity_sha(path):
    """Default-program identity of a release_identity.py receipt: arms.release.sha (64 hex), else MISSING/INVALID."""
    path = Path(path)
    if not path.exists():
        return "MISSING"
    try:
        value = json.loads(path.read_text())["arms"]["release"]["sha"]
    except (ValueError, KeyError, TypeError):
        return "INVALID"
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else "INVALID"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def unix(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def wn3_arm(arm_dir, mtimes_path, cpu, d28):
    run = load(arm_dir / "parallel_run.json")
    if run is None:
        return "MISSING"
    mtimes = load(mtimes_path) or {}
    row = {k: run.get(k) for k in ("start_utc", "end_utc", "wall_s", "case_hours", "wall_s_per_case_hour", "peak_concurrency",
                                   "stop_reason")}
    row["cases"] = {}
    rates = {}
    for name, case in (run.get("cases") or {}).items():
        rec = load(arm_dir / name / "receipt.json") or {}
        files = mtimes.get(name, {})
        d01 = sorted((f, t) for f, t in files.items() if f.startswith("wrfout_d01_"))
        row["cases"][name] = {"rc": case.get("rc"), "wall_s": rec.get("wall_s"), "aot_cache_warm": rec.get("aot_cache_warm"),
                              "aot_cold_executables": rec.get("aot_cold_executables"), "concurrent_at_start": rec.get("concurrent_at_start"),
                              "vmhwm_gb": (rec.get("vmhwm_kb") or 0) / 1e6, "vram_peak_mib": rec.get("vram_peak_mib"),
                              "allocator": rec.get("allocator"), "cli_summary": rec.get("cli_summary"),
                              "frames": {d: sum(f.startswith(f"wrfout_{d}_") for f in files) for d in ("d01", "d02", "d03")}}
        if len(d01) >= 25:  # initial frame + 24 hourly: index k = forecast hour
            rates[name] = (d01[24][1] - d01[2][1]) / 22.0
    row["last_file_unix"] = max((t for files in mtimes.values() for t in files.values()), default=None)
    if row["last_file_unix"] and run.get("start_utc") and run.get("case_hours"):
        row["last_file_s_per_case_hour"] = round((row["last_file_unix"] - unix(run["start_utc"])) / run["case_hours"], 3)
    if rates and len(rates) == len(row["cases"]):
        row["d01_cadence_s_per_sim_h_hours2_24"] = {k: round(v, 3) for k, v in rates.items()}
        row["stepping_throughput_s_per_case_h_I"] = round(1.0 / sum(1.0 / v for v in rates.values()), 3)
    whole = row.get("wall_s_per_case_hour")
    row["vs_cpu_3x4core"] = {"cpu_s_per_case_h": cpu, "speedup_whole_run": round(cpu / whole, 2) if whole else None,
                             "speedup_stepping_I": round(cpu / row["stepping_throughput_s_per_case_h_I"], 2)
                             if row.get("stepping_throughput_s_per_case_h_I") else None,
                             "vs_D28_whole_run": round(whole / d28, 4) if whole else None}
    return row


if __name__ == "__main__":
    ROOT = Path(sys.argv[1]).resolve()
    SKIP = set(sys.argv[2:])
    plan = json.loads((ROOT / "PLAN.json").read_text())
    SNAP = Path(plan["snapshot"])
    ENV = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0", PYTHONPATH=str(SNAP / "src"))
    git = lambda *a: subprocess.run(["git", *a], cwd=SNAP, capture_output=True, text=True).stdout.strip()  # noqa: E731
    assert git("rev-parse", "HEAD") == plan["program"] and git("rev-parse", "HEAD:src") == plan["src_tree"], "snapshot moved"
    base = plan["baselines"]
    result = {"program": plan["program"], "src_tree": plan["src_tree"], "gpuwrf_tree": plan["gpuwrf_tree"]}
    log = (ROOT / "session.log").read_text() if (ROOT / "session.log").exists() else ""
    minutes = re.findall(r"END GPU part: (\d+) GPU-min", log)
    result["gpu_minutes"] = int(minutes[-1]) if minutes else None
    result["session_arms"] = re.findall(r"(?:S1|WN3) arm \S+ rc=\d+|SKIP \S+", log)

    result["quiet_waits_s"] = []
    result["E41"] = {}
    W = ROOT / "wn3_runs"
    proof = next(iter(sorted(W.glob("par24/*/wrfout/proofs/nested_pipeline_run.json"))), None) or \
        next(iter(sorted(W.glob("*/*/wrfout/proofs/nested_pipeline_run.json"))), None)
    for name, cache, run_proof in (("prod", ROOT / "cache_prod/jax", None), ("wn3", ROOT / "cache_wn3/jax", proof)):
        out = ROOT / f"e41_{name}"
        if "--skip-e41" not in SKIP and not (out / "wave_gate.json").exists() and any(cache.glob("*")):
            result["quiet_waits_s"].append(quiet_wait())
            command = [sys.executable, str(SNAP / "scripts/wave_cubin_gate.py"), str(cache), str(out)]
            if run_proof:
                command += ["--run-proof", str(run_proof)]
            subprocess.run(command, cwd=SNAP, env=ENV, timeout=3600)
        gate = load(out / "wave_gate.json")
        result["E41"][name] = {"gate": gate.get("e41_gate"), "hard_stack_gate": gate.get("hard_stack_gate"),
                               "coverage": (gate.get("coverage") or {}).get("status"),
                               "conservative_flags": gate.get("conservative_flags"), "wave_gate": str(out / "wave_gate.json"),
                               "run_proof": str(run_proof) if run_proof else None} if gate else "MISSING"

    result["D6"] = {"verdicts": {}, "cases": {}, "scorer": plan.get("scorer")}
    par_run = load(ROOT / "wn3_runs/par24/parallel_run.json") or {}
    for f20, d6case in (plan.get("d6") or {}).items():
        case = Path(d6case["case"])
        launcher = (par_run.get("cases") or {}).get(d6case["launcher_case"]) or {}
        gpu = ROOT / "wn3_runs/par24" / d6case["launcher_case"] / "wrfout"
        frames = {d: len(list(gpu.glob(f"wrfout_{d}_*"))) for d in d6case["domains"]} if gpu.is_dir() else {}
        ready = launcher.get("rc") == 0 and all(frames.get(d) == d6case["cpu_frames"] // len(d6case["domains"])
                                                for d in d6case["domains"])
        row = {"par24_rc": launcher.get("rc"), "gpu_frames": frames, "domains": {}}
        pins_ok = all(sha256(ROOT / "tools" / n) == v for n, v in (plan.get("scorer") or {}).get("pins", {}).items())
        if ready and pins_ok and "--skip-d6" not in SKIP and not all((case / f"D6_{d}.json").exists() for d in d6case["domains"]):
            result["quiet_waits_s"].append(quiet_wait())
            with open(case / "score.log", "a") as handle:
                row["scorer_rc"] = subprocess.run([sys.executable, str(ROOT / "tools/score_with_handles.py"), str(SNAP), str(case),
                                                   "--domains", *d6case["domains"]], cwd=SNAP, env=ENV, stdout=handle,
                                                  stderr=subprocess.STDOUT, timeout=5400).returncode
        for d in d6case["domains"]:
            got = load(case / f"D6_{d}.json")
            if got and ready:
                row["domains"][d] = {k: got.get(k) for k in ("verdict", "paired_frames", "coverage_complete", "manifest_sha256")} | {
                    "bounded_misses": len(got.get("failures", [])), "nonfinite": len(got.get("nonfinite", [])),
                    "incomplete": len(got.get("incomplete", [])),
                    "missing_vs_cpu_frames": sum(bool(r.get("missing_vs_cpu")) for r in got.get("by_frame", [])),
                    "schema_mismatch_frames": sum(bool(r.get("schema_mismatches")) for r in got.get("by_frame", [])),
                    "header_missing_frames": sum(bool(r.get("missing_header_attributes")) for r in got.get("by_frame", []))}
            else:
                row["domains"][d] = {"verdict": "NOT_RUN" if not ready else ("SCORER_PIN_MISMATCH" if not pins_ok else "MISSING")}
        result["D6"]["cases"][f20] = row
        result["D6"]["verdicts"][f20] = {d: v["verdict"] for d, v in row["domains"].items()}
    flat = [v for c in result["D6"]["verdicts"].values() for v in c.values()]
    result["D6"]["gate"] = "PASS" if flat and all(v == "PASS" for v in flat) else ("FAIL" if "FAIL" in flat else "INCOMPLETE")

    result["identities"], result["identities_expected_v041_default"] = {}, {}
    for name, extra in (("swiss", []), ("prod", [plan["prod_input"]])):
        out = ROOT / f"identity_{name}.json"
        if "--skip-ids" not in SKIP and not out.exists():
            result["quiet_waits_s"].append(quiet_wait())
            with open(ROOT / f"identity_{name}.log", "a") as handle:
                subprocess.run([sys.executable, str(SNAP / "proofs/s1_port/release_identity.py"), str(SNAP), str(out), *extra],
                               cwd=SNAP, env=dict(ENV, OMP_NUM_THREADS="1"), stdout=handle, stderr=subprocess.STDOUT, timeout=2400)
        result["identities"][name] = identity_sha(out)
        result["identities_expected_v041_default"][name] = {"expected": EXPECTED_IDS[name],
                                                            "equal": result["identities"][name] == EXPECTED_IDS[name]}

    S1 = ROOT / "S1"
    receipts = {k: S1 / arm / "run/receipt.json" for k, arm in (("cold", "prime"), ("warm1", "warm1"), ("warm2", "warm2"))}
    present = {k: v for k, v in receipts.items() if v.exists()}
    bench = lambda out, tag, arms: subprocess.run(  # noqa: E731
        [sys.executable, str(SNAP / "scripts/bench_summary.py"), "--out", str(out), "--tag", tag, "--cache", str(ROOT / "cache_prod"),
         "--src", str(SNAP), "--src-rev", plan["program"], "--src-tree", plan["gpuwrf_tree"], "--xla-flags="]
        + [f"{k}={v}" for k, v in arms.items()], cwd=SNAP, env=ENV, timeout=600)
    result["S1_proxy"] = "MISSING"
    if {"warm1", "warm2"} <= set(present):
        bench(S1 / "bench.json", "w41bdyrec_b1_v041", present)
        b = load(S1 / "bench.json")
        if b:
            value = b.get("s1_warm_s_per_fc_h")
            cold = (b.get("arms") or {}).get("cold") or {}
            result["S1_proxy"] = {"bench_json": str(S1 / "bench.json"), "s1_warm_s_per_fc_h": value, "aa": b.get("aa"),
                                  "headline_valid": b.get("headline_valid"), "reject_reasons": b.get("reject_reasons"),
                                  "cold_arm_t_end_s": cold.get("t_end_s"),
                                  "cold_arm_segments_s_per_fc_h": [r.get("s_per_fc_h") for r in cold.get("segments", [])],
                                  "vs_RC2_3.088": round(value / base["S1_proxy_RC2"], 4) if value else None,
                                  "vs_LW9_5.633": round(value / base["S1_proxy_LW9"], 4) if value else None}
    result["S1_arms_missing"] = sorted(set(receipts) - set(present))

    result["S1_whole"] = "MISSING"
    endpoint = load(S1 / "whole24/endpoint.json")
    if endpoint:
        bench(S1 / "bench_whole24.json", "w41bdyrec_b1_v041_whole24", {"whole24": endpoint["benchmark_receipt"]})
        arm = ((load(S1 / "bench_whole24.json") or {}).get("arms") or {}).get("whole24") or {}
        segs = arm.get("segments", [])
        receipt = load(endpoint["benchmark_receipt"]) or {}
        first = (receipt.get("derived") or {}).get("segments") or [{}]
        s_fch = endpoint["wall_s"] / endpoint["hours"] if endpoint.get("wall_s") else None
        result["S1_whole"] = {"endpoint": str(S1 / "whole24/endpoint.json"), "valid": endpoint.get("valid"),
                              "end_reason": receipt.get("end_reason"), "wall_s_t0_to_last_file": endpoint.get("wall_s"),
                              "s_per_fc_h": round(s_fch, 3) if s_fch else None,
                              "launch_to_t0_s": round(endpoint["t0_wall_utc"] - endpoint["launch_unix"], 3),
                              "n_files": len(endpoint.get("file_write_unix", {})), "n_outputs_receipt": endpoint.get("n_outputs_receipt"),
                              "pre_step_s": first[0].get("t_start"), "first_segment_s_per_fc_h": segs[0]["s_per_fc_h"] if segs else None,
                              "median_seg2_24_s_per_fc_h": statistics.median(r["s_per_fc_h"] for r in segs[1:]) if len(segs) > 1 else None,
                              "output_s_per_fc_h_seg2plus": arm.get("output_s_per_fc_h"), "n_segments": len(segs),
                              "vs_RC2_3.604": round(s_fch / base["S1_whole_run_RC2"], 4) if s_fch else None,
                              "vs_LW9_6.457": round(s_fch / base["S1_whole_LW9"], 4) if s_fch else None}

    W = ROOT / "wn3_runs"
    result["WN3"] = {name: wn3_arm(W / name, W / f"{name}.mtimes.json", base["cpu_wrf_3x4core_s_per_case_h"],
                                   base["wn3_par_D28_s_per_case_h"]) for name in ("cold3", "warm3", "par24")}
    warm3 = result["WN3"]["warm3"]
    par = result["WN3"]["par24"]
    if isinstance(warm3, dict) and isinstance(par, dict) and par.get("stepping_throughput_s_per_case_h_I"):
        result["WN3"]["note"] = ("warm3 = solo warm 3 h reference (0227) that recorded the warm host peak for the par24 "
                                 "admission; par24 is the release benchmark row")

    gate_rows = [line.split("\t") for line in (ROOT / "core_gate.tsv").read_text().splitlines()] \
        if (ROOT / "core_gate.tsv").exists() else []
    checks = [json.loads(line) for line in (ROOT / "core_checks.jsonl").read_text().splitlines()] \
        if (ROOT / "core_checks.jsonl").exists() else []
    result["core_gate"] = {"arms": {r[0]: {"utc": r[1], "state": r[2], "checks": int(r[3])} for r in gate_rows},
                           "refusals": sum(not c["free"] for c in checks),
                           "disclosed_offenders": [c["offenders"] for c in checks if not c["free"]][-3:],
                           "host_load": str(ROOT / "HOST_LOAD.txt")}
    (ROOT / "RESULT.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(json.dumps(result, indent=1, default=str)[:6000])
