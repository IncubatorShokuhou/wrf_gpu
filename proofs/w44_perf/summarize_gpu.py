"""Sanitized bench JSON from completed night arms; missing evidence stays open.

No JAX import, raw profiler read, or permissive timing exclusion. Official S0
terminal periods and call-only periods are reported separately (E196).
"""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("s0_summary", ROOT / "scripts/bench_summary.py")
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)
CASE = "<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725"
GROUPS = {"v032": ["A32_1", "A32_2"], "v040": ["C40_1", "C40_2"], "v043": ["B43_1", "B43_2", "B43_3"]}
HEADS = {"v032": "5f682e5ae933d569ebbeb11d78f119ba291f9283", "v040": "7bb5ae669b2e4b863d6bafdd659a03fea026004f",
         "v043": "888804310bb6bfc0237177c0d5e5452a29d5abf6"}


def arm(path):
    rp = path / "run/receipt.json"
    if not rp.is_file() or not (path / "endpoint.json").is_file():
        return {"complete": False, "reason": "missing receipt/endpoint", "path": str(path)}
    r = json.loads(rp.read_text())
    ep = json.loads((path / "endpoint.json").read_text())
    rows = bs.segment_rows(r)
    warm = rows[1:]
    cache = Counter(e["event"] for e in r.get("cache_events", []))
    requests, hits = cache["/jax/compilation_cache/compile_requests_use_cache"], cache["/jax/compilation_cache/cache_hits"]
    misses = cache["/jax/compilation_cache/cache_misses"]
    zero_compiles = bool(requests > 0 and requests == hits and misses == 0)
    cli = r.get("cli_argv", [])
    input_ok = "--input-dir" in cli and cli[cli.index("--input-dir") + 1] == CASE
    work_ok = input_ok and len(rows) == 3 and sum(s.get("root_steps", 0) for s in r["segments"]) == 200
    device_ok = (r.get("device") or {}).get("platform") == "gpu"
    alloc = r.get("allocator") or {}
    pool_ok = alloc.get("preallocate") == "true" and alloc.get("mem_fraction") == "0.25"
    complete = bool(r.get("rc") == 0 and ep.get("valid") and len(ep.get("files", {})) == 6 and work_ok and device_ok)
    phases = {"pre_segment_s": r["segments"][0]["t_start"],
              "segment_span_s": r["segments"][-1]["t_return"] - r["segments"][0]["t_start"],
              "tail_to_receipt_s": r["t_end"] - r["segments"][-1]["t_return"]}
    token = r.get("env", {}).get("GPUWRF_GPU_LOCK_TOKEN")
    return {"complete": complete, "warm_cache_attested": zero_compiles, "common_pool_attested": pool_ok,
            "receipt": str(rp), "receipt_sha256": hashlib.sha256(rp.read_bytes()).hexdigest(),
            "git_head": r.get("git_head"), "src_tree": r.get("src_tree"), "jax_version": r.get("jax_version"),
            "lock_session_sha256": hashlib.sha256(token.encode()).hexdigest() if token else None,
            "device": r.get("device"), "rc": r.get("rc"), "allocator": alloc, "segments": rows,
            "official_proxy_s_per_fch": statistics.mean(x["s_per_fc_h"] for x in warm) if warm else None,
            "official_stepping_s_per_fch": statistics.mean(x["stepping_s_per_fc_h"] for x in warm) if warm else None,
            "call_only_proxy_s_per_fch": statistics.mean((s["t_return"] - s["t_start"]) / (s["root_steps"] * .015)
                                                        for s in r["segments"][1:]),
            "phases": phases, "whole_3h_s_per_fch": ep.get("whole_to_last_file_s", 0) / 3,
            "pre_segment_s": phases["pre_segment_s"],
            "harness_3h_s_per_fch": ep.get("harness_to_last_file_s", 0) / 3,
            "vram_peak_mib": r.get("derived", {}).get("vram_peak_mib"), "host_vmhwm_kb": r.get("host_vmhwm_kb"),
            "jax_memory_stats": r.get("jax_memory_stats"), "cache_events": dict(cache),
            "errors": r.get("errors"),
            "scope": "3h iteration case; whole-process rate is not 24h S1. Host-outside-call estimate is not GPU idle."}


def main():
    global CASE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("<USER_HOME>/wrf_gpu2_lanes/w44-perf/gpu/runs"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--case", default=CASE, help="Expected input grid; production default unchanged")
    p.add_argument("--plan", type=Path, help="Immutable preregistration for a fresh complete session; never selects individual arms")
    a = p.parse_args()
    CASE=a.case
    plan = json.loads(a.plan.read_text()) if a.plan else None
    selection = plan["groups"] if plan else GROUPS
    metric_keys = plan["aa_metrics"] if plan else ["official_proxy_s_per_fch", "official_stepping_s_per_fch", "call_only_proxy_s_per_fch"]
    if plan:
        assert set(selection) == set(HEADS)
        assert sorted((v, n) for v, names in selection.items() for n in names) == sorted(map(tuple, plan["order"]))
        assert all(len(names) >= 2 for names in selection.values())
    arms = {name: arm(a.runs / name) for names in selection.values() for name in names}
    groups = {}
    for version, names in selection.items():
        entries = [arms[name] for name in names]
        valid = all(r["complete"] and r.get("warm_cache_attested") and r.get("common_pool_attested")
                    and r.get("git_head") == HEADS[version] for r in entries)
        metrics = {}
        if valid:
            for key in metric_keys:
                values = [r[key] for r in entries]
                mean = statistics.mean(values)
                spread = (max(values) - min(values)) / mean * 100
                metrics[key] = {"values": values, "mean": mean, "aa_spread_pct": spread, "aa_pass_5pct": spread <= 5}
        groups[version] = {"valid_arms": valid, "metrics": metrics}
    identities = {(r.get("lock_session_sha256"), r.get("jax_version"),
                   (r.get("device") or {}).get("kind")) for r in arms.values() if r.get("complete")}
    session_ok = len(identities) == 1 and next(iter(identities))[0] is not None if identities else False
    required = metric_keys if plan else ["official_proxy_s_per_fch", "official_stepping_s_per_fch"]
    comparable = session_ok and all(v["valid_arms"] and all(v["metrics"][k]["aa_pass_5pct"] for k in required) for v in groups.values())
    result = {"schema": "w44-perf.bench.v1", "headline_valid": comparable, "case": CASE,
              "scope": "same-session3hproxy; all arms retained, no posthoc subset; no new WRF fidelity claim",
              "arms": arms, "groups": groups, "one_session_jax_device": session_ok,
              "plan_sha256": hashlib.sha256(a.plan.read_bytes()).hexdigest() if a.plan else None,
              "reject_reason": None if comparable else "missing/incomplete arm, warm-cache/pool attestation, or preregistered A/A rule"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"headline_valid": comparable, "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
