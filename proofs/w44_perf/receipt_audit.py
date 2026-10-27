"""Reconcile preserved release timing receipts without importing JAX or reading raw traces.

All comparisons here are historical accounting, never a same-session speed verdict.
Run from the checkout: python proofs/w44_perf/receipt_audit.py --out DIR.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def provenance(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def intervals_union(intervals: list[tuple[float, float]]) -> float:
    end = float("-inf")
    total = 0.0
    for lo, hi in sorted(intervals):
        total += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return total


def audit(rec: dict, endpoint: dict | None = None) -> dict:
    segs = rec.get("segments") or rec["derived"]["segments"]
    assert rec["rc"] == 0 and rec["device"]["platform"] == "gpu"
    start = segs[0]["t_start"]
    last_return = segs[-1]["t_return"]
    end = rec["t_end"]
    phases = {"pre_first_segment_s": start, "segment_span_s": last_return - start,
              "post_last_segment_s": end - last_return}
    assert abs(sum(phases.values()) - end) < 1e-6
    rows = []
    for i, seg in enumerate(segs):
        lo = seg["t_start"]
        hi = segs[i + 1]["t_start"] if i + 1 < len(segs) else last_return
        fc_h = seg["root_steps"] * 54.0 / 3600
        output = intervals_union([
            (max(lo, o["t_start"]), min(hi, o["t_start"] + o["s"]))
            for o in rec.get("outputs", [])
            if o["t_start"] < hi and o["t_start"] + o["s"] > lo])
        rows.append({"k": seg["k"], "forecast_hours": fc_h,
                     "segment_period_excluding_final_tail_s": hi - lo,
                     "period_s_per_fch": (hi - lo) / fc_h,
                     "output_callback_union_s": output if "outputs" in rec else None,
                     "stepping_s_per_fch": (hi - lo - output) / fc_h if "outputs" in rec else None})
    events = rec.get("compile_events", [])
    compile_by_phase = {}
    for label, lo, hi in (("pre_first_segment", 0, start),
                          ("segments", start, last_return),
                          ("tail", last_return, end)):
        compile_by_phase[label] = {}
        for ev in events:
            if lo <= ev["t"] <= hi:
                key = ev["event"]
                compile_by_phase[label][key] = compile_by_phase[label].get(key, 0) + ev["s"]
    warm = rows[1:]
    hours = sum(r["forecast_hours"] for r in rows)
    result = {
        "git_head": rec["git_head"], "src_tree": rec["src_tree"],
        "device": rec["device"], "forecast_hours": hours,
        "receipt_t_end_s": end, "receipt_t_end_s_per_fch": end / hours,
        "phases_disjoint": phases, "segments": rows,
        "seg2plus_period_median_s_per_fch": statistics.median(r["period_s_per_fch"] for r in warm),
        "seg2plus_stepping_median_s_per_fch": (statistics.median(r["stepping_s_per_fch"] for r in warm)
                                                  if "outputs" in rec else None),
        "compile_duration_sums_by_event_not_additive": compile_by_phase,
        "allocator": rec.get("allocator"), "pre_exec": rec.get("pre_exec"),
        "jax_memory_stats": rec.get("jax_memory_stats"),
        "host_vmhwm_kb": rec.get("host_vmhwm_kb"),
        "vram_peak_mib": rec.get("derived", {}).get("vram_peak_mib"),
        "record_has_outputs": "outputs" in rec,
    }
    if endpoint:
        last = max(endpoint["file_write_unix"].values())
        result["endpoint"] = {
            "harness_launch_to_last_file_s": last - endpoint["harness_launch_wall_utc"],
            "measured_t0_to_last_file_s": last - endpoint["t0_wall_utc"],
            "session_launch_to_last_file_s": last - endpoint["session_launch_unix"],
        }
    return result


def dmon(path: Path, rec: dict, endpoint: dict) -> dict:
    """Summarize saved one-second utilization only; no timeline/cause inference."""
    lines = (gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()).splitlines()
    # Producer command is `nvidia-smi dmon -s pucm -o T`: timestamp, gpu,
    # pwr, gtemp, mtemp, sm, mem, enc, dec, jpg, ofa, mclk, pclk, fb, bar1.
    header = next(line.lstrip("#").split() for line in lines if line.startswith("#") and " gpu " in line)
    sm_index = header.index("sm")
    samples = []
    for line in lines:
        parts = line.split()
        if not parts or line.startswith("#") or len(parts) != len(header):
            continue
        try:
            samples.append({"time": parts[0], "sm_pct": int(parts[sm_index])})
        except ValueError:
            pass
    vals = [s["sm_pct"] for s in samples]
    assert vals, f"No utilization samples parsed: {path}"
    # dmon emits host-local time without a date. Validate the timezone against
    # the preserved launch timestamp rather than silently shifting the series.
    tz = ZoneInfo("Atlantic/Canary")
    date = datetime.fromtimestamp(endpoint["session_launch_unix"], tz).date()
    for s in samples:
        s["unix"] = datetime.combine(date, time.fromisoformat(s["time"]), tzinfo=tz).timestamp()
    assert abs(samples[0]["unix"] - endpoint["session_launch_unix"]) < 2
    segs = rec["segments"]
    t0 = rec["t0_wall_utc"]
    # Discard one second at each boundary: sampled rates straddle event edges.
    phases = {}
    for name, lo, hi in (("startup", samples[0]["unix"], t0 + segs[0]["t_start"]),
                          ("stepping_all", t0 + segs[0]["t_start"], t0 + segs[-1]["t_return"]),
                          ("stepping_seg2plus", t0 + segs[1]["t_start"], t0 + segs[-1]["t_return"])):
        v = [s["sm_pct"] for s in samples if lo + 1 <= s["unix"] <= hi - 1]
        phases[name] = {"n": len(v), "mean_sm_pct": statistics.mean(v) if v else None,
                        "min_sm_pct": min(v) if v else None,
                        "below50": sum(x < 50 for x in v), "below80": sum(x < 80 for x in v)}
    return {"source": provenance(path), "n_samples": len(vals),
            "mean_sm_pct": statistics.mean(vals) if vals else None,
            "samples_below_50_pct": sum(v < 50 for v in vals),
            "samples": samples, "phases_with_1s_boundary_margin": phases,
            "timezone": "Atlantic/Canary, first sample within 2s of saved launch",
            "scope": "device-wide sampled SM utilization, including desktop; not kernel busy union or millisecond gap attribution"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    old = ROOT / "docs/release/evidence/v032/prod_receipt_selected.json"
    hero = ROOT / "proofs/w42_bundle042/receipts/hero_b979188128e0"
    sources = [old]
    arms = {"v032_selected": audit(load(old))}
    for name in ("H1", "H2"):
        rp = hero / name / "receipt.json.gz"
        ep = hero / name / "endpoint.json"
        sources.extend([rp, ep])
        rec, endpoint = load(rp), load(ep)
        arms["v042_" + name] = audit(rec, endpoint)
        arms["v042_" + name]["dmon"] = dmon(hero / name / "gpu_dmon.txt.gz", rec, endpoint)
    result = {"schema": "w44-perf.historical_receipt_audit.v1", "evidence": "M: preserved GPU receipts",
              "causal_version_speed_verdict": "UNKNOWN: different sessions/allocators/harness timing bases",
              "sources": [provenance(p) for p in sources], "arms": arms}
    (args.out / "HISTORICAL_RECEIPT_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, arm in arms.items():
        print(json.dumps({"arm": name, "whole_receipt_s_per_fch": arm["receipt_t_end_s_per_fch"],
                          "phases": arm["phases_disjoint"],
                          "period_median": arm["seg2plus_period_median_s_per_fch"],
                          "stepping_median": arm["seg2plus_stepping_median_s_per_fch"],
                          "endpoint": arm.get("endpoint"), "vram_mib": arm["vram_peak_mib"]}))


if __name__ == "__main__":
    main()
