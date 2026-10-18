"""GPU-arm verdicts (CPU, after the lock): bitwise pairs, executable reuse and member cost from the receipts."""
import hashlib
import json
import sys
from pathlib import Path

session = Path(sys.argv[1])
loop = session / "loop"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731


def pair(a_dir, b_dir, names):
    files = {}
    for name in names:
        a, b = a_dir / name, b_dir / name
        files[name] = {"in_process": sha(a) if a.is_file() else None, "separate": sha(b) if b.is_file() else None}
    same = bool(files) and all(v["in_process"] and v["in_process"] == v["separate"] for v in files.values())
    return {"identical": same, "files": files}


cont = sorted(p.name for p in (session / "separate/cont").glob("wrfout_d01_*"))
stream = loop / "streams/zero_restart/m01"
result = {
    "window0_in_process_vs_separate": pair(stream, session / "separate/cont", [n for n in cont if n <= "wrfout_d01_2026-07-26_06:00:00"]),
    "restart_seam_zero_vs_continuous": pair(stream, session / "separate/cont", [n for n in cont if n > "wrfout_d01_2026-07-26_06:00:00"]),
    "window1_file_seam_in_process_vs_separate": pair(loop / "c01/zero_file/fc/m01", session / "separate/c01",
                                                    sorted(p.name for p in (session / "separate/c01").glob("wrfout_d01_*"))),
}
# CYCLE_RECEIPT.json is written in a finally block, which a bound SIGTERM skips: fall back to the per-member
# receipts (each member run writes its own, in run order).
cycle_receipt = loop / "CYCLE_RECEIPT.json"
receipt = json.loads(cycle_receipt.read_text()) if cycle_receipt.is_file() else None
if receipt is not None:
    runs = receipt["driver"]["runs"]
else:
    runs = [json.loads(p.read_text()) for p in sorted(loop.glob("**/member_receipt_*.json"),
                                                      key=lambda p: int(p.stem.rsplit("_", 1)[1]))]
walls = [r["wall_seconds"] for r in runs]
sources = [sorted({d.get("source") for d in r["aot"].values()}) for r in runs]
summary = {
    "platform": sorted({r.get("platform") for r in runs}), "cycle_receipt": receipt is not None,
    "platform_asserted": receipt.get("platform_asserted") if receipt else None,
    "member_runs": len(runs), "all_green": all(r.get("verdict") == "PIPELINE_GREEN" for r in runs),
    "verdicts": {k: v["identical"] for k, v in result.items()},
    "member_wall_first_s": walls[0] if walls else None,
    "member_wall_rest_median_s": sorted(walls[1:])[len(walls[1:]) // 2] if len(walls) > 1 else None,
    "member_wall_rest_mean_s": sum(walls[1:]) / len(walls[1:]) if len(walls) > 1 else None,
    "process_peak_rss_gib": max((r.get("process_peak_rss_kib") or 0) for r in runs) / 2**20 if runs else None,
    "aot_sources_per_run": sources, "loop_wall_s": receipt["wall_seconds"] if receipt else None,
    "prepare_next": {str(w["cycle"]): {arm: {"wall_s": v.get("wall_seconds"),
                                             "inject_wall_s": (v.get("injection") or {}).get("wall_seconds")}
                                       for arm, v in w.get("prepare_next", {}).items()}
                     for w in receipt["windows"]} if receipt else None,
    "pairs": result,
}
(session / "GPU_ARM_SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in ("pairs", "aot_sources_per_run")}, indent=1))
