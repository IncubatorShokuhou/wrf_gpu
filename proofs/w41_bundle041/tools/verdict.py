"""w41-bundle041: compile the RW1-RW4 real-world gate table from the session receipts (CPU only, reads files).

usage: python3 verdict.py S_DIR P_DIR OUT_JSON [--d6 CASE_DIR] [--e41 WAVE_GATE_JSON] [--aotkey DIR]
Missing evidence is reported as MISSING/INCOMPLETE, never as PASS.
  arm facts   rc, consumer line (compiled_new/aliases_new/runtime_new/layer_compiled), bundle check before/after, every
              nested-aot line (domain, loaded, source, cheap key, hlo), /usr/bin/time max RSS + elapsed, per-PID VRAM max
              (nvidia-smi compute-apps, 0.5 s), first/last wrfout mtime vs the arm's t0 (E162), proof wall clocks
  RW1         B1 (fresh process, sealed bundle) vs A3 (cold priming run, same case/hours): rc 0, compiled_new 0, every
              nested-aot program loaded=true from aot_blob, 0 loaded=false, P BUNDLE_OK after, whole-file SHA of every
              wrfout identical; same for Swiss B5 vs A1 and PROD B6 vs A2; startup (first forecast frame) vs v0.4.0 B7
  RW2/RW4     B2 0502, B3 0614 (48 h), B4 0227 48 h-input: compiled_new 0, all programs loaded, check after OK; the
              GPU cheap keys of d01 and fused/d02 are ONE key per program over 0227/0502/0614/48h-input and are sealed
              blobs of the bundle manifest; CPU gpuwrf aot-key pairs SHARED
  RW3         B1 vs B7 (v0.4.0 bundle, same inputs/hours/session): whole-run s/fc-h = (last wrfout mtime - t0)/72,
              stepping median (d01 frame cadence leads 2-72), peak VRAM per PID, host max RSS, compile s = cold - warm
              process wall of the same arm (v0.4.1: A3 - B1 at 72 h; v0.4.0: its prime - replay at 3 h); > 10 %
              regression -> STOP; D6/integrity from d6_72h.py receipts
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

S, P, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
opt = {sys.argv[i]: sys.argv[i + 1] for i in range(4, len(sys.argv) - 1, 2)}
R = S / "runs"
AOT = re.compile(r"\[gpuwrf:nested-aot\] domain=(\S+) loaded=(\S+) source=(\S+)(.*)")


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def arm(name: str) -> dict:
    d = R / name
    if not d.is_dir():
        return {"present": False}
    rec = {"present": True, "rc": text(d / "rc.txt").strip() or None}
    scope = text(R / f"{name}.scope.log")
    m = [l for l in scope.splitlines() if l.startswith("BUNDLE_CONSUMER rc=")]
    rec["consumer"] = m[-1] if m else None
    if rec["consumer"]:
        rec["consumer_counts"] = {k: int(v) for k, v in re.findall(r"(\w+_new|layer_compiled)=(\d+)", rec["consumer"])}
    for c in ("check_before", "check_after"):
        rec[c] = text(d / f"{c}.txt").strip().splitlines()[-1:] or None
    lines = []
    for line in text(d / "run.log").splitlines():
        g = AOT.search(line)
        if g:
            rest = g.group(4)
            key = re.search(r"/k_([0-9a-f]{64})\.xlaexec", rest) or re.search(r"cheap_key=([0-9a-f]+)", rest)
            hlo = re.search(r"hlo=([0-9a-f]+)", rest)
            lines.append(dict(domain=g.group(1), loaded=g.group(2) == "true", source=g.group(3),
                              key=key.group(1) if key else None, hlo=hlo.group(1) if hlo else None))
    rec["aot"] = lines
    rec["all_loaded"] = bool(lines) and all(x["loaded"] and x["source"] == "aot_blob" for x in lines)
    tv = text(d / "time.txt")
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", tv)
    rec["max_rss_gib"] = round(int(rss.group(1)) / 2**20, 3) if rss else None
    try:
        t0 = float(text(d / "t0_unix")); t1 = float(text(d / "t_end_unix"))
        rec["process_wall_s"] = round(t1 - t0, 2)
    except ValueError:
        t0 = None
    mt = load(d / "mtimes.json") or {}
    rec["n_wrfout"] = len(mt)
    if mt and t0:
        d01 = sorted(v for k, v in mt.items() if k.startswith("wrfout_d01_"))
        rec["first_file_s"] = round(min(mt.values()) - t0, 2)
        rec["last_file_s"] = round(max(mt.values()) - t0, 2)
        rec["d01_frames"] = len(d01)
        if len(d01) > 2:
            rec["d01_lead1_s"] = round(d01[1] - t0, 2)
            steps = [b - a for a, b in zip(d01[1:], d01[2:])]
            rec["stepping_median_s_per_h"] = round(statistics.median(steps), 3)
    proof = load(d / "proof/nested_pipeline_run.json") or {}
    rec["proof_wall_total_s"] = proof.get("wall_clock_total_s")
    rec["proof_wall_forecast_s"] = proof.get("wall_clock_forecast_only_s")
    peak = 0
    try:
        with open(d / "vram_apps.csv") as fh:
            for row in csv.reader(fh):
                if len(row) >= 3 and row[2].strip().isdigit():
                    peak = max(peak, int(row[2]))
        rec["vram_peak_pid_mib"] = peak or None
    except OSError:
        rec["vram_peak_pid_mib"] = None
    rec["sha"] = dict(l.split()[::-1] for l in text(d / "wrfout_sha.txt").splitlines() if l.strip())
    return rec


ARMS = ["A1_swiss", "A2_prod", "A3_wn3_0227_72h", "B2_wn3_0502_3h", "B4_wn3_0227in48h_3h", "B5_swiss_3h", "B6_prod_3h",
        "B3_wn3_0614_48h", "B1_wn3_0227_72h", "B7_v040_wn3_0227_72h"]
arms = {n: arm(n) for n in ARMS}
res = {"session_log": str(S / "session.log"), "arms": {k: {kk: vv for kk, vv in v.items() if kk != "sha"}
                                                       for k, v in arms.items()}}


def replay_ok(a):
    c = a.get("consumer_counts") or {}
    return (a.get("present") and a.get("rc") == "0" and c.get("compiled_new") == 0 and a["all_loaded"]
            and bool(a.get("check_after")) and a["check_after"][0].startswith("BUNDLE_OK missing=0 changed=0 new=0 "))


def bytes_equal(a, b):
    sa, sb = arms[a].get("sha") or {}, arms[b].get("sha") or {}
    return {"files": len(sa), "identical": bool(sa) and sa == sb}


rw1 = {}
for warm, cold in (("B1_wn3_0227_72h", "A3_wn3_0227_72h"), ("B5_swiss_3h", "A1_swiss"), ("B6_prod_3h", "A2_prod")):
    rw1[warm] = dict(replay_ok=bool(replay_ok(arms[warm])), cold=cold, **bytes_equal(warm, cold),
                     startup_lead1_s=arms[warm].get("d01_lead1_s"), process_wall_s=arms[warm].get("process_wall_s"))
rw1["verdict"] = "PASS" if all(v["replay_ok"] and v["identical"] for v in rw1.values()) else "FAIL_OR_INCOMPLETE"
res["RW1"] = rw1

wn3 = ["A3_wn3_0227_72h", "B1_wn3_0227_72h", "B2_wn3_0502_3h", "B3_wn3_0614_48h", "B4_wn3_0227in48h_3h"]
keys = {}
for n in wn3:
    for x in arms[n].get("aot") or []:
        if x["key"] and len(x["key"]) == 64:  # capture lines carry no/abbreviated keys; the lookup line has the full one
            keys.setdefault(x["domain"], {})[n] = x["key"]
manifest = load(P / "sealed/BUNDLE_MANIFEST.json") or {}
sealed_keys = {re.sub(r".*/(\w+)/k_([0-9a-f]{64})\.xlaexec$", r"\1:\2", f) for f in (manifest.get("files") or {})
               if re.search(r"/k_[0-9a-f]{64}\.xlaexec$", f)}
rw2 = {n: dict(replay_ok=bool(replay_ok(arms[n])), aot=arms[n].get("aot"), consumer=arms[n].get("consumer"))
       for n in ("B2_wn3_0502_3h", "B3_wn3_0614_48h", "B4_wn3_0227in48h_3h")}
one_key = {dom: (len({k for k in v.values() if k}) == 1 and all(v.get(n) for n in wn3)) for dom, v in keys.items()}
in_bundle = {dom: all(f"{dom.replace('fused/', 'fused_')}:{k}" in sealed_keys for k in set(v.values()) if k)
             for dom, v in keys.items()}
aotkey = {}
for f in sorted(Path(opt.get("--aotkey", "/nonexistent")).glob("*.json")):
    rep = load(f) or {}
    aotkey[f.stem] = {name: row.get("shared") for name, row in (rep.get("comparison") or {}).items()}
res["RW2_RW4"] = dict(arms=rw2, gpu_keys_by_program=keys, one_key_per_program=one_key,
                      keys_are_sealed_blobs=in_bundle, cpu_aot_key_pairs=aotkey,
                      verdict="PASS" if (all(v["replay_ok"] for v in rw2.values()) and keys and all(one_key.values())
                                         and all(in_bundle.values()) and aotkey
                                         and all(all(x) for x in aotkey.values() if x)) else "FAIL_OR_INCOMPLETE")

b1, b7, a3 = arms["B1_wn3_0227_72h"], arms["B7_v040_wn3_0227_72h"], arms["A3_wn3_0227_72h"]


def per_h(a, hours=72):
    return round(a["last_file_s"] / hours, 3) if a.get("last_file_s") else None


v040_prime = load(Path("<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0/prime/wn3_0227_3nest/proof/nested_pipeline_run.json")) or {}
v040_replay = load(Path("<DATA_ROOT>/wrf_gpu2/release_bundles/v0.4.0/prime/wn3_0227_3nest_replay/proof/nested_pipeline_run.json")) or {}
metrics = {
    "whole_run_s_per_fch": (per_h(b1), per_h(b7)),
    "stepping_median_s_per_h": (b1.get("stepping_median_s_per_h"), b7.get("stepping_median_s_per_h")),
    "startup_lead1_s": (b1.get("d01_lead1_s"), b7.get("d01_lead1_s")),
    "vram_peak_pid_mib": (b1.get("vram_peak_pid_mib"), b7.get("vram_peak_pid_mib")),
    "host_max_rss_gib": (b1.get("max_rss_gib"), b7.get("max_rss_gib")),
}
reg = {}
for k, (new, old) in metrics.items():
    reg[k] = dict(v041=new, v040=old, ratio=round(new / old, 4) if new and old else None,
                  regression_gt_10pct=bool(new and old and new / old > 1.10))
compile_new = (round(a3["proof_wall_total_s"] - b1["proof_wall_total_s"], 1)
               if a3.get("proof_wall_total_s") and b1.get("proof_wall_total_s") else None)
compile_old = (round(v040_prime["wall_clock_total_s"] - v040_replay["wall_clock_total_s"], 1)
               if v040_prime.get("wall_clock_total_s") and v040_replay.get("wall_clock_total_s") else None)
reg["compile_s_cold_minus_warm"] = dict(v041=compile_new, v040=compile_old, v041_basis="A3 - B1 (72 h)",
                                        v040_basis="v0.4.0 prime - replay (3 h, r-pkg040)",
                                        ratio=round(compile_new / compile_old, 4) if compile_new and compile_old else None,
                                        regression_gt_10pct=bool(compile_new and compile_old and compile_new / compile_old > 1.10))
def d6_of(case):
    out = {}
    for dom in ("d01", "d02", "d03"):
        r = load(Path(case) / f"SCORING_IO_{dom}.json")
        out[dom] = ({k: r.get(k) for k in ("verdict", "paired_frames", "bounded_misses", "nonfinite", "failures",
                                           "incomplete")} if r else "MISSING")
    return out


d6 = d6_of(opt["--d6"]) if "--d6" in opt else {}
d6_v040 = d6_of(opt["--d6old"]) if "--d6old" in opt else {}
integrity = {k: (load(v) or {}).get("pass", "MISSING") for k, v in (("v041", opt.get("--int")), ("v040", opt.get("--intold")))
             if v}
floor = load(opt["--floor"]) if "--floor" in opt else None
floor_old = load(opt["--floorold"]) if "--floorold" in opt else None
res["RW3"] = dict(b1_replay_ok=bool(replay_ok(b1)), b7_replay_ok=bool(replay_ok(b7)), metrics=reg, d6_72h=d6,
                  d6_72h_floor_A3=({k: floor[k] for k in ("raw72", "binding72_A3", "counts", "rows")} if floor else "MISSING"),
                  d6_72h_floor_A3_v040=({k: floor_old[k] for k in ("raw72", "binding72_A3", "counts", "rows")} if floor_old else "MISSING"),
                  d6_72h_v040_reference=d6_v040, output_integrity=integrity,
                  regression=any(v.get("regression_gt_10pct") for v in reg.values()),
                  d6_verdict=("PASS" if d6 and all(isinstance(v, dict) and v["verdict"] == "PASS" for v in d6.values())
                              else ("FAIL" if any(isinstance(v, dict) and v["verdict"] == "FAIL" for v in d6.values())
                                    else "INCOMPLETE")))
if "--e41" in opt:
    g = load(opt["--e41"]) or {}
    res["E41"] = {k: g.get(k) for k in ("e41_gate", "hard_stack_gate", "conservative_flags")} | {
        "coverage": (g.get("coverage") or {}).get("status"), "file": opt["--e41"]}
OUT.write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else None)
                  for k, v in res.items() if k.startswith("RW")} | {"RW3_regression": res["RW3"]["regression"],
                  "RW3_d6_raw": res["RW3"]["d6_verdict"],
                  "RW3_d6_binding_A3": (floor or {}).get("binding72_A3", "MISSING"),
                  "RW3_integrity": res["RW3"]["output_integrity"]}))
