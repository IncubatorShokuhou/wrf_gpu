"""w43-release: compile the v0.4.3 RW1-RW4 real-world gate table from the session receipts (CPU only, reads files).

usage: python3 verdict.py S_DIR P_DIR OUT_JSON [--rw3t-root DIR] [--rw3t-verdict JSON] [--d6 SCORE_DIR] [--floor JSON]
                          [--e41 WAVE_GATE_JSON] [--aotkey DIR] [--ref042 RW_VERDICT_v042.json] [--disk RW3T_DISK.json]
                          [--icloud0-score DIR --icloud0-e41 WAVE_GATE_JSON --icloud0-tools proofs/w42_icloud0 DIR]
icloud=0 (w42-icloud0 BUNDLE_ARM_SPEC.md acceptance): D6_d0{1,2,3}.json PASS (frozen 9df4fab8, 25 paired frames, 0 failures,
0 non-finite, coverage complete), ICLOUD0_report cldfra_zero_both_arms_all_forecast_frames on all 3 domains, E41 hard stack +
coverage PASS without conservative flags (e41_gate disclosed; slim AOT blobs give INCOMPLETE as on v0.4.2), icloud0 key group
PASS (one key per program over A4 + B8, sealed) -> QUALIFIED / NOT_QUALIFIED / INCOMPLETE (missing evidence).
RW3 timing (manager 2026-10-19T05:59:13Z): PRIMARY = stepping median (d03 hourly publication cadence, leads 2..H);
whole-run = disclosed secondary, an arm with arm-window <DATA_ROOT> await > 50 ms is whole-run DISTURBED (not a regression).
Copy of the v0.4.2 w41-bundle041 verdict.py (arm parsing unchanged). Changes: v0.4.3 arm list incl. optional icloud=0 arms
(A4 prime, B8 cross-date replay: own key group); RW3 = (a) the pre-registered interleaved 24 h N/O verdict (rw3t_verdict.py),
(b) the 72 h pair B1 (N72, phase 1) vs O72 (v0.4.2 bundle, phase 2; if absent: v0.4.2 release-day receipt, labelled
cross-session), (c) compile s = A3 - B1 vs v0.4.2 612.2 s, (d) VRAM/RSS, (e) D6 72 h from the NEW scorer (scripts/wn3_score.py
summary.json, sequential domains) + frozen-A3 floor reading (floor72_new.py) + output integrity from the same summary.
Missing evidence is reported as MISSING/INCOMPLETE, never as PASS.
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
        d03 = sorted(v for k, v in mt.items() if k.startswith("wrfout_d03_"))
        if len(d03) > 2:   # primary RW3 metric (manager 05:59:13Z): d03 hourly publication cadence median, leads 2..H
            rec["stepping_median_d03_s_per_h"] = round(statistics.median(b - a for a, b in zip(d03[1:], d03[2:])), 3)
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



ICLOUD = [n for n in ("A4_wn3_0227_icloud0", "B8_wn3_0502_icloud0_3h") if (R / n).is_dir()]
ARMS = ["A1_swiss", "A2_prod", "A3_wn3_0227_72h", "B2_wn3_0502_3h", "B4_wn3_0227in48h_3h", "B5_swiss_3h", "B6_prod_3h",
        "B3_wn3_0614_48h", *ICLOUD]
arms = {n: arm(n) for n in ARMS}
RT = Path(opt.get("--rw3t-root", str(S.parent / "rw3t")))
R_SAVE, R = R, RT / "runs"
for n in ("N72", "O72"):
    arms[n] = arm(n)
R = R_SAVE
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
for warm, cold in (("N72", "A3_wn3_0227_72h"), ("B5_swiss_3h", "A1_swiss"), ("B6_prod_3h", "A2_prod")):
    rw1[warm] = dict(replay_ok=bool(replay_ok(arms[warm])), cold=cold, **bytes_equal(warm, cold),
                     startup_lead1_s=arms[warm].get("d01_lead1_s"), process_wall_s=arms[warm].get("process_wall_s"))
rw1["verdict"] = "PASS" if all(v["replay_ok"] and v["identical"] for v in rw1.values()) else "FAIL_OR_INCOMPLETE"
res["RW1"] = rw1

manifest = load(P / "sealed/BUNDLE_MANIFEST.json") or {}
sealed_keys = {re.sub(r".*/(\w+)/k_([0-9a-f]{64})\.xlaexec$", r"\1:\2", f) for f in (manifest.get("files") or {})
               if re.search(r"/k_[0-9a-f]{64}\.xlaexec$", f)}


def key_group(names):
    keys = {}
    for n in names:
        for x in arms[n].get("aot") or []:
            if x["key"] and len(x["key"]) == 64:  # capture lines carry no/abbreviated keys; the lookup line has the full one
                keys.setdefault(x["domain"], {})[n] = x["key"]
    one = {dom: (len({k for k in v.values() if k}) == 1 and all(v.get(n) for n in names)) for dom, v in keys.items()}
    sealed = {dom: all(f"{dom.replace('fused/', 'fused_')}:{k}" in sealed_keys for k in set(v.values()) if k)
              for dom, v in keys.items()}
    return keys, one, sealed


wn3 = ["A3_wn3_0227_72h", "N72", "B2_wn3_0502_3h", "B3_wn3_0614_48h", "B4_wn3_0227in48h_3h"]
keys, one_key, in_bundle = key_group(wn3)
rw2 = {n: dict(replay_ok=bool(replay_ok(arms[n])), aot=arms[n].get("aot"), consumer=arms[n].get("consumer"))
       for n in ("B2_wn3_0502_3h", "B3_wn3_0614_48h", "B4_wn3_0227in48h_3h", *[x for x in ICLOUD if x.startswith("B")])}
aotkey = {}
for f in sorted(Path(opt.get("--aotkey", "/nonexistent")).glob("*.json")):
    rep = load(f) or {}
    aotkey[f.stem] = {name: row.get("shared") for name, row in (rep.get("comparison") or {}).items()}
ic = {}
if ICLOUD:
    ik, ione, isealed = key_group(ICLOUD)
    ic = dict(arms=ICLOUD, gpu_keys_by_program=ik, one_key_per_program=ione, keys_are_sealed_blobs=isealed,
              differs_from_default={dom: (set(v.values()) != set(keys.get(dom, {}).values())) for dom, v in ik.items()},
              verdict="PASS" if (ik and all(ione.values()) and all(isealed.values())
                                 and all(replay_ok(arms[n]) for n in ICLOUD if n.startswith("B"))) else "FAIL_OR_INCOMPLETE")
res["RW2_RW4"] = dict(arms=rw2, gpu_keys_by_program=keys, one_key_per_program=one_key,
                      keys_are_sealed_blobs=in_bundle, cpu_aot_key_pairs=aotkey, icloud0=ic or "NOT_IN_BUNDLE",
                      verdict="PASS" if (all(v["replay_ok"] for v in rw2.values()) and keys and all(one_key.values())
                                         and all(in_bundle.values()) and aotkey
                                         and all(all(x) for x in aotkey.values() if x)
                                         and (not ic or ic["verdict"] == "PASS")) else "FAIL_OR_INCOMPLETE")

b1, a3 = arms["N72"], arms["A3_wn3_0227_72h"]
o72 = arms.get("O72")
ref042 = load(opt.get("--ref042", "")) or {}
r42 = (ref042.get("arms") or {}).get("N72") or {}   # v0.4.2 RW_VERDICT: N72 = v0.4.2 72 h warm replay


def per_h(a, hours=72):
    return round(a["last_file_s"] / hours, 3) if a and a.get("last_file_s") else None


old = o72 if (o72 and o72.get("rc") == "0") else None
basis = "O72 (v0.4.2 sealed bundle, same session, watched)" if old else "v0.4.2 release-day B1 receipt (CROSS-SESSION, disclosed)"
src_old = old or r42
disk = (load(opt["--disk"]) or {}).get("arms", {}) if "--disk" in opt else {}


def disturbed(name):
    v = disk.get(name) or {}
    return str(v.get("disposition", "")).startswith("DISCLOSED") or ((v.get("nvme3n1") or {}).get("await_ms") or 0) > 50


metrics = {
    "stepping_median_d03_s_per_h (PRIMARY)": (b1.get("stepping_median_d03_s_per_h"), src_old.get("stepping_median_d03_s_per_h")),
    "whole_run_s_per_fch": (per_h(b1), per_h(src_old)),
    "stepping_median_s_per_h": (b1.get("stepping_median_s_per_h"), src_old.get("stepping_median_s_per_h")),
    "startup_lead1_s": (b1.get("d01_lead1_s"), src_old.get("d01_lead1_s")),
    "vram_peak_pid_mib": (b1.get("vram_peak_pid_mib"), src_old.get("vram_peak_pid_mib")),
    "host_max_rss_gib": (b1.get("max_rss_gib"), src_old.get("max_rss_gib")),
}
reg = {}
whole_disturbed = [n for n in ("N72", "O72") if disturbed(n)]
for k, (new, prev) in metrics.items():
    reg[k] = dict(v043=new, v042=prev, v042_basis=basis, ratio=round(new / prev, 4) if new and prev else None,
                  regression_gt_10pct=bool(new and prev and new / prev > 1.10))
    if k == "whole_run_s_per_fch":   # disclosed secondary; DISTURBED (arm-window <DATA_ROOT> await > 50 ms) is not a regression
        reg[k].update(disclosed_secondary=True, disturbed_arms=whole_disturbed,
                      regression_gt_10pct=bool(reg[k]["regression_gt_10pct"] and not whole_disturbed))
compile_new = (round(a3["proof_wall_total_s"] - b1["proof_wall_total_s"], 1)
               if a3.get("proof_wall_total_s") and b1.get("proof_wall_total_s") else None)
compile_old = 612.2   # v0.4.2 RW_VERDICT (A3 - N72)
reg["compile_s_cold_minus_warm"] = dict(v043=compile_new, v042=compile_old, basis="A3 - warm 72 h proof wall (v0.4.3: N72, v0.4.2: B1)",
                                        ratio=round(compile_new / compile_old, 4) if compile_new else None,
                                        regression_gt_10pct=bool(compile_new and compile_new / compile_old > 1.10))
rw3t = load(opt["--rw3t-verdict"]) if "--rw3t-verdict" in opt else None
d6 = load(Path(opt["--d6"]) / "summary.json") if "--d6" in opt else None
floor = load(opt["--floor"]) if "--floor" in opt else None
d6_dom = ({dom: dict(pass_=v.get("pass"), policy_failures=len(v.get("policy_failures") or []),
                     paired=(v.get("pairing") or {}).get("paired_file_count"),
                     integrity=(v.get("output_integrity") or {}).get("pass"))
           for dom, v in (d6.get("domains") or {}).items()} if d6 else "MISSING")
res["RW3"] = dict(b1_replay_ok=bool(replay_ok(b1)), o72_replay_ok=bool(old and replay_ok(old)), metrics_72h=reg,
                  rw3t_interleaved_24h=({k: rw3t.get(k) for k in ("verdict", "pooled_ratio", "spread", "counted", "rule")
                                         if k in rw3t} if rw3t else "MISSING"),
                  d6_72h_new_scorer=d6_dom, d6_72h_raw=("PASS" if d6 and d6.get("pass") else ("FAIL" if d6 else "MISSING")),
                  d6_72h_floor_A3=({k: floor[k] for k in ("raw72", "binding72_A3", "counts", "rows", "other_policy_failures")}
                                   if floor else "MISSING"),
                  output_integrity=(d6.get("output_integrity_pass") if d6 else "MISSING"),
                  regression_72h=any(v.get("regression_gt_10pct") for v in reg.values()))
if "--e41" in opt:
    g = load(opt["--e41"]) or {}
    res["E41"] = {k: g.get(k) for k in ("e41_gate", "hard_stack_gate", "conservative_flags")} | {
        "coverage": (g.get("coverage") or {}).get("status"), "file": opt["--e41"]}
if "--icloud0-score" in opt:  # icloud=0 qualification: acceptance of w42-icloud0 BUNDLE_ARM_SPEC.md (manager 06:05:46Z)
    sc, frozen = Path(opt["--icloud0-score"]), "9df4fab8514523286f07b127eb8bf57a7545459c4448fef721ac44c0ecc351dc"
    d6i = {d: load(sc / f"D6_{d}.json") for d in ("d01", "d02", "d03")}
    rep = (load(sc / "ICLOUD0_report.json") or {}).get("domains") or {}
    summ, gi = load(sc / "D6_SUMMARY.json"), load(opt.get("--icloud0-e41", "")) or {}
    morning = load(Path(opt.get("--icloud0-tools", "/nonexistent")) / "fixtures/d6_summary_morning.json") or {}
    d6_ok = {d: bool(v) and v.get("verdict") == "PASS" and v.get("paired_frames") == 25 and not v.get("failures")
             and not v.get("nonfinite") and not v.get("incomplete") and v.get("coverage_complete") is True
             and v.get("manifest_sha256") == frozen for d, v in d6i.items()}
    cld_ok = {d: (rep.get(d) or {}).get("cldfra_zero_both_arms_all_forecast_frames") is True for d in d6i}
    e41_ok = gi.get("hard_stack_gate") == "PASS" and (gi.get("coverage") or {}).get("status") == "PASS" \
        and gi.get("conservative_flags") is False
    vs_morning = {d: {f: dict(now=(r or {}).get("rmse"), morning=(((morning.get(d) or {}).get("worst_lead_rmse") or {}).get(f) or {}).get("rmse"))
                      for f, r in ((summ or {}).get(d, {}).get("worst_lead_rmse") or {}).items()} for d in d6i}
    missing = [n for n, v in (("D6", all(d6i.values())), ("report", bool(rep)), ("summary", bool(summ)), ("E41", bool(gi)),
                              ("keys", bool(ic))) if not v]
    ok = all(d6_ok.values()) and all(cld_ok.values()) and e41_ok and bool(ic) and ic.get("verdict") == "PASS"
    res["ICLOUD0"] = dict(score_dir=str(sc), d6_pass=d6_ok, cldfra_zero_both_arms=cld_ok,
                          e41=dict(e41_gate=gi.get("e41_gate"), hard_stack_gate=gi.get("hard_stack_gate"),
                                   conservative_flags=gi.get("conservative_flags"), coverage=(gi.get("coverage") or {}).get("status")),
                          keys=ic.get("verdict") if ic else "MISSING", worst_lead_rmse_vs_morning=vs_morning, missing=missing,
                          verdict="QUALIFIED" if ok else ("INCOMPLETE" if missing else "NOT_QUALIFIED"))
OUT.write_text(json.dumps(res, indent=1, default=str) + "\n")
print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) and "verdict" in v else None)
                  for k, v in res.items() if k.startswith("RW")} | {
                  "RW3_regression_72h": res["RW3"]["regression_72h"], "RW3_d6_raw": res["RW3"]["d6_72h_raw"],
                  "RW3_d6_binding_A3": (floor or {}).get("binding72_A3", "MISSING"),
                  "RW3_integrity": res["RW3"]["output_integrity"],
                  "RW3T": (rw3t or {}).get("verdict", "MISSING"),
                  "ICLOUD0": (res.get("ICLOUD0") or {}).get("verdict", "NOT_IN_BUNDLE")}))
