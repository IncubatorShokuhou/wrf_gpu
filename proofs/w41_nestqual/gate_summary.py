"""Nest-gate summary of one G2 arm (lane w41-nestqual): reads the merged bounded WN3 scorer output (main
scripts/wn3_score.py --sequential-domains: <score_dir>/summary.json + d0{1,2,3}.json) of a case plus PREPARE.json /
GPU_STAGE.json / cpu_run.json and writes <case>/NEST_GATE.json.
Gate (contract): D6 24 h PASS on d01-d03 vs the original CPU-WRF twin = every domain 'pass' of the scorer (frozen
manifest, 0 bounded misses, 0 non-finite in either arm, every CPU variable present and complete on all 25 frames) AND
input/output identity of the GPU process (E184: staged input SHAs, live namelist, rd-s6 output manifest) AND GPU arm
rc 0. output_integrity (GPU field constant where CPU varies, WRF globals) is reported next to the verdict and disclosed
(never hidden); unbounded fields are REPORT_ONLY. Reads only wrf_gpu2 files (no ALISIOS root).
usage: python gate_summary.py <case_dir> <score_dir>
"""
import hashlib
import json
import sys
from pathlib import Path

SCHEMES = {"C2": "YSU bl_pbl_physics=1 + revised-MM5 sf_sfclay_physics=1",
           "C3": "Shin-Hong bl_pbl_physics=11 + sf_sfclay_physics=1",
           "C4": "ACM2 bl_pbl_physics=7 + sf_sfclay_physics=1",
           "C7": "EEPS bl_pbl_physics=16 + sf_sfclay_physics=1"}
NONFINITE = {"nonfinite_gpu", "nonfinite_cpu"}
COVERAGE = {"missing_cpu_field", "incompatible_common", "unscored_numeric_field", "incomplete_field", "time_metadata"}
case, score = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
prep = json.loads((case / "PREPARE.json").read_text())
stage = json.loads((case / "GPU_STAGE.json").read_text())
cpu = json.loads((case / "cpu_run.json").read_text())
summary = json.loads((score / "summary.json").read_text()) if (score / "summary.json").exists() else {"domains": {}}
domains, ok = {}, True
for d in ("d01", "d02", "d03"):
    s, path = summary["domains"].get(d), score / f"{d}.json"
    if s is None or not path.exists():
        domains[d] = dict(verdict="MISSING")
        ok = False
        continue
    r = json.loads(path.read_text())
    worst, bounded = {}, 0
    for n, f in r["field_summaries"].items():
        leads = [x for x in f.get("by_lead", []) if x.get("rmse") is not None
                 and x["tolerance_result"].get("spec", {}).get("rmse")]
        if not any(x["tolerance_result"].get("pass") is not None for x in f.get("by_lead", [])):
            continue
        bounded += 1
        if leads:
            w = max(leads, key=lambda x: x["rmse"] / x["tolerance_result"]["spec"]["rmse"])
            lim = w["tolerance_result"]["spec"]["rmse"]
            worst[n] = dict(rmse=w["rmse"], lead_h=w["lead_h"], limit=lim, ratio=w["rmse"] / lim)
    pf, integ = s["policy_failures"], s["output_integrity"]
    kinds = sorted({f["kind"] for f in pf})
    domains[d] = dict(verdict="PASS" if s["pass"] else "FAIL", paired_frames=s["pairing"]["paired_file_count"],
                      compared_fields=s["summaries"]["comparable_field_count"], bounded_fields=bounded,
                      failures=len(pf), failure_kinds=kinds, failure_examples=pf[:12],
                      nonfinite=sum(f["kind"] in NONFINITE for f in pf) + len(s["gpu_only_nonfinite"]),
                      coverage_complete=not (COVERAGE & set(kinds)) and not s["missing_required_fields"],
                      missing_vs_cpu=sorted(integ["missing_variables_vs_cpu"]),
                      d6_tolerance_failures=s["summaries"]["tolerance_failure_count"],
                      worst_lead_rmse=worst,
                      output_integrity_pass=integ["pass"],
                      report_only_constant_vs_varying=integ["degenerate_fields"],   # GPU constant where CPU varies
                      global_attrs_missing=dict(vs_cpu=integ["global_attrs"]["missing_vs_cpu"],
                                                required=integ["global_attrs"]["missing_required"]),
                      d6_json_sha256=sha(path))
    ok &= bool(s["pass"])
ident = stage["input_identity"]
identity = ident["ok"] and all(ident["staged_input_sha256"][f] == prep["inputs"][f]["sha256"] for f in prep["inputs"])
identity &= ident["gpu_namelist_sha256"] == prep["gpu_arm_namelist_sha256"]
verdict = "PASS" if ok and identity and stage["gpu_arm_rc"] == "0" else (
    "FAIL" if any(v.get("verdict") == "FAIL" for v in domains.values()) else "INCOMPLETE")
out = dict(schema="w41-nestqual.nest_gate.v2", case=case.name, issue=prep["issue"], arm=prep["arm"],
           scheme=SCHEMES.get(prep["arm"]), verdict=verdict,
           gate="D6 24 h d01-d03 vs original CPU-WRF twin, merged bounded scorer scripts/wn3_score.py, frozen manifest",
           scorer=dict(score_dir=str(score), summary_sha256=sha(score / "summary.json") if (score / "summary.json").exists()
                       else None, manifest=summary.get("manifest"), twin_thresholds=summary.get("twin_thresholds"),
                       output_integrity_pass=summary.get("output_integrity_pass")),
           domains=domains, input_identity=identity,
           gpu=dict(run_dir=stage["gpu_run_dir"], arm_rc=stage["gpu_arm_rc"], aot_loaded_true=stage["aot_loaded_true"],
                    aot_fallback_missing=stage["aot_fallback_missing"], engine=stage.get("engine"),
                    output_manifest=ident.get("output_manifest")),
           cpu=dict(executable=cpu["executable"], timing=cpu.get("timing"), output_manifest_sha256=cpu["output_manifest_sha256"]),
           twin_namelist_diff_vs_gpu_arm=prep["twin_namelist_diff_vs_arm"],
           note="GPU-vs-GPU never fidelity evidence; unbounded fields REPORT_ONLY; output_integrity (constant-vs-varying "
                "fields, WRF globals) reported per domain and disclosed, the verdict is the contract's D6 gate")
(case / "NEST_GATE.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(dict(arm=out["arm"], verdict=verdict, identity=identity,
                      domains={d: v.get("verdict") for d, v in domains.items()},
                      integrity={d: v.get("output_integrity_pass") for d, v in domains.items()})))
