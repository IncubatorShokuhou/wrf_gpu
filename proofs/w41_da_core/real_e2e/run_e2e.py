"""Real-data CPU end-to-end of `gpuwrf da` on the guarded A1 development case 0227 (analysis 2026-02-28 12Z, d01).

Ensemble: static same-phase archive (12Z frames of Jan-Feb WN3 runs within 45 days, causal; NO V5 confirmatory
run), innovation-calibrated, centred on the control (original CPU-WRF wg_20260227 frame at 2026-02-28 12Z).
Observations: the guarded Guimar IGRA2 extract (extract_sonde.py). Governance: the 'alisios' plug-in (guard
scan, DENY drop before open, admit, read spy). Then the written analysis wrfinput is turned into a d01 cycle
case from the original run's forcing (guarded) and admitted by the port's preflights (`gpuwrf run --dry-run`).

  JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src taskset -c 24,25,28,29 nice -n 19 \
      python proofs/w41_da_core/real_e2e/run_e2e.py WORK_DIR SONDE_EXTRACT_DIR
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time

RUN = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
T_A = "2026-02-28T12:00:00Z"
STAMP = "2026-02-28_12:00:00"



def _governance(g):
    """Admitted read set + spy counts for either backend of the 'alisios' plug-in."""
    session = g["alisios_guard"]
    out = {"plugin": g["plugin"], "backend": g["backend"]["kind"], "dropped": g["dropped"],
           "analysis_phase_observed": len(session["spy"]["observed"]),
           "analysis_phase_refused": session["spy"]["refused_paths"]}
    if g["backend"]["kind"] == "guarded_read":
        files = g["backend"]["files"]
        out.update(facade_commit=g["backend"]["facade"]["commit"], guard_sha256=g["backend"]["guard"]["sha256"],
                   staged_files=len(files), counters=g["backend"]["counters"], stage_removed=g["backend"]["stage_removed"],
                   issue_days=sorted({d for f in files.values() for d in f["issue_days"]}),
                   valid_days=sorted({d for f in files.values() for d in f["valid_days"]}),
                   read_spy=g["backend"]["read_spy"])
    else:
        out.update(fallback_reason=g["backend"]["fallback_reason"], guard_result=session["guard_result"],
                   issue_days=session["issue_days"], valid_days=session["valid_days"])
    return out

def main(work, sonde_dir):
    work, sonde_dir = Path(work), Path(sonde_dir)
    work.mkdir(parents=True, exist_ok=False)
    extract = next(sonde_dir.glob("igra2_*_admitted.txt"))
    config = {
        "schema": "gpuwrf.da.v1",
        "description": "w41-da-core real-data CPU e2e: A1 dev case 0227, analysis 2026-02-28 12Z, Guimar sonde",
        "t_a": T_A, "template_wrfinput": str(RUN / "wrfinput_d01"), "control_file": str(RUN / f"wrfout_d01_{STAMP}"),
        "ensemble": {"kind": "same_phase", "runs_glob": "<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260[12]*_18z_a*/run/run",
                     "max_age_days": 45, "max_members": 12, "scale": "innovation"},
        "obs": {"sources": [{"format": "igra2", "path": str(extract), "station": "SPM00060018"}]},
        "params": {"workers": 4},
        "governance": {"plugin": "alisios", "options": {"purpose": "w41-da-core real-data CPU e2e (dev case 0227)"}},
        "experiments": [{"name": "sonde", "assimilate": ["sonde"], "write_mean": True, "land": "cycled",
                         "profile_site": {"lat": 28.3181, "lon": -16.3822}}],
    }
    cfg_path = work / "da_0228_12z.json"
    cfg_path.write_text(json.dumps(config, indent=1) + "\n")
    from gpuwrf.cli import main as cli
    t0 = time.perf_counter()
    rc_check = cli(["da", "check", "--config", str(cfg_path)])
    t1 = time.perf_counter()
    rc = cli(["da", "analyse", "--config", str(cfg_path), "--out", str(work / "analysis")])
    t2 = time.perf_counter()
    summary = json.loads((work / "analysis" / "summary.json").read_text())
    receipt = json.loads((work / "analysis" / "receipt_sonde.json").read_text())

    # --- admission of the written analysis as a d01 cycle case (original run forcing, guarded reads) ---------------
    from gpuwrf.assimilation.letkf import alisios_guard, governance
    from gpuwrf.assimilation.letkf.cycle_case import write_cycle_case
    policy = governance.AlisiosPolicy({"receipt_dir": str(work / "case_guard")})
    days = policy._run_days(RUN, "20260227", work / "case_guard" / "metadata")
    names = ("namelist.input", "wrfbdy_d01", "wrflowinp_d01", "wrfinput_d01", "freezeH2O.dat", "qr_acr_qg_V4.dat",
             "qr_acr_qsV2.dat")
    allowed = {str(RUN / n): days for n in names if (RUN / n).exists()}
    with alisios_guard.guarded_session(allowed, ["20260227"], work / "case_guard" / "session",
                                       purpose="w41-da-core e2e: cycle case from the original run forcing") as s:
        case = write_cycle_case(RUN, {"d01": work / "analysis" / "wrfinput_d01"}, STAMP, 12, work / "case")
    t3 = time.perf_counter()
    exp = summary["experiments"]["sonde"]
    out = {"schema": "w41_da_core.real_e2e.v1", "config": str(cfg_path), "config_sha256": summary["config_sha256"],
           "rc_check": rc_check, "rc_analyse": rc, "admissible": exp["admissible"],
           "refusal_reasons": exp["refusal_reasons"], "analysis_wrfinput": exp["analysis_wrfinput"],
           "n_members": receipt["n_members"], "assimilated": receipt["assimilated"], "qc": receipt["qc"],
           "calibration": receipt["calibration"], "stats_assimilated": receipt["stats_assimilated"],
           "moisture": receipt["moisture"], "mean_moisture": receipt.get("mean_moisture"),
           "local_obs": receipt["local_obs"], "analysis_grid": receipt["analysis_grid"],
           "timings_s": {"check_cli": t1 - t0, "analyse_cli": t2 - t1, "analysis_phases": receipt["timings_s"],
                         "load_ensemble": summary["load_ensemble_s"], "case_build_and_admission": t3 - t2},
           "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
           "governance": _governance(summary["governance"]),
           "case": {"output_case": case["output_case"],
                    "admission": {k: (v.get("status") if isinstance(v, dict) else v)
                                  for k, v in case["admission"].items() if k != "wrfbdy_coverage"},
                    "guard": s.receipt["guard_result"]},
           "profile_site": receipt.get("profile", {}).get("j"), "values_logged": "statistics only"}
    (work / "E2E_RECEIPT.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: out[k] for k in ("rc_check", "rc_analyse", "admissible", "n_members", "assimilated",
                                          "timings_s", "peak_rss_mb", "case")}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
