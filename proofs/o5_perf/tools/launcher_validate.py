"""E225 launcher check: the REAL product run-config validation for one exact launcher invocation (CPU only).

usage: JAX_PLATFORMS=cpu PYTHONPATH=<src>/src python3 launcher_validate.py OUT.json -- run --input-dir CASE ... (exact argv)

1. Runs `python3 -m gpuwrf.cli <argv> --dry-run` as a subprocess with the caller's environment (the bundle/snapshot src on
   PYTHONPATH, JAX_PLATFORMS=cpu). This is the product CLI path incl. its wrfbdy coverage / scheme / land preflights; no
   hook is replaced. rc and the printed plan are recorded.
2. Explicit alignments with the SAME source tree's planning functions (nested_pipeline._dt_by_domain,
   operational_mode._steps_for_hours, nested_pipeline._history_alarm_steps): per domain dt (product parent_grid_ratio
   chain == WRF parent_time_step_ratio chain), integer own steps for --hours, last history alarm == last step, wrfbdy
   interval_seconds and auxinput4_interval (when sst_update=1) multiples of the root dt.
A FAIL in either part exits 1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

assert os.environ.get("JAX_PLATFORMS") == "cpu"


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def main() -> int:
    out = Path(sys.argv[1]); argv = sys.argv[sys.argv.index("--") + 1:]
    assert argv[0] == "run", argv
    rec = {"argv": argv, "pythonpath": os.environ.get("PYTHONPATH"), "checks": {}}
    proc = subprocess.run([sys.executable, "-m", "gpuwrf.cli", *argv, "--dry-run"], capture_output=True, text=True,
                          timeout=600)
    rec["dry_run_rc"] = proc.returncode
    plan_text = proc.stdout[proc.stdout.find("{"):] if "{" in proc.stdout else ""
    try:
        rec["dry_run_plan"] = json.loads(plan_text)
    except ValueError:
        rec["dry_run_stdout_tail"] = proc.stdout[-1500:]
    rec["dry_run_stderr_tail"] = proc.stderr[-1500:]

    from gpuwrf.io.gen2_accessor import parse_namelist
    from gpuwrf.integration import nested_pipeline as npl
    from gpuwrf.runtime.operational_mode import _steps_for_hours
    case = Path(_opt(argv, "--input-dir"))
    nml = parse_namelist(case / "namelist.input")
    dom, tc, phys = nml["domains"], nml["time_control"], nml.get("physics", {})
    hours = float(_opt(argv, "--hours"))
    max_dom = int(_opt(argv, "--max-dom") or dom["max_dom"]) if "--domains-from-namelist" not in argv else int(dom["max_dom"])
    names = npl.domain_names_for(max_dom)
    lst = lambda v, i: (v[i] if i < len(v) else v[-1]) if isinstance(v, (list, tuple)) else v
    run = SimpleNamespace(namelist=nml, grid=lambda n: SimpleNamespace(
        parent_id=lst(dom["parent_id"], int(n[1:]) - 1), parent_grid_ratio=lst(dom["parent_grid_ratio"], int(n[1:]) - 1)))
    dt = npl._dt_by_domain(run, names)
    root_dt = dt[names[0]]
    hist = npl._history_interval_minutes_by_domain(run, names)
    ok = True
    for i, n in enumerate(names):
        c = {"dt_s": dt[n]}
        if i:
            parent = f"d{int(lst(dom['parent_id'], i)):02d}"
            c["wrf_dt_s"] = dt[parent] / float(lst(dom["parent_time_step_ratio"], i))
            c["dt_matches_wrf_parent_time_step_ratio"] = c["wrf_dt_s"] == dt[n]
            ok &= c["dt_matches_wrf_parent_time_step_ratio"]
        try:
            steps = _steps_for_hours(hours, dt[n]); c["own_steps"] = steps
            alarms, integral = npl._history_alarm_steps(total_steps=steps, dt_s=dt[n], interval_minutes=hist[n])
            c.update(history_interval_min=hist[n], history_alarms=len(alarms), history_cadence_integral=integral,
                     last_alarm_is_last_step=bool(alarms) and alarms[-1] == steps)
            ok &= c["last_alarm_is_last_step"]
        except (ValueError, AssertionError) as exc:
            c["error"] = f"{type(exc).__name__}: {exc}"; ok = False
        rec["checks"][n] = c
    isec = float(tc.get("interval_seconds", 0) or 0)
    rec["checks"]["wrfbdy_interval_s"] = {"value": isec, "multiple_of_root_dt": isec > 0 and isec % root_dt == 0}
    ok &= rec["checks"]["wrfbdy_interval_s"]["multiple_of_root_dt"]
    if int(lst(phys.get("sst_update", 0), 0) or 0) == 1:
        a4 = float(lst(tc.get("auxinput4_interval", 0), 0) or 0)
        rec["checks"]["auxinput4_interval_min"] = {"value": a4, "multiple_of_root_dt": a4 > 0 and (a4 * 60) % root_dt == 0}
        ok &= rec["checks"]["auxinput4_interval_min"]["multiple_of_root_dt"]
    plan = rec.get("dry_run_plan", {})
    rec["checks"]["plan_matches_checked_run"] = {
        "effective_hours": plan.get("effective_hours"), "effective_max_dom": plan.get("effective_max_dom"),
        "ok": plan.get("effective_hours") is not None and float(plan["effective_hours"]) == hours
        and int(plan.get("effective_max_dom", -1)) == max_dom}
    ok &= rec["checks"]["plan_matches_checked_run"]["ok"]
    rec["root_steps_integer"] = "error" not in rec["checks"][names[0]]
    rec["alignment_pass"] = bool(ok)
    rec["verdict"] = "PASS" if (ok and proc.returncode == 0) else "FAIL"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(rec["verdict"], "dry_run_rc", proc.returncode, "alignment", ok, case.name, "hours", hours,
          {n: rec["checks"][n].get("own_steps") for n in names})
    return 0 if rec["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
