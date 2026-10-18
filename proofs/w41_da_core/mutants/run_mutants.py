"""Genuine-assertion mutant kills for the w41-da-core product code (E202).

Each mutant copies src/gpuwrf/assimilation/letkf to a temp dir, applies ONE textual change (must match exactly
once), and runs the named tests with the copy first on the package path (mutant_overlay plugin). A mutant counts as
KILLED only if >= 1 test FAILS with an AssertionError (pytest.raises-not-raised counts: 'DID NOT RAISE' is an
assertion of the test) and there are 0 collection/setup errors. Writes MUTANTS.json.

  [W41_MUTANT_JOBS=4] JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH=src python proofs/w41_da_core/mutants/run_mutants.py OUT.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "src/gpuwrf/assimilation/letkf"
T = "tests/da/"

MUTANTS = [
    ("M00 control: unmutated copy through the overlay", "core.py", None, None, [T]),
    ("M01 letkf transform without symmetric square root", "core.py",
     "xp.sqrt((n_members - 1) / lam)", "((n_members - 1) / lam)", [T + "test_letkf.py"]),
    ("M02 per-point inflation collapsed to its mean", "core.py",
     "a = a + ((n_members - 1) / rho)[:, None, None]", "a = a + ((n_members - 1) / rho.mean())",
     [T + "test_letkf.py"]),
    ("M03 Gaspari-Cohn inner coefficient", "core.py", "* a - 5.0 / 3.0) * a * a + 1.0", "* a - 4.0 / 3.0) * a * a + 1.0",
     [T + "test_letkf.py", "tests/a1_da/test_letkf_core.py"]),
    ("M04 variable localisation factor ignored", "analysis.py",
     "rho_g = rho_loc if factor is None else rho_loc * factor[idx]", "rho_g = rho_loc", [T + "test_letkf.py"]),
    ("M05 maintenance hook bypassed", "analysis.py",
     "xa = maintenance(name, xb, lgrid.transform_field(xb, weights, agrid, stagger=stagger))",
     "xa = lgrid.transform_field(xb, weights, agrid, stagger=stagger)", [T + "test_letkf.py"]),
    ("M06 water budget without dry-air mass", "moisture.py",
     "added = (np.asarray(clipped, np.float64) - np.asarray(raw, np.float64)) * mass",
     "added = (np.asarray(clipped, np.float64) - np.asarray(raw, np.float64)) + 0 * mass", [T + "test_moisture.py"]),
    ("M07 clip budget never gates", "moisture.py", "    if budget is not None:\n        values = (",
     "    if False:\n        values = (", [T + "test_moisture.py"]),
    ("M08 background negatives treated as analysis negatives", "moisture.py",
     "        negative &= ~inherited\n    counts", "        pass\n    counts", [T + "test_moisture.py"]),
    ("M09 hybrid layer mass sign/levels swapped", "moisture.py", "(pf[:, :-1] - pf[:, 1:]) / G",
     "(pf[:, 1:] - pf[:, :-1]) / G", [T + "test_moisture.py"]),
    ("M10 config accepts unknown keys", "config.py", "    if unknown:\n        _fail(where, f\"unknown key(s)",
     "    if False:\n        _fail(where, f\"unknown key(s)", [T + "test_config.py"]),
    ("M11 config accepts naive times", "config.py", "    if when.tzinfo is None:\n        _fail(",
     "    if when.tzinfo is None:\n        when = when.replace(tzinfo=timezone.utc)\n    if False:\n        _fail(",
     [T + "test_config.py"]),
    ("M12 default policy lets ALISIOS data through", "governance.py",
     "        if forbidden:\n            raise PlanError", "        if False:\n            raise PlanError",
     [T + "test_governance.py"]),
    ("M13 ALISIOS issue day not checked for archive frames", "governance.py",
     "pairs += [(e, m.group(1)) for e in archive", "pairs += [(e, m.group(1)) for e in [] ", [T + "test_governance.py"]),
    ("M14 unplanned opens not flagged", "governance.py",
     "unplanned = sorted(p for p, row in recorder.opened.items() if not row[\"planned\"])", "unplanned = []",
     [T + "test_governance.py"]),
    ("M15 CSV reader takes the latest instead of the earliest tie", "readers.py", "key = (off, when)",
     "key = (off, -when.timestamp())", [T + "test_readers.py"]),
    ("M16 ISD-Lite calm rule dropped", "readers.py", "or (direction == 0.0 and speed > 0.0)", "or False",
     [T + "test_readers.py"]),
    ("M18 water budget sees float32 storage rounding", "analysis.py",
     "else mst.water_budget(raw_qv, clipped_qv, mass, geo))", "else mst.water_budget(raw_qv, analysis[\"QVAPOR\"], mass, geo))",
     [T + "test_moisture.py"]),
    ("M19 NaN water budget passes the gate", "moisture.py",
     "        if not all(np.isfinite(v) for v in values):", "        if False:", [T + "test_moisture.py"]),
    ("M17 mean wrfinput skips the moisture gate", "state_io.py",
     "    counters[\"admissible\"] = not reasons", "    reasons = []\n    counters[\"admissible\"] = True",
     [T + "test_moisture.py"]),
    ("M20 ALISIOS admission not trimmed to max_members", "api.py",
     "    plan = governance.limit_archive(plan, cfg.data[\"ensemble\"].get(\"max_members\"))\n", "",
     [T + "test_governance.py"]),
    ("M21 facade backend: analysis reads the original ALISIOS paths (no staging)", "api.py",
     "        run_cfg, run_plan = localize(cfg, plan) if localize else (cfg, plan)\n",
     "        run_cfg, run_plan = cfg, plan\n", [T + "test_alisios_facade.py"]),
    ("M22 facade stage kept after the session", "governance.py",
     "                    shutil.rmtree(stage_root)\n", "                    pass\n", [T + "test_alisios_facade.py"]),
    ("M23 facade chosen although a role is read outside its planned path", "governance.py",
     "        roles = sorted({e.role for e in alisios} - STAGEABLE_ROLES)\n", "        roles = []\n",
     [T + "test_alisios_facade.py"]),
    ("M24 read spy records but does not refuse an unlisted ALISIOS path", "alisios_guard.py",
     "            raise _guard_call().GuardRefused(f\"a1 read spy refused an unlisted ALISIOS payload path: {p}\")\n",
     "            return\n", [T + "test_alisios_facade.py", T + "test_governance.py"]),
    # critic cr-w41-da-core FIX items 1-4 (delta)
    ("M25 template admitted under the issue day, not its start day (critic item 1)", "governance.py",
     "                days = self._coverage(e, windows[key])\n",
     "                days = [run.group(1)] if e.role == \"template\" else self._coverage(e, windows[key])\n",
     [T + "test_governance.py", T + "test_alisios_facade.py"]),
    ("M26 history file covers only its file-name day", "governance.py",
     "            return _days_between(t0, last)                         # every frame the file can hold",
     "            return _days_between(t0, t0)", [T + "test_governance.py"]),
    ("M27 only issue days are guard-scanned, not the coverage days", "governance.py",
     "pairs = [(e, d) for e in alisios for d in sorted(set(e.valid_days) | {_RUN_RE.search(e.path).group(1)})]",
     "pairs = [(e, _RUN_RE.search(e.path).group(1)) for e in alisios]",
     [T + "test_governance.py", T + "test_alisios_facade.py"]),
    ("M28 run without namelist falls back to the issue day", "governance.py",
     "            raise PlanError(f\"cannot establish the valid-day coverage of the ALISIOS run {run_dir}: no namelist.input\")",
     "            day = datetime.strptime(issue_day, \"%Y%m%d\")\n            return {\"namelist\": None, \"start\": [day], "
     "\"end\": day, \"history_interval_min\": [None], \"frames_per_outfile\": [10], \"max_dom\": 1}",
     [T + "test_governance.py"]),
    ("M29 frames_per_outfile default 1 instead of WRF's 10", "governance.py",
     "frames.append(int(per(\"frames_per_outfile\", d, 10)))", "frames.append(int(per(\"frames_per_outfile\", d, 1)))",
     [T + "test_governance.py"]),
    ("M30 analysis ignores the case boundary width (critic item 2)", "analysis.py",
     "    case = geo.get(\"SPEC_BDY_WIDTH\")\n", "    case = None\n",
     [T + "test_letkf.py"]),     # the API refuses an unknown width loudly (ConfigError, not an assertion): see M31-M33
    ("M31 explicit width contradicting the case is accepted", "analysis.py",
     "        if explicit is not None and int(explicit) != int(case):", "        if False:",
     [T + "test_letkf.py", T + "test_api_cli.py"]),
    ("M32 member writer validates a default-5 zone", "member_cycle.py",
     "    spec_bdy_width, width_source = resolve_spec_bdy_width(geo, spec_bdy_width)   # the case zone, never a default 5",
     "    spec_bdy_width, width_source = (spec_bdy_width or 5), \"default\"", [T + "test_api_cli.py"]),
    ("M33 mean writer zone check off", "api.py",
     "    if any(zone.values()):        # the shared boundary files own this zone: the mean must leave it untouched",
     "    if False:", [T + "test_api_cli.py"]),
    ("M34 row role ignored by the API selection: withheld rows calibrate (critic item 3)", "api.py",
     "        use = kinds & ~role_withheld\n", "        use = kinds\n", [T + "test_api_cli.py"]),
    ("M35 role-withheld rows lost from verification", "api.py",
     "        withhold = (np.isin(kind, [KINDS[k] for k in exp[\"withhold\"]]) | (kinds & role_withheld)) & ~use\n",
     "        withhold = np.isin(kind, [KINDS[k] for k in exp[\"withhold\"]]) & ~use\n", [T + "test_api_cli.py"]),
    ("M36 analysis ignores obs.role", "analysis.py",
     "    withheld = withheld | role_withheld | cloud\n", "    withheld = withheld | cloud\n", [T + "test_letkf.py"]),
    ("M37 analysis assimilates cloud rows", "analysis.py",
     "    withheld = withheld | role_withheld | cloud\n", "    withheld = withheld | role_withheld\n",
     [T + "test_letkf.py"]),
    ("M38 config accepts cloud assimilation", "config.py",
     "    if \"cloud\" in out[\"assimilate\"]:", "    if False:", [T + "test_api_cli.py"]),
    ("M39 supersaturation cap/to_cloud accepted again (critic item 4)", "moisture.py",
     "SUPERSATURATION_POLICIES = (\"keep\",)\nREFUSED_SUPERSATURATION = (\"cap\", \"to_cloud\")",
     "SUPERSATURATION_POLICIES = (\"keep\", \"cap\", \"to_cloud\")\nREFUSED_SUPERSATURATION = ()",
     [T + "test_moisture.py", "tests/a1_da/test_analysis.py"]),
    ("M40 rebalance accepts a changed hydrometeor", "state_io.py",
     "    if changed:   # WRF's hydrostatic recursion", "    if False:   # WRF's hydrostatic recursion",
     [T + "test_moisture.py"]),
    ("M41 namelist without spec_bdy_width treated as unknown (WRF runs its default 5)", "state_io.py",
     "    return (5 if value is None else int(value)), str(namelist), value is None",
     "    return (None if value is None else int(value)), str(namelist), value is None", [T + "test_letkf.py"]),
]


def run(mutant, workdir, out_xml):
    name, fname, old, new, tests = mutant
    target = workdir / "letkf"
    shutil.copytree(PKG, target, ignore=shutil.ignore_patterns("__pycache__"))
    text = (target / fname).read_text()
    if old is not None:
        if text.count(old) != 1:
            return {"name": name, "status": "INVALID", "reason": f"pattern occurs {text.count(old)} times"}
        (target / fname).write_text(text.replace(old, new))
    env = dict(os.environ, W41_MUTANT_DIR=str(target), PYTHONPATH=f"{ROOT / 'src'}:{Path(__file__).parent}",
               JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "mutant_overlay", "-p", "no:cacheprovider",
           "--basetemp", str(workdir / "pytest"), f"--junitxml={out_xml}", *tests]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    root = ET.parse(out_xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    failures, errors, assertion = [], [], []
    for case in suite.iter("testcase"):
        for f in case.findall("failure"):
            failures.append(case.get("name"))
            msg = (f.get("message") or "") + (f.text or "")
            if "AssertionError" in msg or "assert " in msg or "DID NOT RAISE" in msg or "Failed: DID NOT" in msg:
                assertion.append(case.get("name"))
        errors += [case.get("name") for _ in case.findall("error")]
    killed = bool(assertion) and not errors
    status = "KILLED" if killed else "SURVIVED"
    if old is None:     # unmutated control through the same overlay: everything must pass
        status = "CONTROL_PASS" if not failures and not errors and proc.returncode == 0 else "CONTROL_FAIL"
    return {"name": name, "file": fname, "tests": tests, "status": status,
            "assertion_failures": assertion[:8], "n_failures": len(failures), "errors": errors,
            "pytest_rc": proc.returncode}


def _one(mutant):
    with tempfile.TemporaryDirectory(dir=os.environ.get("W41_MUTANT_TMP")) as tmp:
        res = run(mutant, Path(tmp), Path(tmp) / "junit.xml")
    print(res["status"], res["name"], res.get("assertion_failures", res.get("reason")), flush=True)
    return res


def main(out):
    from concurrent.futures import ThreadPoolExecutor     # each mutant is its own pytest subprocess
    with ThreadPoolExecutor(max_workers=int(os.environ.get("W41_MUTANT_JOBS", "1"))) as pool:
        only = os.environ.get("W41_MUTANT_ONLY")         # e.g. "M00,M07": rerun a subset (control always first)
        chosen = [m for m in MUTANTS if not only or m[0].split()[0] in only.split(",")]
        results = list(pool.map(_one, chosen))
    control = [r["status"] for r in results if r["name"].startswith("M00")]
    summary = {"schema": "w41_da_core.mutants.v1", "control": control[0] if control else "NOT_RUN",
               "killed": sum(r["status"] == "KILLED" for r in results), "total": len(results) - len(control),
               "results": results}
    Path(out).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("control", "killed", "total")}))


if __name__ == "__main__":
    main(sys.argv[1])
