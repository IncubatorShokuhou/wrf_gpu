"""Deletion-sensitivity mutants for o2-rootbdy: apply one source edit, run the fast new tests, restore."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path("<USER_HOME>/src/wrf_gpu2_wt/o2-rootbdy")
M = {
 "M1_drop_mp40_route": ("src/gpuwrf/runtime/operational_mode.py", '    40: ("Ns", "Ng", "Nc"),         # morr_tm_aero', '    # 40 dropped'),
 "M2_drop_aerosol_branch": ("src/gpuwrf/coupling/boundary_apply.py", "if name in flow_only and name in _ROOT_FLOW_AEROSOL_NUMBERS:", "if False:"),
 "M3_drop_mp28_Nc_transport": ("src/gpuwrf/runtime/operational_mode.py", 'return _MOISTURE_SPECIES + ("nwfa", "nifa", "Nc")', 'return _MOISTURE_SPECIES + ("nwfa", "nifa")'),
 "M4_drop_cli_nested_refusal": ("src/gpuwrf/cli.py", "        refuse_single_domain_mp_nested(namelist, max_dom)\n", "        pass\n"),
 "M5_drop_mp24_island": ("src/gpuwrf/coupling/scan_adapters.py", "(1, 2, 3, 4, 6, 10, 13, 14, 16, 24, 97)", "(1, 2, 3, 4, 6, 10, 13, 14, 16, 97)"),
}
res = {}
for name, (rel, old, new) in M.items():
    p = ROOT / rel; src = p.read_text(); assert src.count(old) == 1, name
    p.write_text(src.replace(old, new))
    try:
        sel = ["tests/v040/rootbdy/test_root_flow_extras.py", "-k", "not chain and not missing and not swiss"]
        if name == "M5_drop_mp24_island":
            sel = ["tests/v040/rootbdy/test_root_flow_extras.py", "-k", "island"]
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "--basetemp", "<USER_HOME>/wrf_gpu2_lanes/o2-rootbdy/pytest_mut", *sel],
                           cwd=ROOT, capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()[-3:]
        res[name] = {"rc": r.returncode, "killed": r.returncode == 1 and ("AssertionError" in r.stdout or "assert" in r.stdout or "Error" in r.stdout) and " error" not in tail[-1], "tail": tail}
    finally:
        p.write_text(src)
    print(name, res[name], flush=True)
Path("<USER_HOME>/wrf_gpu2_lanes/o2-rootbdy/mutants.json").write_text(json.dumps(res, indent=1))
