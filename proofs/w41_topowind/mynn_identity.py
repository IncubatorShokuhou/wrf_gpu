"""topo_wind with MYNN is a WRF no-op: the release program (jaxpr) must stay identical.

Same mechanism as proofs/s1_port/release_identity.py (o1 probe, native facade at trace
time, CPU only) with arms release / topo_wind=1 / topo_wind=2 on the release Swiss case
(MYNN 5/5) whose wrfinput carries the lane-geogrid VAR_SSO (GOT_VAR_SSO=1), plus the
original RD11 case (VAR_SSO=0) where topo_wind=1 must be REFUSED like WRF start_em.F:222.
usage: python mynn_identity.py <repo> <out_dir>
"""
import json, os, subprocess, sys
from pathlib import Path

REPO, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
SHIM = r'''
import sys
from pathlib import Path
root, output, case, arms = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
import os
os.environ["O1_CASE"] = case
sys.path[:0] = [str(root / "src"), str(root / "proofs/o1_nlbind")]
from gpuwrf.backends import pallas as facade
from gpuwrf.integration import nested_pipeline as pipeline
original_load = pipeline._load_domains
def load_on_cpu(*a, **k):
    facade.cuda_native = lambda: False
    try:
        return original_load(*a, **k)
    finally:
        facade.cuda_native = lambda: True
pipeline._load_domains = load_on_cpu
probe = root / "proofs/o1_nlbind/probe.py"
source = probe.read_text().replace('LANE = Path("<USER_HOME>/wrf_gpu2_lanes/o1-nlbind")',
                                   'LANE = Path("' + str(output.parent) + '")')
sys.argv = [str(probe), str(root), str(output), *arms]
exec(compile(source, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
'''
runs = {"vsso": ("<DATA_ROOT>/wrf_gpu2/scratch/w41-topowind/cases/rd11_vsso", ["release", "topo_wind=1", "topo_wind=2"]),
        "rd11": ("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu", ["topo_wind=1"])}
summary = {}
for tag, (case, arms) in runs.items():
    out = OUT / f"{tag}.json"
    subprocess.run([sys.executable, "-c", SHIM, str(REPO), str(out), case, *arms], check=True,
                   env=dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0"))
    res = json.loads(out.read_text())["arms"]
    summary[tag] = {arm: {k: res[arm].get(k) for k in ("sha", "error")} for arm in arms}
RELEASE = "3356e24924a9207b2264cec1dc8df0591cd85e1a856097251b6f2efeb7327c57"
v = summary["vsso"]
summary["PASS"] = (all(v[a]["sha"] == RELEASE for a in ("release", "topo_wind=1", "topo_wind=2"))
                   and "VAR_SSO" in (summary["rd11"]["topo_wind=1"]["error"] or ""))
(OUT / "MYNN_IDENTITY.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
