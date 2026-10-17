import sys, json, tempfile
from pathlib import Path
sys.path.insert(0, "<USER_HOME>/src/wrf_gpu2_wt/o2-rootbdy/tests/v040/rootbdy")
import test_root_flow_extras as t
import test_root_scalar_bdy_rk1 as b43
from pristine_boundary import build
ops = b43._load_prod_operands()
lib = build(Path(tempfile.mkdtemp(dir="<USER_HOME>/wrf_gpu2_lanes/o2-rootbdy/tmp")))
out = {}
for mp in sorted(t.EXTRAS):
    for drop in (False, True):
        wrf, port = t._chain(ops, lib, mp, drop_flow=t.EXTRAS[mp] if drop else ())
        spec, relax, interior = b43._zones(wrf["qv"].shape)
        for n in t.EXTRAS[mp]:
            e = b43._ulp_of_max(port[n], wrf[n])
            out[f"mp{mp}{'_drop' if drop else ''}_{n}"] = dict(spec=float(e[:, spec].max()), relax=float(e[:, relax].max()), interior=float(e[:, interior].max()))
print(json.dumps(out, indent=0))
json.dump(out, open("<USER_HOME>/wrf_gpu2_lanes/o2-rootbdy/dbg/margins.json", "w"), indent=1)
