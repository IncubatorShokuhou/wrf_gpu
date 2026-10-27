"""Per-file, per-variable sha256 of the RAW stored values (no mask/scale) + global attributes, CPU only; values never written.
usage: JAX_PLATFORMS=cpu python3 digests.py DIR DOMAIN OUT.json [MAX_FILES]
Compare two outputs with: digests.py --compare A.json B.json  (frames value-identical, first differing frame/vars, attr diffs)."""
import glob, hashlib, json, os, sys

import numpy as np
from netCDF4 import Dataset

if sys.argv[1] == "--compare":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    common = sorted(set(a) & set(b))
    same, first, attrs = 0, None, set()
    for n in common:
        dv = sorted(v for v in set(a[n]["vars"]) | set(b[n]["vars"]) if a[n]["vars"].get(v) != b[n]["vars"].get(v))
        attrs |= {k for k in set(a[n]["attrs"]) | set(b[n]["attrs"]) if a[n]["attrs"].get(k) != b[n]["attrs"].get(k)}
        if dv:
            first = first or {"file": n, "n_vars": len(dv), "vars": dv[:15]}
        else:
            same += 1
    print(json.dumps({"common_frames": len(common), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
                      "frames_value_identical": same, "first_diff": first, "attr_keys_differing": sorted(attrs)}))
    sys.exit(0)
d, dom, out = sys.argv[1:4]
lim = int(sys.argv[4]) if len(sys.argv) > 4 else None
fs = sorted(p for p in glob.glob(f"{d}/wrfout_{dom}_*") if not p.endswith(".tmp"))[:lim]
res = {}
for p in fs:
    with Dataset(p) as ds:
        ds.set_auto_maskandscale(False)
        res[os.path.basename(p)] = {"vars": {n: hashlib.sha256(np.ascontiguousarray(v[:]).tobytes()).hexdigest()[:24] for n, v in ds.variables.items()},
                                    "attrs": {k: str(ds.getncattr(k)) for k in ds.ncattrs()}}
json.dump(res, open(out, "w"))
print("digested", len(res), "files")
