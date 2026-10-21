"""Stage the AMEND1-admitted teacher_extract example/reference extracts + ALISIOS kit via alisios_guard.facade_stage
(scan -> guard_call.admit -> READ_SPY), one GuardedRead session per group; staged sha256 must equal the admission's sha256.
usage: JAX_PLATFORMS=cpu python3 analysis/stage_te.py ADMISSION_JSON STAGE_ROOT LABEL"""
import json, sys
from collections import defaultdict
from datetime import datetime, timedelta
sys.path.insert(0, "src")
from gpuwrf.assimilation.letkf import alisios_guard as ag
adm, root, label = sys.argv[1:4]
A = json.load(open(adm)); assert A["admission_id"] == "A-V043-TEACHER-QUAL-AMEND1" and A["status"] == "ISSUED"
want = {f["path"]: f["sha256"] for f in A["files"]}
groups = defaultdict(list)
for p in want:
    seg = p.split("/teacher_extract/")[-1].split("/")[0] if "/teacher_extract/" in p else ""
    groups[seg[:8] if seg[:8].isdigit() else "20260102"].append(p)   # registry + kit ride with the example issue
out = {}
for issue, fs in sorted(groups.items()):
    d0 = datetime.strptime(issue, "%Y%m%d"); days = [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range(4)]
    staged, rec = ag.facade_stage({f: ([issue], days) for f in fs}, f"{root}/{issue}", f"analysis/receipts/{label}/{issue}",
        purpose=f"w43-teacher: teacher_extract format reproduction check ({issue}) under A-V043-TEACHER-QUAL-AMEND1")
    for k, v in rec["files"].items():
        assert v["sha256"] == want[k], (k, v["sha256"], want[k])
        out[k] = dict(staged=staged[k], sha256=v["sha256"], size=v["size"])
    print(issue, len(fs), rec["counters"], rec["spy"]["refused_paths"], flush=True)
assert set(out) == set(want), sorted(set(want) - set(out))
json.dump(out, open(f"analysis/receipts/{label}/INDEX.json", "w"), indent=1)
print("staged", len(out), "all sha == admission")
