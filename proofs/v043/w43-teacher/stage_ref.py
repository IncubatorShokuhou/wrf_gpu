"""Stage one WT2 original CPU-WRF 72 h reference (219 wrfout) via alisios_guard.facade_stage, one GuardedRead session per domain
(CIFS identity glitches abort a whole session -> smaller sessions; a refused session leaves its receipt and is retried by the caller).
usage: JAX_PLATFORMS=cpu python3 analysis/stage_ref.py ISSUE(YYYYMMDD) STAGE_ROOT LABEL [DOMS=d01,d02,d03]"""
import glob, json, sys
from datetime import datetime, timedelta
sys.path.insert(0, "src")
from gpuwrf.assimilation.letkf import alisios_guard as ag
issue, root, label = sys.argv[1:4]
doms = sys.argv[4].split(",") if len(sys.argv) > 4 else ["d01", "d02", "d03"]
R = f"<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_{issue}_18z_a1/run/run"
d0 = datetime.strptime(issue, "%Y%m%d"); days = [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range(5)]
out = {}
for dom in doms:
    fs = sorted(glob.glob(f"{R}/wrfout_{dom}_*")); assert len(fs) == 73, (dom, len(fs))
    staged, rec = ag.facade_stage({f: ([issue], days) for f in fs}, f"{root}/{dom}", f"analysis/receipts/{label}/{dom}",
        purpose=f"w43-teacher: original CPU-WRF {issue} 72 h reference frames {dom} (WT2) for teacher D6/A3/G-INT scoring")
    out.update({k: dict(staged=staged[k], sha256=v["sha256"]) for k, v in rec["files"].items()})
    print(dom, rec["counters"], rec["spy"]["refused_paths"], flush=True)
json.dump(out, open(f"analysis/receipts/{label}/INDEX_{'_'.join(doms)}.json", "w"), indent=1)
print("staged", len(out))
