"""Stage WT2 reference wrfout ONE FILE PER facade_stage session with up to 3 attempts (CIFS <DATA_ROOT>/server ctime/identity flaps
abort a session: guarded_read file_identity_changed). Refused attempts keep their receipts (<label>/<file>/aN).
usage: JAX_PLATFORMS=cpu python3 analysis/stage_ref_files.py ISSUE STAGE_ROOT LABEL DOMAIN [DOMAIN ...]"""
import glob, json, shutil, sys, time
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, "src")
from gpuwrf.assimilation.letkf import alisios_guard as ag
issue, root, label, doms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
R = f"<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_{issue}_18z_a1/run/run"
d0 = datetime.strptime(issue, "%Y%m%d"); days = [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range(5)]
idx, refusals = {}, []
for dom in doms:
    fs = sorted(glob.glob(f"{R}/wrfout_{dom}_*")); assert len(fs) == 73, (dom, len(fs))
    for f in fs:
        name = Path(f).name
        for a in range(1, 4):
            sr = Path(root) / dom / name / f"a{a}"
            try:
                staged, rec = ag.facade_stage({f: ([issue], days)}, sr, f"analysis/receipts/{label}/{dom}/{name}/a{a}",
                    purpose=f"w43-teacher: original CPU-WRF {issue} 72 h reference frame {name} (WT2) for teacher scoring")
                idx[f] = dict(staged=staged[f], sha256=rec["files"][f]["sha256"], attempt=a)
                break
            except Exception as e:
                refusals.append(dict(file=name, attempt=a, error=f"{type(e).__name__}: {str(e)[:160]}"))
                shutil.rmtree(sr, ignore_errors=True); time.sleep(5)
        else:
            raise SystemExit(f"FAILED after 3 attempts: {name}")
    print(dom, "staged", sum(1 for k in idx if f"wrfout_{dom}_" in k), "refusals so far", len(refusals), flush=True)
out = Path(f"analysis/receipts/{label}/INDEX_{'_'.join(doms)}.json")
out.write_text(json.dumps({"files": idx, "refused_attempts": refusals}, indent=1))
print("staged", len(idx), "refused attempts", len(refusals))
