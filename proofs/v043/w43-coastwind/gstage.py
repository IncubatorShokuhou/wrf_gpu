"""Guarded staging of ALLOW-day ALISIOS files for w43-coastwind (alisios_guard.facade_stage: sealed GuardedRead,
scan+admit before any payload open, READ_SPY). usage: JAX_PLATFORMS=cpu python gstage.py LABEL LIST.json
LIST.json = [[path, [issue_days], [valid_days]], ...]. Stage root <DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/stage/LABEL,
receipts analysis/receipts/LABEL. Values never logged."""
import json, os, sys
from pathlib import Path
assert os.environ.get("JAX_PLATFORMS") == "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpuwrf.assimilation.letkf import alisios_guard as ag
label, lst = sys.argv[1], json.load(open(sys.argv[2]))
targets = {p: (i, v) for p, i, v in lst}
allowed, verdicts = ag.drop_deny_days([(p, d) for p, (i, v) in targets.items() for d in sorted(set(i) | set(v))])
bad = sorted({d for d, x in verdicts.items() if x != "ALLOW"})
if bad:
    print(json.dumps({"refused_days": bad})); sys.exit(2)
staged, receipt = ag.facade_stage(targets, f"<DATA_ROOT>/wrf_gpu2/scratch/w43-coastwind/stage/{label}",
                                  Path(__file__).resolve().parent / "receipts" / label,
                                  purpose="w43-coastwind (V043 P1 coastal wind, El Medano): " + label)
Path(Path(__file__).resolve().parent / "receipts" / label / "STAGED.json").write_text(json.dumps(staged, indent=1))
print(json.dumps({"n": len(staged), "counters": receipt["counters"], "refused": receipt["spy"]["refused_paths"]}))
