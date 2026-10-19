"""Replay the wn3_score gate policy (old main vs candidate) on existing real comparator reports: report_gate_failures
must return identical lists and no reference exemption may apply, so summary.json bytes cannot change.
usage: python policy_replay.py OLD_TREE NEW_TREE OUT_JSON LABEL=DIR:FRAMES ..."""
import importlib.util
import json
import sys
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old = load(Path(sys.argv[1]) / "scripts/wn3_score.py", "old_score")
new = load(Path(sys.argv[2]) / "scripts/wn3_score.py", "new_score")
rows, ok = {}, True
for arg in sys.argv[4:]:
    label, rest = arg.split("=", 1)
    d, frames = rest.rsplit(":", 1)
    summary = json.loads((Path(d) / "summary.json").read_text())
    for dom in ("d01", "d02", "d03"):
        rep = json.loads((Path(d) / f"{dom}.json").read_text())
        a, b = old.report_gate_failures(rep, int(frames)), new.report_gate_failures(rep, int(frames))
        ex = new.reference_nonfinite_exempted(rep)
        same = a == b == summary["domains"][dom]["policy_failures"] and ex == []
        ok &= same
        rows[f"{label}/{dom}"] = {"identical": same, "policy_failures": len(b), "kinds": sorted({f["kind"] for f in b}),
                                  "exempted": ex}
Path(sys.argv[3]).write_text(json.dumps({"identical": ok, "rows": rows}, indent=1) + "\n")
print(json.dumps({"identical": ok, "rows": rows}))
sys.exit(0 if ok else 1)
