"""Score completed GPU component outputs against the retained original WRF arms.

CPU-only, no GPU launch. Each changed control has a recompiled-PARAMETER
reference and a default-value omission control at the frozen increment bounds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--manifest",type=Path,required=True)
parser.add_argument("--run",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
assert os.environ.get("JAX_PLATFORMS")=="cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists()
base=Path("<USER_HOME>/wrf_gpu2_lanes/w44-thparams")
manifest=json.loads(args.manifest.read_text())
receipt_path=args.run/"receipt.json"
receipt=json.loads(receipt_path.read_text())
assert receipt["gpu_executed"] and receipt["platform"]=="gpu"
assert receipt["pin"]==manifest["source_pin"]
assert receipt["fixture_sha256"]==manifest["fixture_sha256"]
assert receipt["status"]=="PASS",receipt.get("error")
assert receipt["default_roundoff"]["within_roundoff"] and receipt["traces"]==2
assert all(row["prognostic_byte_exact"] for row in receipt["events"].values())
refs={"nt50":"NT04/wrf_nt50", "nt150":"NT05/wrf_nt150", "tno":"TP05/wrf_tno",
      "ato":"TP06/wrf_ato", "av_c":"TP14/wrf_avc"}
tree=Path(manifest["snapshot"])
fixture=tree/"tests/v025/b_thompson/fixtures/mixed_phase_columns.npz"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(fixture)==manifest["fixture_sha256"]
assert not args.out.exists(),"refuse to overwrite GPU science evidence"
args.out.mkdir(parents=True)
results={}
for name,relative in refs.items():
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise SystemExit("QUIET: retain partial scores; remaining arms unmeasured")
    ref=base/relative
    original=json.loads((ref/"receipt.json").read_text())
    assert original["selection"]==receipt["selected_columns"]
    assert original["fixture_sha256"]==manifest["fixture_sha256"]
    cpu_score=json.loads((ref.parent/"score.json").read_text())
    assert sha(ref/"reference.npz")==cpu_score["hashes"][str(ref/"reference.npz")], "original reference changed"
    output=args.out/(name+".json")
    command=[sys.executable,str(tree/"proofs/w44_thparams/score_oracle.py"),
             "--fixture",str(fixture),"--selection",str(receipt_path),
             "--reference",str(ref/"reference.npz"),"--port",str(args.run/(name+".npz")),
             "--mutant",str(args.run/"default.npz"),"--out",str(output)]
    with (args.out/(name+".log")).open("w") as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
    report=json.loads(output.read_text())
    results[name]={"rc":result.returncode,"candidate_pass":report["candidate_pass"],
                   "mutant_killed":report["mutant_killed"],"score_sha256":sha(output),
                   "original_receipt_sha256":sha(ref/"receipt.json")}
summary={"source_pin":receipt["pin"],"gpu_receipt_sha256":sha(receipt_path),
         "scope":"Actual GPU column outputs vs original recompiled WRF, no forecast/station claim",
         "fields":results,"science_pass":all(r["rc"]==0 and r["candidate_pass"] and r["mutant_killed"] for r in results.values()),
         "pre_mask_events":receipt["events"],"inadmissible_events_need_review":receipt["inadmissible_events_need_review"]}
(args.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
assert summary["science_pass"]
