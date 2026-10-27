"""Frozen mixed-phase increment gate, finite-first; changed WRF vs native CPU.

Same tolerances/floor rule as tests/v025/b_thompson/test_mixed_phase_wrf_oracle.py.
The selected changed-value reference forms its own fixture-wide scale. Also
score the default-value port against the changed WRF as an omission mutant.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("--fixture",type=Path,required=True)
parser.add_argument("--selection",type=Path,required=True)
parser.add_argument("--reference",type=Path,required=True)
parser.add_argument("--port",type=Path,required=True)
parser.add_argument("--mutant",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
from oracle_gate import score_arrays
z=dict(np.load(args.fixture,allow_pickle=False))
w=dict(np.load(args.reference,allow_pickle=False))
selection=json.loads(args.selection.read_text())
idx=np.array([z["names"].tolist().index(n) for n in selection["selected_columns"]])
assert all(np.isfinite(v).all() for k,v in z.items() if k.startswith(("in_","wrf_")))
assert all(np.isfinite(v).all() for v in w.values())


def score(path):
    return score_arrays(z,idx,w,dict(np.load(path,allow_pickle=False)))

candidate,mutant=score(args.port),score(args.mutant)
receipt=dict(platform="cpu",gpu_executed=False,columns=len(idx),candidate=candidate,default_omission_mutant=mutant,
             candidate_pass=all(row["failed_columns"]==0 for row in candidate.values()),
             mutant_killed=any(row["failed_columns"]>0 for row in mutant.values()),
             hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (args.fixture,args.selection,args.reference,args.port,args.mutant)})
args.out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
assert receipt["candidate_pass"],candidate
assert receipt["mutant_killed"],mutant
