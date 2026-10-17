"""Report compiled deviations from independently frozen original intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS=dict(qv="QV",qc="QC",qr="QR",qi="QI",qs="QS",qg="QG",Ni="NI",Nr="NR",Nc="NC",Ng="NG",qvolg="QB",nwfa="NWFA",nifa="NIFA",T="T")


def main(intervals, compiled, output):
    rows=[];commits=set()
    with np.load(intervals,allow_pickle=False) as bounds:
        for path in compiled:
            receipt=json.loads(path.read_text());commits.add(receipt.get("candidate_commit","legacy-H60-model91d"))
            for index,case in enumerate(receipt["cases"]):
                names=list(FIELDS.items())+[(name,None) for name in receipt["public_surface"]]
                for name,code in names:
                    key=f"case{case}__{code+'_OUT' if code else name}"
                    actual=np.asarray(receipt["fields"][name]["actual"][index] if code else receipt["public_surface"][name]["actual"][index],np.float32)
                    lower,upper=bounds[key+"__lower"],bounds[key+"__upper"]
                    assert actual.shape==lower.shape==upper.shape and np.isfinite(actual).all()
                    excess=np.maximum(np.maximum(lower.astype(float)-actual,actual.astype(float)-upper),0.)
                    unit="number kg-1" if name in ("Ni","Nr","Nc","Ng","nwfa","nifa") else "kg kg-1" if name in ("qv","qc","qr","qi","qs","qg") else "m3 kg-1" if name=="qvolg" else "K" if name=="T" else "1" if name=="SR" else "mm"
                    rows.append(dict(case=case,field=name,unit=unit,total=int(actual.size),outside=int(np.count_nonzero(excess)),max_excess=float(excess.max())))
    result=dict(scope="Independent original-WRF per-field input-ULP/FMA response interval; excess is disclosed, never widened by candidate",
                candidate_commits=sorted(commits),interval_sha256=hashlib.sha256(intervals.read_bytes()).hexdigest(),
                total=sum(row["total"] for row in rows),outside=sum(row["outside"] for row in rows),all_finite=True,
                strict_interval_PASS=not any(row["outside"] for row in rows),small_excess_policy="Standing wave policy; final coupledD6 pending",rows=rows)
    output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("intervals",type=Path);parser.add_argument("output",type=Path);parser.add_argument("compiled",nargs="+",type=Path)
    args=parser.parse_args();main(args.intervals,args.compiled,args.output)
