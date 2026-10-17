"""Full-equation old-table mutant plus finite/field-completeness controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from check_literal_columns import main as compare
from gpuwrf.physics.thompson_hail import native


FIELDS=set("qv qc qr qi qs qg Ni Nr Nc Ng qvolg nwfa nifa T".split())
SURFACES=set("RAINNCV SNOWNCV GRAUPELNCV SR RAINNC SNOWNC GRAUPELNC".split())


def primary_pass(receipt):
    if set(receipt.get("fields",{}))!=FIELDS or set(receipt.get("public_surface",{}))!=SURFACES:
        return False
    count=len(receipt["cases"])
    return all(row.get("finite") and row.get("byte_equal") and row.get("matching")==row.get("total")==40*count
               for row in receipt["fields"].values()) and all(
                   row.get("finite") and row.get("byte_equal") and row.get("matching")==row.get("total")==count
                   for row in receipt["public_surface"].values())


def main(reference, baseline, output):
    output.mkdir(parents=True,exist_ok=True)
    original=json.loads(baseline.read_text())
    assert primary_pass(original)
    nonfinite=copy.deepcopy(original);nonfinite["fields"]["qv"]["finite"]=False
    deletion=copy.deepcopy(original);del deletion["fields"]["Ni"]
    assert not primary_pass(nonfinite) and not primary_pass(deletion)
    saved=native._freezing_tables
    native._freezing_tables=lambda:{name:np.asarray(getattr(native.THOMPSON_AERO_TABLES,name)) for name in ("qrfz4","qcfz")}
    mutant=output/"old_table_mutant.json"
    try:compare(reference,mutant,(5,))
    finally:native._freezing_tables=saved
    changed=json.loads(mutant.read_text());assert not primary_pass(changed)
    assert not changed["fields"]["qi"]["byte_equal"]
    result=dict(scope="Unchanged production equation code objects with legacy freezing-table numerical mutant",
        original_baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),primary_baseline_PASS=True,
        old_table_mutant_rejected=True,nonfinite_rejected=True,missing_field_rejected=True,
        failing_mutant_fields=[name for name,row in changed["fields"].items() if not row["byte_equal"]])
    (output/"receipt.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("reference",type=Path);parser.add_argument("baseline",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();main(args.reference,args.baseline,args.output)
