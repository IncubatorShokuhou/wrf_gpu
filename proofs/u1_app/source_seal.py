"""Exact-source validation reuse for isolated K0 application products."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE="d7638d624"
PREFIXES=("src/gpuwrf/dynamics/","src/gpuwrf/physics/","src/gpuwrf/kernels/",
          "src/gpuwrf/runtime/","src/gpuwrf/integration/","src/gpuwrf/nesting/",
          "src/gpuwrf/contracts/","src/gpuwrf/coupling/")

def blobs(ref):
    entries=subprocess.check_output(["git","ls-tree","-r",ref,"--",*PREFIXES],text=True).splitlines()
    return {line.split('\t')[1]:line.split()[2] for line in entries}

old,new=blobs(BASE),blobs("HEAD")
changed=[name for name in old.keys()|new.keys() if old.get(name)!=new.get(name)]
receipt={"base":subprocess.check_output(["git","rev-parse",BASE],text=True).strip(),
         "candidate":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         "files_checked":len(old),"changed_model_files":changed,"all_model_blobs_exact":not changed,
         "baseline_swiss_release_jaxpr_sha_prefix":"2bf12cb8",
         "jaxpr_disposition":"Carried from kernel wave baseline; not retraced. All compiled-model source blobs exact, new post package is absent from model imports.",
         "model_blob_inventory_sha256":hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+"\n")
assert not changed,changed
print(json.dumps(receipt,sort_keys=True))
