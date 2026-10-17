"""Deterministic broad sounding search for missing pristine SAS94 regimes.

Reuses the independently hydrostatic sounding generator of the scale-SAS oracle;
these are constructed oracle columns, not a forecast or a speed benchmark.
"""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def generate(work, n):
    path = ROOT/"proofs/v034/scalesas/gen_inputs.py"
    spec = importlib.util.spec_from_file_location("sas_columns",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    work.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(940040)
    kinds = ("tropical","lowbase","moist_neutral","capped","cold","mixed")
    cols = [module.synthetic_column(rng,kinds[i%len(kinds)]) for i in range(n)]
    module.write_case(work,"search_shallow1",cols,1.+(np.arange(n)%2),np.full(n,9000.),
                      (18.,1,10,9000.,.55),[kinds[i%len(kinds)] for i in range(n)])
    inp = work/"search_shallow1.txt"
    lines = inp.read_text().splitlines()
    lines[1] += " 1"
    inp.write_text("\n".join(lines)+"\n")
    report = {"source":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "seed":940040,"columns":n,"kinds":kinds,"oracle_only":True}
    (work/"generation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    generate(Path(sys.argv[1]),int(sys.argv[2]))
