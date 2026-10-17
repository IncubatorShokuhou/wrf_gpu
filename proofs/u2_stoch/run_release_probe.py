"""Use the current official release identity probe with lane-owned scratch."""
import argparse
import os
from pathlib import Path
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--case", type=Path)
args = parser.parse_args()
tree = args.tree.resolve()
probe = tree / "proofs/s1_port/release_identity.py"
text = probe.read_text()
old = "<USER_HOME>/wrf_gpu2_lanes/s1-port"
assert text.count(old) == 1
text = text.replace(old, str(args.output.resolve().parent))
if args.case:
    os.environ["O1_CASE"] = str(args.case.resolve())
sys.argv = [str(probe), str(tree), str(args.output.resolve())]
if args.case:
    sys.argv.append(str(args.case.resolve()))
exec(compile(text, str(probe), "exec"), {"__file__": str(probe), "__name__": "__main__"})
