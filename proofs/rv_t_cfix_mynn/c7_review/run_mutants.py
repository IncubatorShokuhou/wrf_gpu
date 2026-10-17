"""Only C7's changed structural/face-operand mutants; earlier numerical oracles are reused."""
from pathlib import Path
import sys

root, base, output = sys.argv[1:4]
script = Path(root) / "tests/v025/t_cfix_mynn/mutants.py"
ns = {"__file__": str(script), "__name__": "critic_c7_mutants"}
exec(compile(script.read_text(), str(script), "exec"), ns)
ns["MUTANTS"] = {k: v for k, v in ns["MUTANTS"].items() if k.startswith(("M7_", "M8_", "M9_", "M10_"))}
assert len(ns["MUTANTS"]) == 4
ns["main"](base, output)
