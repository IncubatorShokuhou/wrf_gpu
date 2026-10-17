"""Run the C5 conditional-placement and ANY-versus-ALL mutants only."""
from pathlib import Path
import sys

root, base, output = sys.argv[1:4]
script = Path(root) / "tests/v025/t_cfix_mynn/mutants.py"
ns = {"__file__": str(script), "__name__": "critic_c5_mutants"}
exec(compile(script.read_text(), str(script), "exec"), ns)
ns["MUTANTS"] = {key: ns["MUTANTS"][key] for key in ("M7_reformation_in_step_graph", "M8_cond_all_columns")}
ns["main"](base, output)
