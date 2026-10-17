"""Run just the two changed-boundary mutants; retain the unchanged face evidence."""
from pathlib import Path
import sys

root, base, output = sys.argv[1:4]
script = Path(root) / "tests/v025/t_cfix_mynn/mutants.py"
namespace = {"__file__": str(script), "__name__": "critic_fix_mutants"}
exec(compile(script.read_text(), str(script), "exec"), namespace)
selected = ("M5_producer_limits_leak", "M6_view_unlimited")
namespace["MUTANTS"] = {key: namespace["MUTANTS"][key] for key in selected}
namespace["main"](base, output)
