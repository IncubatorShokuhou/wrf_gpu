"""Retained S0 harness; only its CLI selection changes to 3 nests + t0 history."""
import os
from pathlib import Path

source = Path(os.environ["BENCH_SRC_REPO"]) / "scripts/bench/s0_nested_harness.py"
code = source.read_text()
old = '"--max-dom", "2", "--hours", str(args.hours)'
assert code.count(old) == 1
code = code.replace(old, '"--max-dom", "3", "--emit-initial-history", "--hours", str(args.hours)')
exec(compile(code, str(source), "exec"), {"__file__": str(source), "__name__": "__main__"})
