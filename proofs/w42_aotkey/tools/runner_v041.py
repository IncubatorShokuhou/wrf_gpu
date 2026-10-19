"""Run `gpuwrf aot-key` with the v0.4.1 aot_key_cli.py (the unfixed tool) on the current tree (src tree == v0.4.1)."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("gpuwrf.runtime.aot_key_cli", Path(__file__).with_name("aot_key_cli_v041.py"))
module = importlib.util.module_from_spec(spec)
import gpuwrf.runtime  # noqa: E402,F401  (parent package first)
sys.modules["gpuwrf.runtime.aot_key_cli"] = module
spec.loader.exec_module(module)
from gpuwrf import cli  # noqa: E402

assert cli.build_parser().parse_args(["aot-key", "--input-dir", "."]).func is module.command
print("runner: v0.4.1 aot_key_cli", file=sys.stderr)
sys.exit(cli.main(sys.argv[1:]))
