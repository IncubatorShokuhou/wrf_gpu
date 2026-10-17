"""Independent deletions of API admission / last-option / abbreviation fixes."""
import inspect
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/"scripts/v025"))
import cpu_guard
import pytest
from gpuwrf.io import cammg_config

kind, base, output = sys.argv[1:]
if kind == "api":
    cammg_config.validate_input_dir = lambda *a, **k: None
    selection = "api_refuses_before_initialization or all_batch_inputs_admitted_before_dfi or api_admitted_control"
elif kind == "first":
    source = inspect.getsource(cammg_config.bootstrap_refusal)
    needle = "range(len(arguments) - 1, -1, -1)"
    assert source.count(needle) == 1
    exec(source.replace(needle, "range(len(arguments))"), cammg_config.__dict__)
    selection = "effective_options_use_actual_argparse or api_admitted_control"
elif kind == "abbreviation":
    source = inspect.getsource(cammg_config.bootstrap_refusal)
    needle = 'return "GPU admission requires full --namelist / --input-dir spellings; abbreviations are refused before JAX."'
    assert source.count(needle) == 1
    exec(source.replace(needle, "continue"), cammg_config.__dict__)
    selection = "abbreviations_refuse_directly or api_admitted_control"
else:
    raise ValueError(kind)
raise SystemExit(pytest.main([str(root/"tests/test_cammg_entry_order.py"), "-q", "-k", selection,
                             "--basetemp", base, "--junitxml", output]))
