"""Unchanged E41 thresholds, also discover the actual budget component exports."""
import importlib.util
import os
from pathlib import Path
import re
import sys


def main():
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    root=Path(os.environ['BUDGET_MODEL_ROOT'])
    sys.path.insert(0,str(root/'src'))
    spec=importlib.util.spec_from_file_location('budget_wave_gate',root/'scripts/wave_cubin_gate.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    # Extend discovery only. Coverage, native stack/callee and optimized-HLO
    # checks, missing evidence and all numeric thresholds stay unchanged.
    module.STEP_PROGRAM=re.compile(r'(?:fused_)?d\d+|budget_component_wn3_d03_(?:day|night)_edmf[01]_(?:off|on)\Z')
    return module.main()


if __name__=='__main__':raise SystemExit(main())
