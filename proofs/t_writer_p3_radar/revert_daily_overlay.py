"""Restore the actual reviewed host helper; the RUC route must value-ASSERT."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

def pytest_configure(config):
    from gpuwrf.integration import daily_pipeline as daily
    revision='6e911aa42c6f9559a0a720526d3e7fb915bfdd07'
    text=subprocess.check_output(['git','show',revision+':src/gpuwrf/integration/daily_pipeline.py'],text=True)
    fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_surface_diagnostics_for_output')
    old=ast.get_source_segment(text,fn)
    exec(compile(old,str(daily.__file__)+'::<reviewed-overlay-restored>','exec'),vars(daily))
    Path(os.environ['T_WRITER_MUTANT_RECEIPT']).write_text(json.dumps(dict(
        restored_source=revision,function=fn.name,source_sha256=hashlib.sha256(old.encode()).hexdigest(),
        scope='actual prior helper restored in module globals; real daily loop unchanged'),indent=2)+'\n')
