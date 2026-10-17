"""Restore an exact pre-fix caller and require genuine numerical test failures."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser();p.add_argument('tree',type=Path);p.add_argument('out',type=Path)
    p.add_argument('kind',choices=('tiedtke','acm2'));a=p.parse_args()
    sys.path[:0]=[str(a.tree/'scripts/v025'),str(a.tree/'src'),str(a.tree/'tests')]
    import cpu_guard  # establish the project's backend guard before importing JAX
    import pytest
    from gpuwrf.coupling import scan_adapters as sa
    from gpuwrf.runtime import operational_mode as om
    module,name,path=(sa,'tiedtke_adapter','src/gpuwrf/coupling/scan_adapters.py') if a.kind=='tiedtke' else (
        om,'_physics_step_forcing','src/gpuwrf/runtime/operational_mode.py')
    old=subprocess.check_output(['git','show','507273a35:'+path],cwd=a.tree,text=True)
    function=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name==name)
    # Keep the real module globals: a copied namespace freezes kernel aliases
    # and defeats the caller test's boundary spy (a KeyError is not a kill).
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 '<exact-parent-caller-mutant>','exec'),module.__dict__)
    a.out.mkdir(parents=True,exist_ok=True);xml=a.out/(a.kind+'.xml')
    rc=pytest.main([str(a.tree/'tests/test_t_rainnc_callers.py'),'-q','--tb=short','-k',a.kind,
                   '--basetemp','<USER_HOME>/wrf_gpu2_lanes/t-rainnc/pytest','--junitxml',str(xml)])
    root=ET.parse(xml).getroot();failures=root.findall('.//failure');errors=root.findall('.//error')
    verdict=dict(kind=a.kind,parent='507273a35',returncode=rc,
                 numeric_assertions=sum('AssertionError' in (e.text or '') for e in failures),
                 other_failures=sum('AssertionError' not in (e.text or '') for e in failures),
                 errors=len(errors),genuine=all('AssertionError' in (e.text or '') for e in failures))
    (a.out/(a.kind+'.json')).write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict))
    assert rc==1 and failures and not errors and verdict['genuine'],'not a genuine deletion kill'


if __name__=='__main__':main()
