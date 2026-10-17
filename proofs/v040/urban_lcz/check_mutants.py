"""Two real source deletions; only assertion failures count as killed (E202)."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT=Path(sys.argv[1]).resolve()
OUT=Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/mutants')
OUT.mkdir(exist_ok=True)
cases=[('lcz_default_type','src/gpuwrf/physics/urban_slucm.py',
    'utype = jnp.where(v == isurban, 5,','utype = jnp.where(v == isurban, 2,',
    'tests/test_v040_urban_lcz.py::test_loader_wires_lcz_morphology_and_ahe','gpuwrf.physics.urban_slucm'),
    ('clock_forward','src/gpuwrf/coupling/noahmp_surface_hook.py',
    '**({"urban_lead_seconds":urban_lead_seconds} if urban is not None\n           and getattr(urban[0],"requires_inputs",False) else {}),','',
    'tests/test_v040_urban_lcz.py::test_production_surface_hook_forwards_urban_clock','gpuwrf.coupling.noahmp_surface_hook')]
receipts={}
for name,file,old,new,node,module in cases:
    source=(ROOT/file).read_text()
    assert source.count(old)==1,(name,source.count(old))
    mutant=source.replace(old,new)
    directory=OUT/name;directory.mkdir(exist_ok=True)
    path=directory/'mutant.py';path.write_text(mutant)
    plugin=directory/'plugin.py'
    plugin.write_text('''import importlib,sys
def pytest_configure(config):
    original=importlib.import_module(MODULE)
    exec(compile(SOURCE,FILE,'exec'),original.__dict__)
'''.replace('MODULE',repr(module)).replace('SOURCE',repr(mutant)).replace('FILE',repr(str(path))))
    env=dict(os.environ,PYTHONPATH=str(directory)+os.pathsep+str(ROOT/'src'))
    assert env.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    result=subprocess.run([sys.executable,'-m','pytest','-q','-p','plugin',node,
        '-o','pythonpath='+str(ROOT/'src'),'--tb=short','--basetemp','<USER_HOME>/wrf_gpu2_lanes/u2-lcz/pytest'],
        cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    (directory/'result.log').write_text(result.stdout)
    setup='ERROR at setup' in result.stdout or 'ERROR collecting' in result.stdout
    killed=result.returncode==1 and ('AssertionError' in result.stdout or '\nE   assert' in result.stdout) and not setup
    receipts[name]={'genuine_assertion':killed,'setup_error':setup,'returncode':result.returncode,
        'node':node,'baseline_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'mutant_sha256':hashlib.sha256(mutant.encode()).hexdigest(),'source_file':str(ROOT/file)}
    print(name,receipts[name],flush=True)
(OUT/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
assert all(r['genuine_assertion'] for r in receipts.values()),receipts
