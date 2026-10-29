"""Default PLAN_ONLY. Manager may --apply after the token-fix critic accepts. No paid-cache touch."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,sys
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');A=Path('<DATA_ROOT>/wrf_gpu2/scratch/v45-rc/session_254c48237452_gpu_v6/runs/A2_prod');O=R/'TOKEN_FIX/failed_A2_prod';sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args();assert os.environ.get('JAX_PLATFORMS')=='cpu';assert not Path('/tmp/wrf_gpu2_quiet').exists()
assert A.is_dir()and not A.is_symlink();end=json.loads((A/'REGISTRY_END.json').read_text());start=json.loads((A/'REGISTRY_START.json').read_text());t=json.loads((A/'timer.json').read_text());assert end['rc']==75 and end['status']=='FAILED' and t['teardown']['all_reaped']
assert not list((A/'wrfout').glob('*')),'retain all actual raw; this repair only handles zero-output preflight failure'
rows=[]
for f in sorted(A.rglob('*')):
 if f.is_file():
  assert not f.is_symlink();rows.append(dict(relative=str(f.relative_to(A)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
print('RETRY PREP', 'APPLY'if a.apply else'PLAN_ONLY','zero outputs; preserve',len(rows),'metadata files; paid A1/cache untouched')
if not a.apply:raise SystemExit(0)
assert not O.exists();shutil.copytree(A,O)
for row in rows:assert hashlib.sha256((O/row['relative']).read_bytes()).hexdigest()==row['sha256']
receipt=R/'TOKEN_FIX/FAILED_A2_ARCHIVE.json';receipt.write_text(json.dumps(dict(original=str(A),archive=str(O),rows=rows,run_id=start['run_id'],zero_wrfout=True),indent=2)+'\n')
# Authorized retention metadata repair; no START/END/backfill, original FAILED status unchanged.
spec=importlib.util.spec_from_file_location('retry_registry',R/'archive/scripts/alisios_registry.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
m.Registry().update(dict(run_id=start['run_id'],by='v45-rc',raw_deleted=True,extract=[str(receipt)+'::'+hashlib.sha256(receipt.read_bytes()).hexdigest()],notes='Zero-output preflight failure metadata SHA-archived on ROOT before E42-clean retry; FAILED status unchanged'),authorization='manager22:54 urgent token-fix retry preparation; critic accepted delta')
# Only this proven failed zero-output directory; never the paid A1, source, bundle or case.
shutil.rmtree(A)
print('RETRY PREP complete; canonical A2 output path fresh, all failed evidence archived')
