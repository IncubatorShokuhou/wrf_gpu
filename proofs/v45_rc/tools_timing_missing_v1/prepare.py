"""Build an empty ROOT-only bundle skeleton around the one pinned RC archive.
No JAX, source copy, DATA write or paid-cache seed. Run freeze.sh first.
"""
import argparse,datetime,hashlib,json,re,shutil,subprocess
from pathlib import Path
R=Path('<USER_HOME>/wrf_gpu2_lanes/v45-rc');REPO=Path('<USER_HOME>/src/wrf_gpu2');T=R/'tools_v6'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True).strip()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('pin');p.add_argument('--standin',action='store_true');p.add_argument('--source-tree',type=Path,default=R/'archive');p.add_argument('--allow-src-diverge',action='store_true');a=p.parse_args()
 assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
 assert shutil.disk_usage('/').free>=21*2**30,'ROOT free must allow >=20GiB throughout'
 pin=git('rev-parse',a.pin+'^{commit}');src=git('rev-parse',pin+':src');source=a.source_tree.resolve()
 for item,want in [('HEAD',pin),('HEAD:src',src)]:assert subprocess.check_output(['git','-C',str(source),'rev-parse',item],text=True).strip()==want
 assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all','--','src','scripts'],text=True).strip()
 assert (source/'data/wrf_pristine/WRF/run/MPTABLE.TBL').is_file()
 ver=re.search(r'^__version__ = "([^"]+)"',git('show',pin+':src/gpuwrf/__init__.py'),re.M).group(1)
 pver=re.search(r'^version = "([^"]+)"',git('show',pin+':pyproject.toml'),re.M).group(1)
 assert ver==pver=='0.4.5'
 main_src=git('rev-parse','main:src');assert a.allow_src_diverge or main_src==src
 root=R/'smoke_v6' if a.standin else R/'runtime';mdir=root if a.standin else T
 manpath=mdir/'PIN_MANIFEST.json';assert not manpath.exists(),'write-once pin'
 c=Path('<DATA_ROOT>/wrf_gpu2/scratch/w43-release/cases');cs=Path('<USER_HOME>/wrf_gpu2_lanes/w43-release/cases_sha256.txt')
 for line in cs.read_text().splitlines():
  if line.strip():h,name=line.split(None,1);assert sha(c/name)==h,name
 inputs=dict(SWISS='<DATA_ROOT>/wrf_gpu2/archive/v040/v-prep/F20_20261016/Swiss/input',PROD='<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725',**{'WN3_'+k.upper():str(c/('wn3_'+k)) for k in ['0227','0227_48h','0227_icloud0','0502','0502_icloud0','0614']})
 metadata={}
 for k,d in inputs.items():
  nml=(Path(d)/'namelist.input').read_text();dates=[]
  for f in ['year','month','day','hour']:
   match=re.search(r'\bstart_'+f+r'\s*=\s*(\d+)',nml,re.I);assert match,(k,f);dates.append(int(match.group(1)))
  issue='%04d%02d%02d_%02dz'%tuple(dates)
  nests=int(re.search(r'\bmax_dom\s*=\s*(\d+)',nml,re.I).group(1))
  metadata[k]=dict(forcing='ERA5' if k=='SWISS' else 'WN2' if k=='PROD' else 'WN3',issue=issue,nests=nests)
 bundles={}
 for k in ['prod','mini']:
  b=root/('bundle' if k=='prod' else 'cuda_mini');assert not (b/'jax/aot').exists() and not (b/'sealed').exists()
  for d in ['jax/xla_gpu_per_fusion_autotune_cache_dir','autotune','triton','cuda','meta','tools','receipts']:(b/d).mkdir(parents=True,exist_ok=True)
  if not (b/'src').exists():(b/'src').symlink_to(source,target_is_directory=True)
  assert (b/'src').resolve()==source
  text=(f'# v0.4.5 sealed-bundle environment (v45-rc prepare, pin {pin}): source this file unchanged in priming AND in every\n'
        '# consumer arm (release defaults: set no other GPUWRF_* model flag, no XLA_FLAGS, no CUDA_ROOT)\n'
        f'export PYTHONPATH={source}/src\n'
        f'export GPUWRF_JAX_CACHE_DIR={b}/jax JAX_COMPILATION_CACHE_DIR={b}/jax GPUWRF_XLA_AUTOTUNE_CACHE_DIR={b}/autotune\n'
        f'export TRITON_CACHE_DIR={b}/triton CUDA_CACHE_PATH={b}/cuda\n')
  (b/'bundle.env').write_text(text);(b/'meta/bundle.env').write_text(text)
  names=['compile_bundle.py','bundle_consumer.py','bundle_consumer.sh']
  for n in names:shutil.copy2(source/'scripts/release'/n,b/'tools'/n)
  (b/'tools/SHA256SUMS').write_text(''.join(f'{sha(b/"tools"/n)}  {n}\n' for n in names))
  bundles[k]=dict(dir=str(b),src=str(source))
 old=Path('<DATA_ROOT>/wrf_gpu2/scratch/w44-rc/rc_bundle_3e5219bc8088');oldman=old/'sealed/BUNDLE_MANIFEST.json';olddata=json.loads(oldman.read_text())
 seed=T/'fresh_autotune.sha256';assert seed.read_bytes()==b''
 floor=source/'docs/release/evidence/rw3/wn3_0227_full_floor.json';metrics=source/'docs/release/evidence/v033/w2/W2bR/metrics.json';assert floor.is_file() and metrics.is_file()
 man=dict(schema='v45-rc.pin-manifest.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),standin=a.standin,pin=pin,src_tree=src,gpuwrf_tree=git('rev-parse',pin+':src/gpuwrf'),version=ver,pyproject_version=pver,main_src_tree_at_prepare=main_src,allow_src_diverge=a.allow_src_diverge,repo=str(REPO),snapshot=str(source),bundles=bundles,old_bundle=dict(dir=str(old),manifest_sha256=sha(oldman),bundle_sha256=olddata.get('bundle_sha256')),cases=dict(dir=str(c),sums_file=str(cs),sums_sha256=sha(cs)),inputs=inputs,input_metadata=metadata,input_sha256={k:{n:sha(Path(inputs[k])/n) for n in ['namelist.input','wrfinput_d01']} for k in ['SWISS','PROD']},autotune_seed=dict(source='EMPTY; no old paid cache reused',list=str(seed),list_sha256=sha(seed)),tools_dir=str(T),tools_sha256sums_sha256=sha(T/'SHA256SUMS'),rw3_floor=dict(path=str(floor),sha256=sha(floor),metrics=str(metrics),metrics_sha256=sha(metrics)),resource_authority='manager MAIN MAILBOX 2026-10-28T12:47:13Z; carried v0.4.4 CPU admission authority unchanged in kind',root_cap_gib=8,root_min_free_gib=20,data_peak_gib=31,data_floor_gib=365,optional_o72_min_free_gib=411)
 write(manpath,man)
 subprocess.run(['<USER_HOME>/miniconda3/bin/python3','-I',str(T/'prereg_v044.py'),str(manpath),str(mdir/'PREREG.json')],check=True)
 for b in bundles.values():
  for n in ['PIN_MANIFEST.json','PREREG.json']:shutil.copy2(mdir/n,Path(b['dir'])/'meta'/n)
 print('PREPARE_OK',manpath,sha(manpath),'EMPTY caches; DATA writes=0')
if __name__=='__main__':main()
