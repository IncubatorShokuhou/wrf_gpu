"""Remove only this completed arm's disposable overlays, scratch and NetCDFs."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys

root=Path(sys.argv[1])
assert root.parent==Path('<USER_HOME>/wrf_gpu2_lanes/w44-perf/launch_v5_runs') and root.is_dir()
assert os.environ['JAX_PLATFORMS']=='cpu'
if Path('/tmp/wrf_gpu2_quiet').exists():
    token=root/'quiet_token'
    if token.is_file() and token.read_bytes()==Path('/tmp/wrf_gpu2_quiet').read_bytes():
        Path('/tmp/wrf_gpu2_quiet').unlink()
    else:raise SystemExit('QUIET: retain exact declared run paths until cleanup can resume')
removed=[]
for version in ['v032','v040','v043']:
    path=root/('layer_'+version)
    if not path.exists():continue
    assert not path.is_symlink()
    # overlayfs leaves work/work at mode 000: open each child BEFORE os.walk lists it.
    os.chmod(path,stat.S_IMODE(os.stat(path).st_mode)|0o700)
    for directory,dirs,files in os.walk(path,topdown=True):
        for name in dirs:
            child=os.path.join(directory,name)
            if not os.path.islink(child):
                os.chmod(child,stat.S_IMODE(os.lstat(child).st_mode)|0o700)
    shutil.rmtree(path);removed.append(str(path))
outputs=[]
for run in (root/'runs').glob('*'):
    for path in (run/'run/wrfout').glob('wrfout_d0*'):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as stream:
                while data:=stream.read(1<<20):h.update(data)
            outputs.append({'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()});path.unlink()
    path=run/'run/scratch'
    if path.is_dir():shutil.rmtree(path);removed.append(str(path))
(root/'CLEANUP.json').write_text(json.dumps({'removed_exact_directories':removed,'removed_outputs':outputs,'gpu_tree_reaped_before_cleanup':True},indent=2)+'\n')
