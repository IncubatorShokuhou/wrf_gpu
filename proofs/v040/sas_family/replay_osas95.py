"""Replay the recorded original XKT2 at the RNG boundary; original physics unchanged.

The first control MUST reproduce SS40's complete stdout BYTE before interpreting
any phase or compiler control. We never claim independent SYSTEM_CLOCK reruns
are reproducible. Caller supplies CPU/QUIET/nice/timeout.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess


def run(reference,work,mode,fma=False):
    work.mkdir(parents=True,exist_ok=True)
    tmp=work/'tmp';tmp.mkdir(exist_ok=True)
    env=dict(os.environ,TMPDIR=str(tmp))
    b=work/'build';b.mkdir(exist_ok=True)
    ref=reference/f'build_{mode}'
    for name in ('module_gfs_machine','module_gfs_physcons','module_gfs_funcphys'):
        for suffix in ('.mod','.o'):shutil.copyfile(ref/(name+suffix),b/(name+suffix))
    source=(ref/'module_cu_osas.F').read_text()
    anchor='     call random_number(XKT2)'
    assert len(re.findall(r'^'+re.escape(anchor)+r'$',source,re.M))==1
    replay=source.replace(anchor,anchor+"\n     open(unit=96,file='replay_draws.txt',status='old')\n     read(96,*) XKT2\n     close(96)")
    (b/'module_cu_osas.F').write_text(replay)
    shutil.copyfile(ref/'osas95_oracle.F90',b/'osas95_oracle.F90')
    flags=['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1','-DNMM_CORE=0']
    if mode=='r8':flags+=['-fdefault-real-8','-fdefault-double-8']
    if fma:flags+=['-march=native','-ffp-contract=fast']
    c='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    for args in (flags+['-c','module_cu_osas.F'],flags+['-c','osas95_oracle.F90'],
                 ['-O2','-o','oracle','module_gfs_machine.o','module_gfs_physcons.o','module_gfs_funcphys.o','module_cu_osas.o','osas95_oracle.o']):
        subprocess.run([c]+args,cwd=b,env=env,check=True)
    report={'mode':mode,'fma':fma,'flags':flags,'recorded_draw_replay':True,
            'physics_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'replay_source_sha256':hashlib.sha256(replay.encode()).hexdigest(),'arms':{}}
    manifest=json.loads((reference/'manifest.json').read_text())
    inputs=Path('<USER_HOME>/wrf_gpu2_lanes/s2-sas/SS01')
    for dt,stepcu in ((54,5),(18,1)):
        stem=f'dt{dt}_stepcu{stepcu}_shallow1'
        raw=(reference/f'{stem}_{mode}.out').read_bytes()
        draw=raw.splitlines()[0].removeprefix(b'DRAW_XKT2 ')
        (b/'replay_draws.txt').write_bytes(draw+b'\n')
        # Authenticate the original input arm from the SS40 manifest.
        matches=[p for p in inputs.rglob(stem+'.txt')]
        assert len(matches)==1, matches
        stdin=matches[0].read_bytes()
        assert hashlib.sha256(stdin).hexdigest()==manifest['arms'][f'{stem}_{mode}']['input_sha256']
        out=subprocess.check_output([str(b/'oracle')],input=stdin,cwd=b,env=env)
        (work/f'{stem}_{mode}.out').write_bytes(out)
        same=out==raw
        report['arms'][stem]={'stdout_byte':same,'output_sha256':hashlib.sha256(out).hexdigest()}
        if not fma:assert same, 'recorded-draw replay must reproduce pristine SS40 stdout'
    (work/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('work',type=Path)
    p.add_argument('--mode',choices=('r4','r8'),default='r8');p.add_argument('--fma',action='store_true')
    a=p.parse_args();run(a.reference.resolve(),a.work.resolve(),a.mode,a.fma)
