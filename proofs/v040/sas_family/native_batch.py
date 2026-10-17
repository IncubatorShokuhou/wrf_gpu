"""Broader native SAS gate on the independently constructed 455-column corpus.

build: native_batch.py build WORK [14|96]
probe: native_batch.py probe WORK SCHEME MODE
One shared i-row in the pristine driver, exactly row_width=N in NSAS.
"""
from pathlib import Path
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
KEYS=('T','QV','QC','QI','P','PI','RHO','DZ','U','V','W')
OUTPUTS=('RAINCV','PRATEC','HBOT','HTOP','RTHCUTEN','RQVCUTEN','RQCCUTEN','RQICUTEN','RUCUTEN','RVCUTEN')


def build(work,scheme):
    work.mkdir(parents=True,exist_ok=True)
    tmp=work/'tmp';tmp.mkdir(exist_ok=True)
    env=dict(os.environ,TMPDIR=str(tmp))
    name='ksas' if scheme==14 else 'nsas'
    source=Path(f'<USER_HOME>/src/wrf_pristine/WRF/phys/module_cu_{name}.F')
    input_path=HERE.parents[1]/'v034/scalesas/savepoints/columns_inputs.npz'
    with np.load(input_path) as raw: inp=dict(raw)
    n,kx=inp['T'].shape
    inp.update(HPBL=np.full(n,750.),HPBL_HOLD=np.full(n,650.),
               HFX=np.resize(np.array([-100.,0.,100.,300.]),n),
               QFX=np.resize(np.array([0.,0.,1.e-4,3.e-4]),n),
               ZNU=.5*(np.linspace(1.,0.,kx+1)[:-1]+np.linspace(1.,0.,kx+1)[1:]))
    # One static case contains the full 455 original sounding variations and
    # positive/nonpositive shallow surface fluxes, mixed land and sea columns.
    dt,stepcu,mp,dx,pgcon,dxfactor=18.,1,8,9000.,.55,0
    np.savez_compressed(work/'inputs.npz',**inp)
    manifest={'input_original_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
              'inputs_sha256':hashlib.sha256((work/'inputs.npz').read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'work':dict(N=n,KX=kx,DT=dt,STEPCU=stepcu,MP=mp,DX=dx,PGCON=pgcon,DX_FACTOR_NSAS=dxfactor),
              'arms':{}}
    for mode in ('r4','r8'):
        b=work/f'build_{mode}';b.mkdir(exist_ok=True)
        shutil.copyfile(source,b/source.name)
        driver=f'{name}{scheme}_columns.F90'
        shutil.copyfile(HERE/driver,b/driver)
        flags=['-O2','-ftree-vectorize','-funroll-loops','-w','-ffree-form','-ffree-line-length-none','-cpp',
               '-DEM_CORE=1','-DNMM_CORE=0','-finit-real=zero','-finit-integer=0','-finit-logical=false']
        if mode=='r8':flags+=['-fdefault-real-8','-fdefault-double-8']
        compiler='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
        for args in (flags+['-c',source.name],flags+['-c',driver],
                     ['-O2','-o','columns',f'module_cu_{name}.o',f'{name}{scheme}_columns.o']):
            subprocess.run([compiler]+args,cwd=b,env=env,check=True)
        header=f'{dt} {stepcu} {mp} {dx}' + (f' {pgcon} {dxfactor}' if scheme==96 else '')
        fmt=lambda a:' '.join(format(float(x),'.17g') for x in np.asarray(a).flat)
        rows=[f'{n} {kx}',header,fmt(inp['ZNU'])]
        for i in range(n):
            rows.append(fmt([inp[key][i] for key in
                        (('XLAND','HPBL','HPBL_HOLD') if scheme==14 else ('XLAND','HPBL','HFX','QFX'))]))
            rows += [fmt(inp[key][i]) for key in ('T','QV','QC','QI','P','PI','RHO','DZ','U','V','P8W','W')]
        stdin='\n'.join(rows)+'\n'
        (work/f'input_{mode}.txt').write_text(stdin)
        result=subprocess.run([str(b/'columns')],input=stdin.encode(),cwd=b,env=env,check=True,capture_output=True)
        (work/f'output_{mode}.txt').write_bytes(result.stdout)
        data=np.fromstring(result.stdout.decode(),sep=' ').reshape(n,5+6*kx)
        expected={key:data[:,i+1] for i,key in enumerate(OUTPUTS[:4])}
        expected.update({key:data[:,5+i*kx:5+(i+1)*kx] for i,key in enumerate(OUTPUTS[4:])})
        np.savez_compressed(work/f'oracle_{mode}.npz',**expected)
        manifest['arms'][mode]={'flags':flags,'active':int(np.count_nonzero(expected['RAINCV'])),
                                'finite':all(np.isfinite(a).all() for a in expected.values()),
                                'oracle_sha256':hashlib.sha256((work/f'oracle_{mode}.npz').read_bytes()).hexdigest()}
        (work/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(scheme,mode,manifest['arms'][mode],flush=True)


def probe(work,scheme,mode):
    import jax
    from gpuwrf.physics.cumulus_ksas14 import cu_ksas14_columns
    from gpuwrf.physics.cumulus_nsas96 import cu_nsas96_columns
    from probe_ksas14 import compare_fields
    from evidence_gate import require_finite
    jax.config.update('jax_enable_x64',True)
    assert jax.devices()[0].platform=='cpu'
    with np.load(work/'inputs.npz') as raw:inp=dict(raw)
    with np.load(work/f'oracle_{mode}.npz') as raw:expected=dict(raw)
    cfg=json.loads((work/'manifest.json').read_text())['work'];n=cfg['N']
    require_finite(**{k:v for k,v in inp.items() if v.dtype.kind in 'fiu'}, **cfg)
    opts=dict(dt=cfg['DT'],stepcu=cfg['STEPCU'],dx=cfg['DX'],mp_physics=cfg['MP'],literals=mode)
    if scheme==14:
        fn=functools.partial(cu_ksas14_columns,**opts)
        args=[inp[k] for k in KEYS+('XLAND','HPBL','HPBL_HOLD','ZNU')]
    else:
        fn=functools.partial(cu_nsas96_columns,row_width=n,pgcon=cfg['PGCON'],dx_factor_nsas=cfg['DX_FACTOR_NSAS'],**opts)
        args=[inp[k] for k in KEYS+('P8W','XLAND','HPBL','HFX','QFX')]
    start=time.monotonic();exe=jax.jit(fn).lower(*args).compile();compile_s=time.monotonic()-start
    actual={k:np.asarray(v) for k,v in exe(*args).items()}
    samples=[]
    for _ in range(5):
        t=time.monotonic();jax.block_until_ready(exe(*args));samples.append(time.monotonic()-t)
    fields=compare_fields(actual,{**inp,**expected,'DT':np.full(n,cfg['DT']),
                                  'STEPCU':np.full(n,cfg['STEPCU'])},mode)
    memory=exe.memory_analysis()
    out={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'scheme':scheme,'mode':mode,'fields':fields,'all_pass':all(f['pass'] for f in fields.values()),
         'oracle_active':int(np.count_nonzero(expected['RAINCV'])),'port_active':int(np.count_nonzero(actual['RAINCV'])),
         'compile_s':compile_s,'warm_median_s':float(np.median(samples)),
         'argument_bytes':memory.argument_size_in_bytes,'output_bytes':memory.output_size_in_bytes,
         'temp_bytes':memory.temp_size_in_bytes,'platform':'cpu'}
    np.savez_compressed(work/f'candidate_{mode}.npz',**actual)
    (work/f'comparison_{mode}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':
    action,work,scheme=sys.argv[1:4]
    if action=='build':build(Path(work),int(scheme))
    else:probe(Path(work),int(scheme),sys.argv[4])
