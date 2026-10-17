import subprocess,shutil,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('<USER_HOME>/wrf_gpu2_lanes/rv-t-classd/snapshot')
OUT=ROOT.parent
WRF=Path('<USER_HOME>/src/wrf_pristine/WRF')
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
records={}
reference=np.load(ROOT/'proofs/t_classd/morr_radar_oracle/fixtures/oracle.npz')
manifest=json.loads((ROOT/'proofs/t_classd/morr_radar_oracle/fixtures/manifest.json').read_text())
for mode in ('fp32','fp64'):
    for variant in ('morr','morr_aero'):
        arm=mode+'_'+variant
        build=OUT/'oracle_build'/arm;build.mkdir(parents=True,exist_ok=False)
        module='module_mp_morr_two_moment'+('_aero' if variant=='morr_aero' else '')+'.F'
        sources=[WRF/'share/module_model_constants.F',WRF/'phys/module_mp_radar.F',WRF/'phys'/module,
                 ROOT/'proofs/v060/oracle/morrison_stub_modules.f90',ROOT/'proofs/t_classd/morr_radar_oracle/morr_radar_driver.F90']
        for path in sources:shutil.copyfile(path,build/path.name)
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        for line in manifest['arms'][arm]['sources']:
            if line:
                digest,name=line.split();assert hashes[name]==digest,(arm,name)
        flags=['-O2','-ffree-form','-ffree-line-length-none','-ffpe-summary=none','-cpp']
        if variant=='morr': flags+=['-finit-real=zero','-finit-integer=0','-finit-logical=false']
        if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
        order=['morrison_stub_modules.f90','module_model_constants.F','module_mp_radar.F',module,'morr_radar_driver.F90']
        with (build/'compile.log').open('w') as log:
            for name in order:
                extra=['-DAERO'] if name=='morr_radar_driver.F90' and variant=='morr_aero' else []
                subprocess.run([fc,*flags,*extra,'-c',name],cwd=build,stdout=log,stderr=log,check=True,timeout=120)
            objects=[str(Path(name).with_suffix('.o')) for name in order]
            subprocess.run([fc,*flags,'-o','oracle',*objects],cwd=build,stdout=log,stderr=log,check=True,timeout=120)
        result=subprocess.run([str(build/'oracle'),str(OUT/f'input_{mode}.bin')],cwd=build,stdout=subprocess.PIPE,check=True,timeout=30,text=True)
        lines=result.stdout.splitlines()
        dtype=np.float32 if mode=='fp32' else np.float64
        ints=np.array([int(l.split()[3],16) for l in lines if l.startswith('DBZ')],np.uint32 if mode=='fp32' else np.uint64)
        values=ints.view(dtype).reshape(reference[arm].shape)
        records[arm]=dict(sources=hashes,levels=values.size,byte_equal=values.tobytes()==reference[arm].tobytes(),
                          values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),flags=flags)
        assert records[arm]['byte_equal'],arm
        print(arm,records[arm]['levels'],'BYTE',flush=True)
(OUT/'RV07_fresh_oracle.json').write_text(json.dumps(records,indent=2)+'\n')
