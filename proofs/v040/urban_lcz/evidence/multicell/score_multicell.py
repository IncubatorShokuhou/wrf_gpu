from pathlib import Path
import hashlib,json,os,shutil,subprocess
import numpy as np

root=Path('<USER_HOME>/wrf_gpu2_lanes/rv-u2-lcz')
build=root/'multicell_build/fp32'
evidence=root/'multicell_evidence'
evidence.mkdir(exist_ok=True)
for name in ('driver.f90','caller.F'):
    shutil.copy2(build/name,evidence/name)
for name in ('manifest.json','slucm_fp32_default.npz'):
    shutil.copy2(root/'multicell_build'/name,evidence/name)
table=root/'snapshot/proofs/v040/urban_lcz/savepoints/slucm_fp32_default.npz'
z=np.load(evidence/'slucm_fp32_default.npz'); ref=np.load(table)
fields={'HFX':'HFX','TSK':'TSK','U10_URB2D':'U10_URB','V10_URB2D':'V10_URB','UST':'UST'}
result={'identical_cell_inputs':True,'source_bodies':'unmodified pristine WRF4.7.1',
        'compiler_init':'zero real/integer, false logical','tile1_vs_frozen':{},'tile1_vs_tile2':{}}
for extra,base in fields.items():
    result['tile1_vs_frozen'][base]=float(np.max(np.abs(z[base][:,1:]-ref[base][:,1:])))
    d=np.abs(z['TILE2_'+extra][:,1:]-z[base][:,1:])
    result['tile1_vs_tile2'][base]={'max_abs':float(d.max()),'first_step_max_abs':float(d[:,0].max()),
                                  'different':int(np.count_nonzero(d)), 'nonfinite':int((~np.isfinite(d)).sum())}
source=(build/'driver.f90').read_text()
old='(/ 0.1, 0.3, 0.5, 0.7 /)'
assert source.count(old)==1
isotropic=source.replace(old,'(/ 0.1, 0.1, 0.1, 0.1 /)')
(build/'driver_isotropic.f90').write_text(isotropic)
flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-DEM_CORE=1','-ffpe-summary=none',
       '-fallow-argument-mismatch','-finit-integer=0','-finit-real=zero','-finit-logical=false']
fc='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
(build/'compiler_tmp').mkdir(exist_ok=True)
env=dict(os.environ,TMPDIR=str(build/'compiler_tmp'))
subprocess.run([fc,*flags,'-c','driver_isotropic.f90'],cwd=build,env=env,check=True,timeout=30)
objects=[p.name for p in build.glob('*.o') if p.name not in ('driver.o','driver_isotropic.o')]
subprocess.run([fc,*flags,'-o','isotropic','driver_isotropic.o',*objects],cwd=build,env=env,check=True,timeout=30)
with (evidence/'isotropic.txt').open('w') as f:
    subprocess.run([str(build/'isotropic'),'fp32',str(root/'snapshot/proofs/v040/urban_slucm/regimes.txt')],
                   cwd=build,env=env,stdout=f,check=True,timeout=30)
subprocess.run([os.sys.executable,str(root/'snapshot/proofs/v040/urban_slucm/dump_to_npz.py'),
               str(evidence/'isotropic.txt'),str(evidence/'isotropic.npz')],env=env,check=True,timeout=30)
iso=np.load(evidence/'isotropic.npz')
result['isotropic_control']={base:float(np.max(np.abs(iso['TILE2_'+extra][:,1:]-iso[base][:,1:])))
                             for extra,base in fields.items()}
result['evidence_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.iterdir()}
(root/'RV03-multicell-final.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert all(v==0 for v in result['tile1_vs_frozen'].values())
assert all(v==0 for v in result['isotropic_control'].values())
assert result['tile1_vs_tile2']['HFX']['max_abs']>1
