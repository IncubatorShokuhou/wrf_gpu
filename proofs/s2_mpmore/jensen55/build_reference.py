"""Original WRF ISHMAEL caller and columns; no port or operational admission.

The frozen NSSL atmospheric regime inputs provide pressure/theta/vapor and
condensate profiles. Their different scheme outputs are never references for
ISHMAEL. Every reference output here comes from the unchanged Jensen module.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np

SOURCE = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F')
TABLES = Path('<USER_HOME>/src/wrf_pristine/WRF/run')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
NAMES = ('p','dz','th','qv','qc','qr','nr','qi1','ni1','ai1','ci1',
         'qi2','ni2','ai2','ci2','qi3','ni3','ai3','ci3','effc','effi','dbz',
         'vm1','di1','rho1','phi1','vm2','di2','rho2','phi2',
         'vm3','di3','rho3','phi3','type1','type2','type3')
TABLE_FILES = ('ishmael-qi-qc.bin','ishmael-qi-qr.bin','ishmael-gamma-tab.bin')


def caller():
    declarations = ','.join(name+'(1,nz,1)' for name in NAMES)
    prognostic = ','.join(NAMES[:19])
    output = ','.join(NAMES)
    return f'''program jensen_original_caller
 use module_mp_jensen_ishmael, only: jensen_ishmael_init,mp_jensen_ishmael
 implicit none
 integer :: nz,it,unit,case_index
 character(64) :: filename
 real :: dt
 real,allocatable :: {','.join(name+'(:,:,:)' for name in NAMES)}
 real :: rain(1,1),rainstep(1,1),snow(1,1),snowstep(1,1)
 call jensen_ishmael_init()
 do case_index=1,17
 write(filename,'("input_",I0,".bin")') case_index
 open(newunit=unit,file=filename,access='stream',form='unformatted',status='old')
 read(unit) nz,it
 read(unit) dt
 allocate({declarations})
 read(unit) {prognostic}
 close(unit)
 {chr(10).join(name+'=0.' for name in NAMES[19:])}
 rain=0.; rainstep=0.; snow=0.; snowstep=0.
 call mp_jensen_ishmael(it,dt,{prognostic}, &
  1,1,1,1,1,nz,1,1,1,1,1,nz,1,1,1,1,1,nz, &
  rain,rainstep,snow,snowstep, &
  effc,effi,dbz,vm1,di1,rho1,phi1, &
  vm2,di2,rho2,phi2,vm3,di3,rho3,phi3,type1,type2,type3)
 write(filename,'("output_",I0,".bin")') case_index
 open(newunit=unit,file=filename,access='stream',form='unformatted',status='replace')
 write(unit) {output},rain,rainstep,snow,snowstep
 close(unit)
 deallocate({','.join(NAMES)})
 enddo
end program
'''


def input_fields(reference, dtype):
    """Use original atmospheric profiles; seed valid spherical NU=4 ice moments."""
    c = reference['columns']
    f = {name:np.asarray(c[wrf],dtype=dtype) for name,wrf in
         [('p','P'),('dz','DZ'),('th','TH_IN'),('qv','QV_IN'),
          ('qc','QC_IN'),('qr','QR_IN')]}
    f['nr'] = np.asarray(f['qr']/dtype(2.e-8),dtype=dtype)
    # These are input prescriptions, not alternative microphysics equations.
    # A/CI are characteristic dimension moments for a gamma shape NU=4.
    for index,mass,density in [(1,c['QI_IN'],920.),(2,c['QS_IN'],920.),
                               (3,np.asarray(c['QH_IN'])+np.asarray(c['QHL_IN']),50.)]:
        qi=np.asarray(mass,dtype=dtype)
        f[f'qi{index}']=qi
        f[f'ni{index}']=np.asarray(qi/dtype(1.e-10),dtype=dtype)
        volume=np.asarray(qi/(dtype(density)*dtype(4./3.)*dtype(3.14159265)*dtype(120.)),dtype=dtype)
        f[f'ai{index}']=volume.copy();f[f'ci{index}']=volume.copy()
    return f


def main(tree, artifact):
    assert (tree/'.git').is_file(), 'Use an immutable detached worktree'
    if artifact.exists():
        raise SystemExit('Use a fresh artifact directory')
    build=artifact/'build';out=artifact/'fixtures'
    build.mkdir(parents=True);out.mkdir()
    source=SOURCE.read_bytes();driver=caller()
    rows=[];builds=[]
    for mode,dtype in [('fp32',np.float32),('fp64',np.float64)]:
        b=build/mode;b.mkdir()
        (b/'module.F').write_bytes(source)
        (b/'stubs.f90').write_text('module module_wrf_error\nend module\n')
        (b/'driver.F90').write_text(driver)
        for file in TABLE_FILES:
            (b/file).symlink_to(TABLES/file)
        flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp',
               '-finit-real=zero','-finit-integer=0','-finit-logical=false','-ffp-contract=off',
               '-fconvert=big-endian','-frecord-marker=4']  # pristine configure.wrf BYTESWAPIO
        if mode=='fp64':
            flags+=['-fdefault-real-8','-fdefault-double-8']
        for file in ('stubs.f90','module.F','driver.F90'):
            subprocess.run([FC,*flags,'-c',file],cwd=b,check=True,timeout=180)
        subprocess.run([FC,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=b,check=True,timeout=180)
        builds.append(dict(mode=mode,flags=flags,module_sha256=hashlib.sha256((b/'module.F').read_bytes()).hexdigest(),
                           module_object_sha256=hashlib.sha256((b/'module.o').read_bytes()).hexdigest(),
                           driver_sha256=hashlib.sha256((b/'driver.F90').read_bytes()).hexdigest()))
        inputs=[]
        io_dtype=np.dtype(dtype).newbyteorder('>')
        for case in range(1,18):
            refpath=tree/f'proofs/s2_mpmore/nssl1mom/fixtures/v21_{mode}_{case}.json'
            ref=json.loads(refpath.read_text());fields=input_fields(ref,dtype)
            nz=len(fields['th']);dt=dtype(ref['scalars']['DT'])
            it=int(ref['scalars']['ITIMESTEP'])
            raw=np.asarray([nz,it],dtype='>i4').tobytes()+np.asarray(dt,dtype=io_dtype).tobytes()
            raw+=b''.join(np.asarray(fields[name],dtype=io_dtype).tobytes() for name in NAMES[:19])
            (b/f'input_{case}.bin').write_bytes(raw)
            inputs.append((case,refpath,fields,nz,dt,it,raw))
        run=subprocess.run([str(b/'oracle')],cwd=b,timeout=240,capture_output=True)
        (b/'runtime.stdout').write_bytes(run.stdout);(b/'runtime.stderr').write_bytes(run.stderr)
        if run.returncode:
            print(run.stdout.decode(errors='replace'),run.stderr.decode(errors='replace'),flush=True)
            run.check_returncode()
        for case,refpath,fields,nz,dt,it,raw in inputs:
            values=np.frombuffer((b/f'output_{case}.bin').read_bytes(),dtype=io_dtype)
            assert len(values)==len(NAMES)*nz+4,(mode,case,len(values),nz)
            columns={name:values[index*nz:(index+1)*nz].tolist() for index,name in enumerate(NAMES)}
            scalars=dict(zip(('rain','rainstep','snow','snowstep'),map(float,values[-4:])))
            finite=bool(np.isfinite(values).all())
            record=dict(mode=mode,case=case,dt=float(dt),it=it,input={k:v.tolist() for k,v in fields.items()},
                        output=columns,precipitation=scalars,finite=finite)
            path=out/f'{mode}_{case}.json';path.write_text(json.dumps(record,indent=2)+'\n')
            rows.append(dict(mode=mode,case=case,finite=finite,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             input_sha256=hashlib.sha256(raw).hexdigest(),
                             atmospheric_input_source_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
                             stdout_sha256=hashlib.sha256(run.stdout).hexdigest()))
    manifest=dict(scope='Unchanged original Jensen full public caller; independent outputs, input-only reuse of atmospheric regime profiles',
                  source_sha256=hashlib.sha256(source).hexdigest(),caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),
                  tables={name:hashlib.sha256((TABLES/name).read_bytes()).hexdigest() for name in TABLE_FILES},
                  builds=builds,rows=rows)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dict(columns=len(rows),finite=sum(row['finite'] for row in rows)),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
