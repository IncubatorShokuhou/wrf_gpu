"""t-cfix-pblmix F17: unchanged WRF 4.7.1 GBMPBL (REAL4) on real c08 columns WITH RTHRATEN.

Lane copy of proofs/t_c02nan/build_gbm_oracle.py (same flags: -O0, no contraction,
compiler-defined zero init per the standing uninitialized-read policy -- GBM2D reads
kpbl2dx uninitialized), extended to many columns and the RTHRATEN operand
(module_pbl_driver passes RTHRATEN=grid%rthraten; gbmpbl.F:243 t2dten_ra=RTHRATEN*PI3D).
Usage: build_gbm_rthraten_oracle.py <build_dir> <wrfout[,...]> <ncol/file> [--control zero|snan]
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(); p.add_argument('build', type=Path); p.add_argument('wrfout'); p.add_argument('ncol', type=int)
    p.add_argument('--control', choices=('zero', 'snan'), default='zero')
    a = p.parse_args(); a.build.mkdir(parents=True, exist_ok=True)
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    wrf = Path('<USER_HOME>/src/wrf_pristine/WRF')
    fc = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags = ['-O0', '-ffp-contract=off', '-ffree-form', '-ffree-line-length-none', '-cpp',
             '-finit-real=zero', '-finit-integer=0', '-finit-logical=false', '-ffpe-summary=none']
    if a.control == 'snan':
        flags = [s.replace('real=zero', 'real=snan').replace('integer=0', 'integer=-777').replace('logical=false', 'logical=true') for s in flags]
    subprocess.run(['python3', str(HERE / 'gbm_columns.py'), a.wrfout, str(a.ncol), str(a.build / 'cols.txt'),
                    str(a.build / 'inputs.json')], check=True, cwd=HERE)
    hashes = {}
    for name in ('share/module_model_constants.F', 'phys/module_bl_gbmpbl.F'):
        source = wrf / name; hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copyfile(source, a.build / source.name)
        subprocess.run([fc, *flags, '-c', source.name], cwd=a.build, check=True, timeout=120)
    arrays = ['u', 'v', 'th', 't', 'qv', 'qc', 'qi', 'p', 'pii', 'dz', 'z', 'tke', 'rad', 'ru', 'rv', 'rth', 'rqv',
              'rqc', 'rqi', 'kzm', 'kth', 'kethl', 'el']
    scalars = ['psfc', 'znt', 'ust', 'hfx', 'qfx', 'tsk', 'gz', 'wspd', 'br', 'psim', 'psih', 'xland', 'zol', 'hol',
               'pbl', 'regime']
    code = ['program gbm_rad_oracle', 'use module_bl_gbmpbl, only: gbmpbl', 'implicit none',
            'integer::n,ncol,c,k,kpbl(1,1)', 'real::dt=18.',
            'real,allocatable::' + ','.join(f'{x}(:,:,:)' for x in arrays),
            'real::' + ','.join(f'{x}(1,1)' for x in scalars),
            'open(42,file="cols.txt",status="old")', 'read(42,*)ncol,n',
            *[f'allocate({x}(1,n+1,1))' for x in arrays],
            'do c=1,ncol', *[f'{x}=0.' for x in arrays + scalars], 'kpbl=0']
    code += [f'read(42,*){x}(1,1:n,1)' for x in ['u', 'v', 't', 'qv', 'qc', 'p', 'pii', 'dz', 'tke', 'rad']]
    code += ['read(42,*)' + ','.join(f'{x}(1,1)' for x in scalars[:12])]
    code += ['th(1,1:n,1)=t(1,1:n,1)/pii(1,1:n,1)', 'z(1,1,1)=.5*dz(1,1,1)',
             'do k=2,n', 'z(1,k,1)=z(1,k-1,1)+.5*(dz(1,k-1,1)+dz(1,k,1))', 'enddo',
             'call gbmpbl(u,v,th,t,qv,qc,qi,p,pii,ru,rv,rth,rqv,rqc,rqi,kzm,kth,kethl,el, &',
             ' dz,z,psfc,tke,rad,znt,ust,zol,hol,pbl,kpbl,regime,psim,psih,xland,hfx,qfx,tsk,gz,wspd,br, &',
             ' dt,dt/60.,1,2,1,2,1,n+1,1,1,1,1,1,n+1,1,1,1,1,1,n)', 'do k=1,n']
    for label, name in [('u', 'ru'), ('v', 'rv'), ('theta', 'rth'), ('qv', 'rqv'), ('qc', 'rqc'), ('tke', 'tke'),
                        ('kzm', 'kzm'), ('kth', 'kth'), ('el_pbl', 'el'), ('kethl', 'kethl')]:
        code.append(f'write(*,\'(A,I0,1X,I0,1X,ES26.17E3)\') "{label} ",c,k,{name}(1,k,1)')
    code += ['enddo', 'write(*,\'(A,I0,1X,ES26.17E3,1X,I0,1X,ES26.17E3)\') "pbl ",c,pbl(1,1),kpbl(1,1),ust(1,1)',
             'enddo', 'end program']
    driver = a.build / 'driver.f90'; driver.write_text('\n'.join(code) + '\n')
    subprocess.run([fc, *flags, '-o', 'gbm_oracle', driver.name, 'module_model_constants.o', 'module_bl_gbmpbl.o'],
                   cwd=a.build, check=True, timeout=120)
    out = subprocess.run(['./gbm_oracle'], cwd=a.build, check=True, capture_output=True, text=True).stdout
    (a.build / f'out_{a.control}.txt').write_text(out)
    # Discrimination control: same columns with RTHRATEN=0 (the pre-fix port operand).
    lines = (a.build / 'cols.txt').read_text().split('\n'); ncol = int(lines[0].split()[0]); rows, i = [lines[0]], 1
    for _ in range(ncol):
        block = lines[i:i + 11]; block[9] = ' '.join('0.0' for _ in block[9].split()); rows += block; i += 11
    (a.build / 'norad').mkdir(exist_ok=True); (a.build / 'norad' / 'cols.txt').write_text('\n'.join(rows) + '\n')
    norad = subprocess.run(['../gbm_oracle'], cwd=a.build / 'norad', check=True, capture_output=True, text=True).stdout
    (a.build / 'norad' / 'out.txt').write_text(norad)
    # DOUBLE transcription build of the same driver/columns (out in r8/out.txt).
    r8 = a.build / 'r8'; r8.mkdir(exist_ok=True)
    dflags = flags + ['-fdefault-real-8', '-fdefault-double-8']
    for name in ('module_model_constants.F', 'module_bl_gbmpbl.F', 'driver.f90'):
        shutil.copyfile(a.build / name, r8 / name)
        subprocess.run([fc, *dflags, '-c', name], cwd=r8, check=True, timeout=120)
    subprocess.run([fc, *dflags, '-o', 'gbm_oracle', 'driver.o', 'module_model_constants.o', 'module_bl_gbmpbl.o'],
                   cwd=r8, check=True, timeout=120)
    shutil.copyfile(a.build / 'cols.txt', r8 / 'cols.txt')
    (r8 / 'out.txt').write_text(subprocess.run(['./gbm_oracle'], cwd=r8, check=True, capture_output=True, text=True).stdout)
    subprocess.run(['python3', str(HERE / 'make_gbm_fixture.py'), str(a.build), str(HERE / 'gbm_rthraten_fixture.json')],
                   check=True)
    manifest = dict(source_sha256=hashes, flags=flags, compiler=fc, control=a.control,
                    driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
                    cols_sha256=hashlib.sha256((a.build / 'cols.txt').read_bytes()).hexdigest(),
                    out_sha256=hashlib.sha256(out.encode()).hexdigest())
    (a.build / f'build_{a.control}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
