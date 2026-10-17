"""Build unmodified WRF OML1D in a private lane directory; freeze REAL columns.

Run with JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 and lane CPU affinity. No WRF
source bodies are patched. The only stubs are WRF logging routines.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path('<USER_HOME>/wrf_gpu2_lanes/s2-oml/oracle_build')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
WRF = ROOT / 'data/wrf_pristine/WRF'


def main():
    if Path('/tmp/wrf_gpu2_quiet').exists():
        raise SystemExit('QUIET')
    BUILD.mkdir(parents=True, exist_ok=True)
    source = WRF / 'phys/module_sf_oml.F'
    shutil.copy2(source, BUILD / source.name)
    driver = '''program oracle
use module_sf_oml
implicit none
real :: x(21), tsk
integer :: n, i, k, steps
read(*,*) n
do i=1,n
 read(*,*) x,steps
 tsk=x(1)
 do k=1,steps
  call OML1D(1,1,x(1),x(2),x(3),x(4),x(5),x(6),tsk, &
   x(8),x(9),x(10),x(11),x(7),x(12),x(13),x(14),x(15),x(16), &
   5.67051e-8,9.81,x(17),x(18),x(19), &
   1,2,1,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1)
 enddo
 write(*,'(7(ES25.16E3,1x))') x(1:7)
enddo
end program
subroutine wrf_debug(level,msg)
integer :: level
character(*) :: msg
end subroutine
'''
    (BUILD/'driver.f90').write_text(driver)
    flags = ['-O0', '-ffp-contract=off', '-ffree-form', '-ffree-line-length-none',
             '-finit-real=zero', '-finit-integer=0', '-finit-logical=false']
    subprocess.run([FC, *flags, source.name, 'driver.f90', '-o', 'oracle'], cwd=BUILD, check=True)
    # Explicit active regimes: hurricane winds, calm, zero depth, TML below
    # mean, cold water expansion floor, both Coriolis signs, cooling/warming,
    # strong currents/deepening, no-shoaling, and post-deepening relaxation.
    cases = []
    for steps in (1, 8):
        for i in range(12):
            h = (50, 10, 0, 30, 50, 3, 200, 50, 15, 40, 50, 50)[i]
            t = (302, 303, 300, 290, 271, 304, 299, 301, 300, 298, 300, 300)[i]
            current = 200 if i == 8 else (10 if i == 9 else 0)
            x = [t, t+0.25, h, h, current, -current/2, t-5,
                 250, 700, 800 if i%2 else 0, 380, 65 if i%3 else 0,
                 -20 if i%3 else 0, 2.3 if i%3 else 0,
                 5.e-5 if i%2 else -5.e-5, .98, 18, .14,
                 3600 if i in (9, 10) else 0, 0, 0]
            if i == 3:
                x[6] = t+1
            if i == 11:
                x[7:11] = [-500, -200, 1000, 450]
            cases.append((x, steps))
    text = str(len(cases))+'\n'+'\n'.join(' '.join(map(str,x))+f' {steps}' for x,steps in cases)+'\n'
    def run(extra):
        subprocess.run([FC,*flags,*extra,source.name,'driver.f90','-o','oracle'],cwd=BUILD,check=True)
        return np.loadtxt(subprocess.run([str(BUILD/'oracle')],input=text,text=True,capture_output=True,check=True).stdout.splitlines(),dtype=np.float32)
    result = run([])
    sensitivity = run(['-finit-real=snan','-finit-integer=-999','-finit-logical=true'])
    fma = run(['-O2','-march=native','-ffp-contract=fast'])
    # Independent input-ULP conditioning, entirely in the pristine executable.
    # Each operand is perturbed +/- one REAL ULP, including the initial carry.
    # Sum of the per-operand envelopes bounds first-order input rounding.
    perturb=[]
    for x,steps in cases:
        for operand in range(19):
            for direction in (-np.inf,np.inf):
                y=np.asarray(x,np.float32).copy()
                y[operand]=np.nextafter(y[operand],np.float32(direction))
                # Parameters are caller controls, not uncertain operands. Avoid
                # changing the explicit tau<=0/zero-depth branch predicates.
                if operand in (2,3,18) and x[operand]==0:
                    y[operand]=0
                perturb.append((y,steps))
    text=str(len(perturb))+'\n'+'\n'.join(' '.join(map(str,x))+f' {steps}' for x,steps in perturb)+'\n'
    varied=run([]).reshape(len(cases),19,2,7)
    ulp_bound=np.max(np.abs(varied-result[:,None,None,:]),axis=2).sum(axis=1)
    out = ROOT/'proofs/s2_oml/fixtures'
    out.mkdir(exist_ok=True)
    np.savez(out/'oml.npz',inputs=np.asarray([x for x,_ in cases],np.float32),steps=np.asarray([s for _,s in cases]),
             expected=result, fma=fma, input_ulp_bound=ulp_bound)
    (out/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source':'WRF 4.7.1 phys/module_sf_oml.F (unmodified)', 'compiler':FC,'flags':flags,
        'columns':len(cases),'sensitivity_byte_equal':bool(np.array_equal(result,sensitivity)),
        'census':{'hurricane_wind':sum(x[11]==65 for x,_ in cases),'zero_wind':sum(x[11]==0 for x,_ in cases),
                  'zero_depth':sum(x[2]==0 for x,_ in cases),'relaxation':sum(x[18]>0 for x,_ in cases),
                  'strong_currents':sum(x[4]>0 for x,_ in cases),'multiple_steps':12}},indent=2)+'\n')
    print('OML fixtures', result.shape, 'finite',np.isfinite(result).all(),'init sensitivity byte equal',np.array_equal(result,sensitivity))


def generate_initialization():
    """Missing initialization regimes, without repeating the completed OML1D arm."""
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    build=BUILD.with_name('oracle_init');build.mkdir(parents=True,exist_ok=True)
    source=WRF/'phys/module_sf_oml.F';shutil.copy2(source,build/source.name)
    driver='''program init_oracle
use module_sf_oml
implicit none
real :: tsk(2,1),tml(2,1),t0(2,1),h(2,1),h0(2,1),hu(2,1),hv(2,1),tm(2,1),depth
integer :: i
read(*,*) depth
tsk(:,1)=[302.,301.]; tml(:,1)=[280.,281.]; t0(:,1)=[282.,283.]
h(:,1)=[10.,11.];h0(:,1)=[22.,23.];hu(:,1)=[3.,4.];hv(:,1)=[5.,6.];tm(:,1)=[200.5,201.1]
call omlinit(depth,tsk,tml,t0,h,h0,hu,hv,tm,.true.,.true., &
 1,3,1,2,1,2,1,2,1,1,1,1,1,2,1,1,1,1)
do i=1,2
 write(*,'(7(ES25.16E3,1x))') tml(i,1),t0(i,1),h(i,1),h0(i,1),hu(i,1),hv(i,1),tm(i,1)
enddo
end program
subroutine wrf_debug(level,msg)
integer :: level
character(*) :: msg
end subroutine
'''
    (build/'driver.f90').write_text(driver)
    subprocess.run([FC,'-O0','-ffp-contract=off','-ffree-form','-ffree-line-length-none',
                    source.name,'driver.f90','-o','oracle'],cwd=build,check=True)
    result=[]
    for hml0 in (50.,0.,-1.):
        raw=subprocess.run([str(build/'oracle')],input=f'{hml0}\n',text=True,capture_output=True,check=True).stdout
        expected=np.loadtxt(raw.splitlines(),dtype=np.float32)
        result.append({'oml_hml0':hml0,'tsk':[302.,301.],'h0ml':[22.,23.],'huml':[3.,4.],
                       'hvml':[5.,6.],'tmoml':[200.5,float(np.float32(201.1))],'expected':expected.tolist()})
    out=ROOT/'proofs/s2_oml/fixtures/init.json'
    out.write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':result},indent=2)+'\n')
    print('Pristine initialization all three depth modes x2 columns')


if __name__ == '__main__':
    import sys
    generate_initialization() if '--init-only' in sys.argv else main()
