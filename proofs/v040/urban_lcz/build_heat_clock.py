"""Small pristine calc_coszen + distributed AHE caller oracle, REAL4/REAL8."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
WRF=Path('<USER_HOME>/src/wrf_pristine/WRF')
OUT=Path('<USER_HOME>/wrf_gpu2_lanes/u2-lcz/heat_clock')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
OUT.mkdir(exist_ok=True)
src=(WRF/'phys/module_radiation_driver.F').read_text()
pattern=r'(?im)^\s*subroutine calc_coszen\b[\s\S]*?^\s*end subroutine calc_coszen[^\n]*'
body=re.search(pattern,src).group(0)
sf=(WRF/'phys/module_surface_driver.F').read_text().splitlines()
pbl=(WRF/'phys/module_pbl_driver.F').read_text().splitlines()
# Exact source statements; outer tile loops reduce to the one-cell wrapper.
hfx=next(s.strip() for s in sf if 'HFX(i, j) = HFX(i, j) + ahe' in s)
rth=next(s.strip() for s in pbl if 'RTHBLTEN(i, 1, j) = RTHBLTEN(i, 1, j) + ahe' in s)
ihour=next(s.strip() for s in sf if 'ihour = (jmonth - 1) * 24' in s)
driver=f'''program heat_clock
implicit none
real :: coszen(1,1),hrang(1,1),xlon(1,1),xlat(1,1),julian,xtime,gmt,degrad
real :: hfx(1,1),ahe(1,0:287,1),rthblten(1,1,1),dz8w(1,1,1),omg,pi
integer :: k,n,i,j,jmonth,ihour,tloc,tloc2
pi=3.14159;degrad=acos(-1.)/180.;gmt=18.;xlat=28.;xlon=-16.5
read(*,*) n
do k=1,n
read(*,*) julian,xtime
call calc_coszen(1,1,1,1,1,1,1,1,julian,xtime,gmt,0.,degrad,xlon,xlat,coszen,hrang)
omg=hrang(1,1)
tloc=mod(int(omg/pi*180./15.+12.+0.5),24)
if(tloc.lt.0) tloc=tloc+24
if(tloc.eq.0) tloc=24
tloc2=mod(int((omg/pi*180./15.+12.)*2.+0.5),48)
if(tloc2.lt.0) tloc2=tloc2+48
if(tloc2.eq.0) tloc2=48
write(*,'(ES24.16E3,2I6)') omg,tloc,tloc2
end do
do k=0,287
ahe(1,k,1)=real(k)
end do
i=1;j=1;jmonth=12;xtime=60.;hfx=10.;rthblten=.001;dz8w=20.
{ihour}
{hfx}
{rth}
write(*,'(I6,2ES24.16E3)') ihour,hfx(1,1),rthblten(1,1,1)
end program
'''
leads=np.array([0,18,1764,1782,1800,3600,86400,88200],np.float64)
rad=np.floor(leads/1782)*1782
# Jan13,18UTC; WRF passes current fractional zero-based day to calc_coszen.
julian=np.float64(12.75)+rad/86400
solar=(rad+900)/60
text=str(len(leads))+'\n'+'\n'.join(f'{d:.17g} {t:.17g}' for d,t in zip(julian,solar))+'\n'
result={'lead_seconds':leads,'rad_seconds':rad,'julian':julian,'solar_minutes':solar}
flags=['-O2','-ffree-form','-ffree-line-length-none','-finit-real=zero','-finit-integer=0','-finit-logical=false']
for precision in ('fp32','fp64'):
    work=OUT/precision;work.mkdir(exist_ok=True)
    (work/'reference.f90').write_text(body+'\n'+driver)
    extra=['-fdefault-real-8','-fdefault-double-8'] if precision=='fp64' else []
    assert not Path('/tmp/wrf_gpu2_quiet').exists(),'QUIET'
    subprocess.run([FC,*flags,*extra,'reference.f90','-o','reference'],cwd=work,check=True,timeout=30)
    got=subprocess.run([str(work/'reference')],input=text,text=True,stdout=subprocess.PIPE,check=True,timeout=5).stdout
    lines=got.splitlines()
    result['hour_angle_'+precision]=np.array([float(s.split()[0]) for s in lines[:-1]])
    result['hour_indices_'+precision]=np.array([[int(s.split()[1]),int(s.split()[2])] for s in lines[:-1]])
    result['ahe_'+precision]=np.array([float(s) for s in lines[-1].split()])
np.savez_compressed(OUT/'clock_reference.npz',**result)
manifest={'source':'WRF4.7.1 f52c197ed39','calc_coszen_sha256':hashlib.sha256(body.encode()).hexdigest(),
    'surface_ahe_statement':hfx,'pbl_ahe_statement':rth,'ihour_statement':ihour,'flags':flags,
    'fixture_sha256':hashlib.sha256((OUT/'clock_reference.npz').read_bytes()).hexdigest()}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT)
