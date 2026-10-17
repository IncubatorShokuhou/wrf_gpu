"""Compile exact pristine WRF prep/conversion statements in a private wrapper."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

WRF=Path('<USER_HOME>/src/wrf_pristine/WRF')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'


def main():
    p=argparse.ArgumentParser();p.add_argument('build',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    a.build.mkdir(parents=True,exist_ok=True)
    source=(WRF/'dyn_em/module_big_step_utilities_em.F').read_text()
    sub=re.search(r'(?im)^SUBROUTINE conv_t_tendf_to_moist\s*\(.*?^END SUBROUTINE conv_t_tendf_to_moist',source,re.S|re.M).group()
    prep=next(l.strip() for l in source.splitlines() if 'th_phy(i,k,j) = (t(i,k,j)+t0)/(1.+R_v/R_d*moist' in l)
    temp=next(l.strip() for l in source.splitlines() if 't_phy(i,k,j) = th_phy(i,k,j)*pi_phy(i,k,j)' in l)
    wrapper=f'''module operator
use module_model_constants
contains
{sub}
end module
program probe
use operator
implicit none
real :: t(3,1,1), moist(3,1,1,1), th_phy(3,1,1), t_phy(3,1,1), pi_phy(3,1,1)
real :: old(3,1,1), qv(3,1,1), rate(3,1,1), vapor(3,1,1), pressure(3,1,1)
integer :: i,j,k,P_QV
j=1;k=1;P_QV=1
old(:,1,1)=[295.,302.,310.]
qv(:,1,1)=[.004,.012,.020]
pressure(:,1,1)=[95000.,80000.,60000.]
rate(:,1,1)=[.001,-.002,.003]
vapor(:,1,1)=[-2.e-6,3.e-6,-4.e-6]
t=old-t0;moist(:,:,:,1)=qv
pi_phy=(pressure/100000.)**(R_d/cp)
do i=1,3
{prep}
{temp}
enddo
call conv_t_tendf_to_moist(t,qv,rate,vapor,1,4,1,2,1,2,1,3,1,1,1,1,1,3,1,1,1,2)
write(*,'(*(ES26.17E3,1X))') old,qv,pressure,th_phy,t_phy,rate,vapor
end program
'''
    (a.build/'operators.F90').write_text(wrapper)
    constants=WRF/'share/module_model_constants.F'
    rows={}
    for kind in (32,64):
        flags=['-O0','-cpp','-ffree-form','-ffree-line-length-none','-ffp-contract=off']
        if kind==64:flags+=['-fdefault-real-8','-fdefault-double-8']
        subprocess.run([FC,*flags,str(constants),str(a.build/'operators.F90'),'-o',str(a.build/f'oracle{kind}')],cwd=a.build,check=True)
        text=subprocess.check_output([str(a.build/f'oracle{kind}')],cwd=a.build,text=True)
        values=list(map(float,text.split()));assert len(values)==21
        names=('theta_m','qv','p','theta_dry','temperature','theta_m_rate','qv_rate')
        rows[str(kind)]={n:values[3*i:3*i+3] for i,n in enumerate(names)}
    result=dict(reference='Untouched pristine WRF statements, selected use_theta_m=1 caller',
                units=dict(temperature='K',theta_m_rate='K/s',qv_rate='kg/kg/s'),
                bounds=dict(fp32_temperature_ulps=8,fp32_rate_ulps=8,fp64_relative=2.e-14),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                constants_sha256=hashlib.sha256(constants.read_bytes()).hexdigest(),
                wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
                prep_statement=prep,temperature_statement=temp,conversion_subroutine=sub,rows=rows)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(rows=rows,source_sha256=result['source_sha256'])))


if __name__=='__main__':main()
