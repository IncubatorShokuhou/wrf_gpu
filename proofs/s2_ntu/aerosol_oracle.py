"""Pristine NTU aerosol inversion/activation corpus, including held-cutoff calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from build_oracle import FC
from port_scalar import SOURCE

N=16
FIELDS='tk1d w1d nc1d nr1d qacac qacar rx0 ssw'.split()
WIDTH=len(FIELDS)+4+3+1


def observer():
    writes='write(u) '+','.join(FIELDS)+',abcd,zccns,real(merge(1,0,do_fr))'
    return f"""
subroutine observe_activation()
integer::u,ccnty,i,it
integer,parameter::n={N}
real::{','.join(FIELDS)},rc,abcdn(4),frac(n)
real::abcd(4),zccns(3)
logical::do_fr
frac=(/0.,1.e-12,1.e-9,1.e-6,1.e-4,0.01,0.05,0.2,0.5,0.8,0.9,0.99,0.999,0.99999,0.999999,1./)
open(newunit=u,file='activation.bin',access='stream',form='unformatted',status='replace')
do ccnty=1,4
call aero_const(ccnty)
do i=1,n
tk1d=250.+real(i-1)*2.5
w1d=0.1+real(mod(i,5))*0.7
nc1d=real(mod(i,3))*2.e7;nr1d=real(mod(i,4))*10.
abcd(2)=1.e-8*real(ccnty);abcd(1)=abcd(2)*frac(i)
abcd(3)=abcd(2)-abcd(1);abcd(4)=0.
zccns=abcd(2)*wmas(:,1)*rfact(:,1)
rx0=99.;do_fr=.true.;ssw=0.01
qacac=0.;qacar=0.
do it=1,2
{writes}
call find_rc0(dble(abcd(1)/abcd(2)),cnmod(:,1),cnstd(:,1),wmas(:,1),rc,tblxf(:,1),tblrc)
write(u) rc
call activa(tk1d,w1d,nc1d,nr1d,abcd,qacac,qacar,rx0,zccns,ssw,1,do_fr)
{writes}
ssw=ssw*1.2
enddo
enddo
enddo
close(u)
end subroutine observe_activation
"""


def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--kind',choices=('r4','r8'),default='r4');a=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu' and not Path('/tmp/wrf_gpu2_quiet').exists()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=SOURCE.read_text();needle='      END MODULE MODULE_MP_NTU';assert source.count(needle)==1
    (out/'original.F').write_text(source.replace(needle,observer()+needle))
    (out/'error.F90').write_text('module module_wrf_error\ncontains\nsubroutine wrf_error_fatal(s)\ncharacter(*)::s\nprint *,s\nerror stop 1\nend subroutine\nend module\n')
    (out/'driver.F90').write_text('program oracle\nuse module_mp_ntu\ncall observe_activation()\nend program\n')
    cmd=[FC,'-cpp','-ffree-form','-ffree-line-length-none','-O0','-ffp-contract=off','-fno-range-check','-finit-real=zero','-finit-integer=0','-finit-logical=false','error.F90','original.F','driver.F90','-o','oracle']
    if a.kind=='r8':cmd[1:1]=['-fdefault-real-8','-fdefault-double-8']
    subprocess.run(cmd,cwd=out,check=True)
    with (out/'stdout.log').open('w') as log:subprocess.run([str(out/'oracle')],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    dtype=np.dtype('float32' if a.kind=='r4' else 'float64')
    values=np.fromfile(out/'activation.bin',dtype=dtype).reshape(4,N,2,2*WIDTH+1)
    before=values[...,:WIDTH];rc=values[...,WIDTH];after=values[...,WIDTH+1:]
    assert np.isfinite(values).all(),'original activation corpus nonfinite'
    np.savez_compressed(out/'activation.npz',before=before,after=after,rc=rc)
    result={'kind':a.kind,'calls':4*N*2,'compile':cmd,'fields':FIELDS,'width':WIDTH,
            'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'npz_sha256':hashlib.sha256((out/'activation.npz').read_bytes()).hexdigest(),
            'active':{'cloud':int(np.count_nonzero(after[...,4])),'rain':int(np.count_nonzero(after[...,5])),
                      'held_cutoff_calls':4*N},'nonfinite':0}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    for name in ('original.F','error.F90','driver.F90','oracle','activation.bin','module_mp_ntu.mod','module_wrf_error.mod'):(out/name).unlink(missing_ok=True)


if __name__=='__main__':main()
