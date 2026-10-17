"""Original MSKF timescale block and scalar gfortran NINT compatibility.

This is a component fixture for the documented upstream calm singularity,
not a replacement for the complete MSKF_CPS primary oracle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    import numpy as np
    build=args.build_dir.resolve();build.mkdir(parents=True,exist_ok=True)
    (build/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(build/'tmp')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    original=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_cu_mskf.F')
    text=original.read_text()
    first=text.index('        WSPD(KLCL)=SQRT(U0(KLCL)*U0(KLCL)+V0(KLCL)*V0(KLCL))')
    last=text.index('         TIMEC=MIN(TIMEC,86400.)',first)
    last=text.index('\n',last)+1
    block=text[first:last]
    header='''subroutine original_timescale(wind,vconv,tadvec,timec,nic,abe,sclvel)
implicit none
real,intent(in)::wind
real,intent(out)::vconv,tadvec,timec,abe,sclvel
integer,intent(out)::nic
integer,parameter::klcl=1,l5=2,ltop=3,lc=1,kpbl=2,i=1,j=1
real::wspd(3),u0(3),v0(3),p0(3),t0(3),q0(3),z0(3),cldhgt(3)
real::dx,dt,chmin,scale_fac,densplume,wst,hfx(1,1),cp,g,r,pblh(1,1)
real::thetav,eps1u,zlcl,zlcl_kay,ust(1,1),zol(1,1),frc2,zz_kay
u0=wind;v0=0.;p0=100000.;t0=300.;q0=.015;z0=800.;cldhgt=6000.
dx=9000.;dt=54.;chmin=4000.;scale_fac=2.;hfx=150.;cp=1004.5;g=9.81;r=287.
pblh=1000.;zlcl=500.;ust=.3;zol=-.1;abe=1000.
'''
    (build/'timescale.f90').write_text(header+block+'end subroutine\n')
    # A separate compilation unit preserves the scalar NINT ABI used in the
    # original conditional MSKF code; the input builder is never native-FMA.
    (build/'nint_scalar.f90').write_text('''subroutine original_nint(x,y)
implicit none
real,intent(in)::x
integer,intent(out)::y
y=nint(x)
end subroutine
''')
    rng=np.random.default_rng(104711)
    bits=np.concatenate((rng.integers(0,2**32,4096,dtype=np.uint32),np.asarray([
        0,0x80000000,0x3effffff,0x3f000000,0xbf000000,0x3fc00000,0x40200000,
        0x4effffff,0x4f000000,0xcf000000,0x5effffff,0x5f000000,0x7f800000,0xff800000,
        0x7fc00000,0x7fa00000,0x00000001,0x007fffff],np.uint32)))
    bits.tofile(build/'nint_inputs.bin')
    (build/'driver.f90').write_text(f'''program original_probe
implicit none
integer,parameter::n={len(bits)}
real::x(n),vconv,tadvec,timec,abe,sclvel,wind
integer::y(n),k,nic
real,parameter::winds(5)=[0.,.02,.1,2.,15.]
open(70,file='nint_inputs.bin',access='stream',form='unformatted',status='old')
read(70)x
close(70)
do k=1,n
 call original_nint(x(k),y(k))
enddo
open(71,file='nint_expected.bin',access='stream',form='unformatted',status='replace')
write(71)y
close(71)
do k=1,5
 wind=winds(k)
 call original_timescale(wind,vconv,tadvec,timec,nic,abe,sclvel)
 write(*,'(A,I0,1X,5(ES23.15,1X),I0)') 'CASE ',k,wind,vconv,tadvec,timec,sclvel,nic
enddo
end program
''')
    compiler=shutil.which('gfortran') or '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
    flags=['-O2','-ffree-form','-ffree-line-length-none']
    for name in ('timescale','nint_scalar','driver'):
        subprocess.run([compiler,*flags,'-c',name+'.f90'],cwd=build,check=True,timeout=60)
    subprocess.run([compiler,'timescale.o','nint_scalar.o','driver.o','-o','probe'],cwd=build,check=True,timeout=60)
    result=subprocess.run([str(build/'probe')],cwd=build,capture_output=True,text=True,check=True,timeout=30)
    (out/'original.stdout').write_text(result.stdout)
    (out/'original.stderr').write_text(result.stderr)
    expected=np.fromfile(build/'nint_expected.bin','<i4')
    assert len(expected)==len(bits)
    np.savez(out/'nint_fixture.npz',bits=bits,expected=expected)
    rows=[]
    for line in result.stdout.splitlines():
        words=line.split()
        if not words or words[0]!='CASE':continue
        names=('wind','vconv','tadvec','timec','sclvel')
        row={key:float(value) for key,value in zip(names,words[2:7])}
        row['nic']=int(words[7]);row['case']=int(words[1]);rows.append(row)
    assert len(rows)==5 and np.isinf(rows[0]['tadvec']) and rows[0]['nic']==0 and rows[0]['timec']==0
    assert all(np.isfinite([row[key] for key in ('vconv','tadvec','timec','sclvel')]).all() for row in rows[1:])
    for row in rows:
        if not np.isfinite(row['tadvec']):row['tadvec']='Infinity'
    record=dict(original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                block_sha256=hashlib.sha256(block.encode()).hexdigest(),
                lines=[text[:first].count('\n')+1,text[:last].count('\n')],compiler=compiler,flags=flags,
                nint_samples=len(bits),fixture_sha256=hashlib.sha256((out/'nint_fixture.npz').read_bytes()).hexdigest(),
                rows=rows,scope='component only; complete primary CPS/MSKF CMT remains required')
    (out/'manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(build/'timescale.f90',out/'original_timescale.f90')
    print(json.dumps(record,allow_nan=False))


if __name__=='__main__':
    main()
