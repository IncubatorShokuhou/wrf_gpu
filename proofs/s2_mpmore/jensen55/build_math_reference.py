"""Unchanged original Jensen scalar functions on frozen raw inputs."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
import numpy as np

SOURCE=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_jensen_ishmael.F')
FC='<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'
FIELDS=('temperature','gamma_argument','delta_shape','axis','rime_rate')
OUTPUT=('liquid_svp','ice_svp','gamma','gammln','growth_ratio','alphstr','capacitance','dry_growth')


def caller():
    source=SOURCE.read_text()
    igr=re.search(r'igrdata\s*=\s*&\s*(\(/.*?/\))',source,re.S|re.I)[1]
    return f'''program math_caller
use module_mp_jensen_ishmael, only: polysvp,gamma,gammln,get_igr,capacitance_gamma,wet_growth_check
implicit none
integer :: n,i,unit
real :: temperature,x,ds,a,rime,alph,cap,igr(60),v(8),xxlv,xxlf,rho,qv,dv,kt,qs0,fv,fh,rni,ni,nu
logical :: dry
igr={igr}
open(newunit=unit,file='input.bin',access='stream',form='unformatted',status='old')
read(unit) n
open(99,file='output.bin',access='stream',form='unformatted',status='replace')
do i=1,n
read(unit) temperature,x,ds,a,rime
alph=(0.1e-6)**(1.-ds)
v(1)=polysvp(temperature,0)
v(2)=polysvp(temperature,1)
v(3)=gamma(x)
v(4)=gammln(abs(x)+.5)
v(5)=get_igr(igr,temperature)
v(6)=alph
v(7)=capacitance_gamma(a,ds,4.,alph,1./gamma(4.))
xxlv=2.5e6;xxlf=3.34e5;rho=1.;qv=.005;dv=2.e-5;kt=.025;qs0=.0038
fv=1.4;fh=1.2;rni=3.e-4;ni=1.e4;nu=4.
dry=.true.
call wet_growth_check(nu,temperature,rho,xxlv,xxlf,qv,dv,kt,qs0,fv,fh,rime,rni,ni,dry)
v(8)=0.
if(dry) v(8)=1.
write(99) v
enddo
close(unit);close(99)
end program
'''


def inputs(dtype):
    # Original function branch boundaries and active ISHMAEL habit interval.
    t=np.asarray([193.16,213.15,214.15,233.15,253.15,268.15,272.15,273.15,293.15],dtype)
    t=np.concatenate([np.nextafter(t,dtype(-np.inf)),t,np.nextafter(t,dtype(np.inf)),
                      np.linspace(220.,290.,31,dtype=dtype)])
    x=np.asarray([-.5,-1.5,-3.5,1.e-8,.5,1.,1.5,2.,3.55,4.,4.55,5.,5.3,6.,8.5,11.,12.,20.,34.],dtype)
    ds=np.asarray([.55,.6,.8,1.,1.1,1.2,1.3],dtype)
    rows=[]
    for index,temp in enumerate(t):
        rows.append([temp,x[index%len(x)],ds[index%len(ds)],dtype(2.e-6*(index+1)),
                     dtype((1.e-9,1.e-5,1.e-3)[index%3])])
    return np.asarray(rows,dtype=dtype)


def main(tree,artifact):
    assert (tree/'.git').is_file()
    if artifact.exists():raise SystemExit('Use a fresh artifact directory')
    build=artifact/'build';out=artifact/'fixtures';build.mkdir(parents=True);out.mkdir()
    original=SOURCE.read_bytes();driver=caller();rows=[]
    for mode,dtype in [('fp32',np.float32),('fp64',np.float64)]:
        b=build/mode;b.mkdir();(b/'module.F').write_bytes(original)
        (b/'stubs.f90').write_text('module module_wrf_error\nend module\n');(b/'driver.F90').write_text(driver)
        flags=['-O2','-ffree-form','-ffree-line-length-none','-cpp','-finit-real=zero',
               '-finit-integer=0','-finit-logical=false','-ffp-contract=off','-fconvert=big-endian','-frecord-marker=4']
        if mode=='fp64':flags+=['-fdefault-real-8','-fdefault-double-8']
        for name in ('stubs.f90','module.F','driver.F90'):
            subprocess.run([FC,*flags,'-c',name],cwd=b,check=True,timeout=180)
        subprocess.run([FC,*flags,'stubs.o','module.o','driver.o','-o','oracle'],cwd=b,check=True,timeout=180)
        x=inputs(dtype);io_dtype=np.dtype(dtype).newbyteorder('>')
        raw=np.asarray([len(x)],dtype='>i4').tobytes()+x.astype(io_dtype).tobytes()
        (b/'input.bin').write_bytes(raw)
        subprocess.run([str(b/'oracle')],cwd=b,check=True,timeout=30)
        data=np.frombuffer((b/'output.bin').read_bytes(),io_dtype).reshape(len(x),len(OUTPUT))
        assert np.isfinite(data).all()
        value=dict(mode=mode,input_names=FIELDS,output_names=OUTPUT,inputs=x.tolist(),outputs=data.tolist())
        path=out/f'{mode}.json';path.write_text(json.dumps(value,indent=2)+'\n')
        rows.append(dict(mode=mode,observations=data.size,input_sha256=hashlib.sha256(raw).hexdigest(),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest(),flags=flags,
                         module_object_sha256=hashlib.sha256((b/'module.o').read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(original).hexdigest(),
        caller_sha256=hashlib.sha256(driver.encode()).hexdigest(),rows=rows),indent=2)+'\n')
    print(rows,flush=True)


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
