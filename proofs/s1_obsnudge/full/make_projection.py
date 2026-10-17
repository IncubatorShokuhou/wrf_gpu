"""Original full module_llxy + unchanged in4dob rotate_vector savepoints."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'oracle'))
from build import extract,FC,PRISTINE


def main(build,out):
    build.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    llxy=PRISTINE/'share/module_llxy.F';raw=llxy.read_text()
    rot=extract((PRISTINE/'share/wrf_fddaobs_in.F').read_text(),'rotate_vector')
    (build/'llxy.F90').write_text(raw)
    (build/'error.F90').write_text('''module module_wrf_error
contains
subroutine wrf_error_fatal(message)
character(*) :: message
print *,message
error stop 1
end subroutine
end module
''')
    (build/'driver.F90').write_text('''program probe
use module_llxy
implicit none
type(proj_info) :: proj
integer :: n,k,code
real :: la1,lo1,tl1,tl2,sl,dx,dy,lat,lon,u,v,i,j
open(10,file='input.txt',status='old')
open(11,file='output.bin',access='stream',form='unformatted',status='replace')
read(10,*) n
do k=1,n
read(10,*) code,la1,lo1,tl1,tl2,sl,dx,dy,lat,lon,u,v
call map_init(proj)
call map_set(code,proj,lat1=la1,lon1=lo1,truelat1=tl1,truelat2=tl2,stdlon=sl,dx=dx,knowni=1.,knownj=1.)
call latlon_to_ij(proj,lat,lon,i,j)
call rotate_vector(lon,u,v,proj,code)
write(11) i+.5,j+.5,u,v
enddo
close(11)
end program
'''+rot)
    configs=[(1,45.2,7.7,30.,60.,8.,3000.,3000.),
             (1,-45.,170.,-30.,-60.,180.,9000.,9000.),
             (1,40.,-104.,40.,40.,-105.,12000.,12000.),
             (2,65.,-135.,60.,60.,-105.,9000.,9000.),
             (2,-65.,100.,-60.,-60.,105.,9000.,9000.),
             (3,10.,175.,20.,20.,180.,3000.,3000.)]
    cases=[]
    for config in configs:
        for dlat,dlon in ((0.,0.),(.3,.4),(-.2,-.3)):
            cases.append((*config,config[1]+dlat,config[2]+dlon,7.5,-2.25))
    (build/'input.txt').write_text(str(len(cases))+'\n'+'\n'.join(' '.join(map(str,c)) for c in cases)+'\n')
    data=dict(inputs=np.asarray(cases,np.float64))
    flags=['-O2','-ftree-vectorize','-funroll-loops','-ffree-form','-ffree-line-length-none']
    for precision in (4,8):
        command=[str(FC),*flags,*(['-fdefault-real-8'] if precision==8 else []),
                 'error.F90','llxy.F90','driver.F90','-o','probe.exe']
        subprocess.run(command,cwd=build,check=True,timeout=90)
        subprocess.run(['./probe.exe'],cwd=build,check=True,timeout=20)
        data[f'output_r{precision}']=np.fromfile(build/'output.bin',dtype=f'f{precision}').reshape(-1,4)
    dest=out/'projection.npz';np.savez_compressed(dest,**data)
    (out/'PROJECTION_MANIFEST.json').write_text(json.dumps(dict(
        llxy_sha256=hashlib.sha256(raw.encode()).hexdigest(),rotate_sha256=hashlib.sha256(rot.encode()).hexdigest(),
        fixture_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),compiler=str(FC),flags=flags,
        cases=len(cases),precisions=[4,8],bounds='4eps * max(abs(reference),1), finite required'),indent=2)+'\n')


if __name__=='__main__':main(*map(lambda p:Path(p).resolve(),sys.argv[1:]))
