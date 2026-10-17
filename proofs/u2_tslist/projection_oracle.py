"""Build pristine module_llxy privately; save WRF projected REAL coordinates."""
from pathlib import Path
import hashlib,json,os,subprocess,sys

repo=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=True)
source=repo/'data/wrf_pristine/WRF/share/module_llxy.F'
(out/'module_llxy.f90').write_bytes(source.read_bytes())
(out/'error.f90').write_text('''module module_wrf_error
contains
subroutine wrf_error_fatal(message)
character(len=*),intent(in)::message
print *,message
stop 1
end subroutine
end module
''')
cases=[
    [1,30.,60.,10.,3000.,45.,5.,46.608,8.489],
    [1,-30.,-60.,150.,9000.,-40.,145.,-39.5,146.],
    [1,30.,30.,179.,9000.,28.,178.,29.,-179.],
    [2,60.,60.,-105.,9000.,65.,-110.,66.,-108.],
    [2,-60.,-60.,100.,9000.,-65.,90.,-64.,92.],
    [3,20.,20.,0.,3000.,25.,-18.,25.4,-17.5],
    [3,0.,0.,0.,9000.,-5.,179.,-4.,-179.],
]
program='''program probe
use module_llxy
type(proj_info)::p
integer::code,n
real::tl1,tl2,stdlon,dx,lat1,lon1,lat,lon,i,j
open(41,file='cases.txt')
open(42,file='result.txt',status='replace')
do n=1,7
read(41,*)code,tl1,tl2,stdlon,dx,lat1,lon1,lat,lon
call map_init(p)
call map_set(code,p,truelat1=tl1,truelat2=tl2,stdlon=stdlon,dx=dx,lat1=lat1,lon1=lon1,knowni=1.,knownj=1.)
call latlon_to_ij(p,lat,lon,i,j)
write(42,'(2ES24.15)')i,j
enddo
close(41)
close(42)
end program
'''
(out/'probe.f90').write_text(program)
(out/'cases.txt').write_text('\n'.join(' '.join(map(str,c)) for c in cases)+'\n')
fc=os.environ.get('FC','<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran')
subprocess.run([fc,'-O2','-ffp-contract=off','-ffree-line-length-none','error.f90','module_llxy.f90','probe.f90','-o','probe'],cwd=out,check=True,timeout=60)
subprocess.run([str(out/'probe')],cwd=out,check=True,timeout=30)
result=[list(map(float,line.split())) for line in (out/'result.txt').read_text().splitlines()]
receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'compiler':fc,
         'flags':['-O2','-ffp-contract=off'],'cases':cases,'expected_ij':result,
         'scope':'pristine WRF spherical projections, both hemispheres and dateline wrapping'}
(out/'projection.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
