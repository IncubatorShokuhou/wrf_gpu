"""Original unmodified WRF LANDUSE_INIT, REAL4 startup/table qualification."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('tree',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args()
assert os.environ['JAX_PLATFORMS']=='cpu'
assert args.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/s2-ssib'))
assert not Path('/tmp/wrf_gpu2_quiet').exists()
args.out.mkdir();build=args.out/'build';build.mkdir()
original=Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_physics_init.F').read_bytes()
text=original.decode();a=text.index('   SUBROUTINE landuse_init(');b=text.index('   END SUBROUTINE landuse_init',a)+len('   END SUBROUTINE landuse_init')
body=text[a:b];(build/'original.f90').write_text('module native_landuse\ncontains\n'+body+'\nend module\n')
(build/'LANDUSE.TBL').symlink_to('<USER_HOME>/src/wrf_pristine/WRF/run/LANDUSE.TBL')
(build/'stubs.f90').write_text("""module module_wrf_error
integer,parameter :: IWORDSIZE=4,RWORDSIZE=4,LWORDSIZE=4
character*512 :: wrf_err_message
contains
subroutine wrf_message(msg)
character*(*) :: msg
end subroutine
subroutine wrf_debug(level,msg)
integer :: level
character*(*) :: msg
end subroutine
subroutine wrf_error_fatal(msg)
character*(*) :: msg
write(*,*) msg
stop 1
end subroutine
end module
logical function wrf_dm_on_monitor()
wrf_dm_on_monitor=.true.
end function
subroutine wrf_dm_bcast_bytes(x,n)
real :: x(*)
integer :: n
end subroutine
subroutine wrf_dm_bcast_string(x,n)
character*(*) :: x
integer :: n
end subroutine
""")
(build/'driver.f90').write_text("""program caller
use native_landuse
implicit none
integer,parameter :: N=24
real,dimension(N,1) :: lu,snowc,alb,bck,snoalb,mav,em,emb,znt,z0,thc,xland,xice,xicem
real :: state(7501),lat
integer :: map,scenario,i,day,nc,water,isice,lucats,luseas,isn,frac
character*32 :: name
 do map=1,4
 select case(map)
 case(1)
 name='USGS';nc=24;water=16
 case(2)
 name='SSIB';nc=24;water=16
 case(3)
 name='MODIS';nc=20;water=17
 case(4)
 name='MODIFIED_IGBP_MODIS_NOAH';nc=21;water=17
 end select
 do scenario=1,4
 day=15;if(mod(scenario,2)==0)day=172
 lat=45.;if(scenario>2)lat=-45.
 state=0.;xice=0.;snoalb=.8;frac=1;isice=0;lucats=0;luseas=0;isn=0
 do i=1,N
 lu(i,1)=real(min(i,nc));snowc(i,1)=real(mod(i,2))
 enddo
 call landuse_init(lu,snowc,alb,bck,snoalb,mav,em,emb,znt,z0,thc,xland,xice,xicem,day,lat,water,trim(name), &
 isice,lucats,luseas,isn,frac,state,.true.,.false., &
 1,nc+1,1,2,1,2,1,N,1,1,1,1,1,nc,1,1,1,1)
 do i=1,nc
 write(*,'(3I5,7ES25.16)')map,scenario,i,alb(i,1),bck(i,1),em(i,1),mav(i,1),znt(i,1),xland(i,1),snowc(i,1)
 enddo
 enddo
 enddo
end program
""")
try:
 result=subprocess.run(['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-finit-real=zero','-finit-integer=0','-finit-logical=false','-ffp-contract=off','-fallow-argument-mismatch','stubs.f90','original.f90','driver.f90','-o','oracle'],cwd=build,capture_output=True,text=True,timeout=60)
 (args.out/'compile.log').write_text(result.stdout+result.stderr);result.check_returncode()
 result=subprocess.run([str(build/'oracle')],cwd=build,capture_output=True,text=True,timeout=30)
 (args.out/'native.stdout').write_text(result.stdout);(args.out/'native.stderr').write_text(result.stderr);result.check_returncode()
 rows=np.asarray([[float(x) for x in line.split()] for line in result.stdout.splitlines()],np.float32)
 sys.path.insert(0,str(args.tree/'src'))
 from gpuwrf.io.ssib_landuse import ssib_landuse_initial,LANDUSE_SHA256
 fields=('albedo','albbck','emiss','mavail','znt','xland');different=[]
 for row in rows:
  map,scenario,cat=map_int=(int(x) for x in row[:3]);name=('USGS','SSIB','MODIS','MODIFIED_IGBP_MODIS_NOAH')[map-1]
  actual=ssib_landuse_initial(dataset=name,category=np.int32(cat),snowc=row[-1],day=15 if scenario%2 else 172,cen_lat=45 if scenario<3 else -45)
  for index,field in enumerate(fields):
   if np.asarray(actual[field],np.float32).tobytes()!=row[3+index].tobytes():different.append((map,scenario,cat,field,float(actual[field]),float(row[3+index])))
 np.savez_compressed(args.out/'landuse_reference.npz',rows=rows)
 receipt={'kind':'BYTE-vs-unmodified-original-WRF-LANDUSE_INIT','source_sha256':hashlib.sha256(original).hexdigest(),
  'routine_sha256':hashlib.sha256(body.encode()).hexdigest(),'table_sha256':LANDUSE_SHA256,
  'fixture_sha256':hashlib.sha256((args.out/'landuse_reference.npz').read_bytes()).hexdigest(),
  'rows':len(rows),'checks':len(rows)*len(fields),'different':different,'passed':not different}
 (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));assert receipt['passed']
finally:
 shutil.rmtree(build)
