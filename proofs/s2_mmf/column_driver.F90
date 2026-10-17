program mmf_column_driver
use mmf_column_original
use NOAHMP_TABLES,only:read_mp_veg_parameters,read_mp_soil_parameters,read_mp_global_parameters, &
 BEXP_TABLE,DKSAT_TABLE,DWSAT_TABLE,SMCMAX_TABLE,PSISAT_TABLE,SMCWLT_TABLE,FSATMX=>FSATMX_TABLE,ISURBAN_TABLE
implicit none
integer::ncol,ic,soil,veg,ni,no,option_inf,k
character(len=1024)::input_path,output_path
type(noahmp_parameters)::parameters
real::dt,zsoil(4),dzs(4),dzfull(-2:4),qinsur,qseva,etrani(4),smc(4),sh2o(4),sice(4)
real::smcwtd,wtd,smceq(4),runsrf,qdrain,wcnd(4),fcrmax,qtldrn,griddeep,gridrech
real,volatile::runsub,rech,deeprech
call get_command_argument(1,input_path)
call get_command_argument(2,output_path)
open(newunit=ni,file=trim(input_path),access='stream',form='unformatted',status='old')
read(ni)ncol,option_inf,dt
OPT_INF=option_inf
call read_mp_veg_parameters('MODIFIED_IGBP_MODIS_NOAH')
call read_mp_soil_parameters()
call read_mp_global_parameters()
open(newunit=no,file=trim(output_path),access='stream',form='unformatted',status='replace')
write(no)FSATMX
do ic=1,ncol
 read(ni)soil,veg,zsoil,dzs,qinsur,qseva,etrani,smc,sh2o,smcwtd,wtd,smceq,griddeep,gridrech
 parameters%BEXP=BEXP_TABLE(soil);parameters%DKSAT=DKSAT_TABLE(soil)
 parameters%DWSAT=DWSAT_TABLE(soil);parameters%SMCMAX=SMCMAX_TABLE(soil)
 parameters%PSISAT=PSISAT_TABLE(soil);parameters%SMCWLT=SMCWLT_TABLE(soil)
 parameters%FSATMX=FSATMX
 parameters%URBAN_FLAG=veg==ISURBAN_TABLE.or.(veg>=51.and.veg<=61)
 if(parameters%URBAN_FLAG)then
  parameters%SMCMAX=.45;parameters%SMCWLT=.40
 endif
 sice=max(0.,smc-sh2o)
 dzfull=0.;dzfull(1:4)=dzs
 runsub=0.;rech=0.;deeprech=0.
 call SOILWATER(parameters,4,3,dt,zsoil,dzfull,qinsur,qseva,etrani,sice,1,1,0.,9000., &
  sh2o,smc,wtd,veg,smcwtd,deeprech,runsrf,qdrain,runsub,wcnd,fcrmax,qtldrn)
 smc=sh2o+sice
 call SHALLOWWATERTABLE(parameters,3,4,zsoil,dt,dzfull,smceq,1,1,smc,wtd,smcwtd,rech,qdrain)
 sh2o(4)=smc(4)-sice(4)
 griddeep=griddeep+deeprech;gridrech=gridrech+rech*1.e3
 write(no)sh2o,smc,smcwtd,wtd,griddeep,gridrech,runsrf,qdrain,runsub
enddo
close(ni);close(no)
end program
