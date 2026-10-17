! External caller for the verbatim pristine WRF groundwater module.
! Fortran binary arrays use original (i,j) / (i,soil,j) storage order.
program mmf_groundwater_driver
  use module_sf_noahmp_groundwater, only: WTABLE_mmf_noahmp
  use NOAHMP_TABLES
  implicit none
  integer :: nx,ny,ni,no,nsoil
  character(len=1024) :: input_path,output_path
  real :: wtddt,xice_threshold
  integer,allocatable :: isltyp(:,:),ivgtyp(:,:)
  real,allocatable :: xland(:,:),xice(:,:),smoiseq(:,:,:),dzs(:)
  real,allocatable :: fdepth(:,:),area(:,:),topo(:,:),rivercond(:,:),riverbed(:,:),eqwtd(:,:),pexp(:,:)
  real,allocatable :: smois(:,:,:),sh2o(:,:,:),smcwtd(:,:),wtd(:,:),deeprech(:,:)
  real,allocatable :: qlat(:,:),qrf(:,:),qslat(:,:),qrfs(:,:),qsprings(:,:),rech(:,:)
  ! E24: preserve the WRF caller's initialized non-land QSPRING memory, which
  ! WTABLE declares INTENT(OUT) but only writes on active land columns.
  real,allocatable,volatile :: qspring(:,:)
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  open(newunit=ni,file=trim(input_path),access='stream',form='unformatted',status='old')
  read(ni) nx,ny,nsoil,wtddt,xice_threshold
  allocate(isltyp(nx,ny),ivgtyp(nx,ny),xland(nx,ny),xice(nx,ny),smoiseq(nx,nsoil,ny),dzs(nsoil))
  allocate(fdepth(nx,ny),area(nx,ny),topo(nx,ny),rivercond(nx,ny),riverbed(nx,ny),eqwtd(nx,ny),pexp(nx,ny))
  allocate(smois(nx,nsoil,ny),sh2o(nx,nsoil,ny),smcwtd(nx,ny),wtd(nx,ny),deeprech(nx,ny))
  allocate(qlat(nx,ny),qrf(nx,ny),qspring(nx,ny),qslat(nx,ny),qrfs(nx,ny),qsprings(nx,ny),rech(nx,ny))
  read(ni) xland,xice,isltyp,ivgtyp,smoiseq,dzs,fdepth,area,topo,rivercond,riverbed,eqwtd,pexp
  read(ni) smois,sh2o,smcwtd,wtd,deeprech,qslat,qrfs,qsprings,rech,qspring
  close(ni)
  call read_mp_veg_parameters('MODIFIED_IGBP_MODIS_NOAH')
  call read_mp_soil_parameters()
  call WTABLE_mmf_noahmp(nsoil,xland,xice,xice_threshold,ISICE_TABLE, &
        isltyp,smoiseq,dzs,wtddt,fdepth,area,topo,ISURBAN_TABLE,ivgtyp, &
        rivercond,riverbed,eqwtd,pexp,smois,sh2o,smcwtd,wtd,qlat,qrf, &
        deeprech,qspring,qslat,qrfs,qsprings,rech, &
        1,nx+1,1,ny+1,1,1, 1,nx,1,ny,1,1, 1,nx,1,ny,1,1)
  open(newunit=no,file=trim(output_path),access='stream',form='unformatted',status='replace')
  write(no) BEXP_TABLE(1:19),DKSAT_TABLE(1:19),SMCMAX_TABLE(1:19),PSISAT_TABLE(1:19),SMCWLT_TABLE(1:19)
  write(no) smois,sh2o,smcwtd,wtd,deeprech,qlat,qrf,qspring,qslat,qrfs,qsprings,rech
  close(no)
end program

subroutine wrf_error_fatal(message)
  character(len=*),intent(in) :: message
  print *, trim(message)
  stop 1
end subroutine

subroutine wrf_message(message)
  character(len=*),intent(in) :: message
  print *, trim(message)
end subroutine
