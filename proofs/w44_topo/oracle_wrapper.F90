module module_model_constants
  real, parameter :: degrad = 3.14159265358979323846/180.
end module
module module_wrf_error
end module

module pristine_toposhad
contains
! PRISTINE_SHADING
! PRISTINE_FLUX
! PRISTINE_CLOCK
end module

subroutine shadow_probe(nx,ny,nested,terrain,lat,lon,sina,cosa,parent,xtime,gmt,radt,declin,dx,dy,shadlen,mask,height) bind(C)
  use iso_c_binding
  use pristine_toposhad
  implicit none
  integer(c_int), value :: nx,ny,nested
  real(c_float), intent(in) :: terrain(-1:nx+2,-1:ny+2),lat(-1:nx+2,-1:ny+2),lon(-1:nx+2,-1:ny+2)
  real(c_float), intent(in) :: sina(-1:nx+2,-1:ny+2),cosa(-1:nx+2,-1:ny+2),parent(-1:nx+2,-1:ny+2)
  real(c_float), value :: xtime,gmt,radt,declin,dx,dy,shadlen
  integer(c_int), intent(out) :: mask(-1:nx+2,-1:ny+2)
  real(c_float), intent(out) :: height(-1:nx+2,-1:ny+2)
  real :: local(-1:nx+2,-1:ny+2)
  local=terrain
  height=parent
  mask=0
  call toposhad_init(height,local,mask,nested/=0,1, &
      1,nx+1,1,ny+1,1,1,-1,nx+2,-1,ny+2,1,1,1,nx,1,ny,1,1,1,nx,1,ny,1,1)
  call toposhad(lat,lon,sina,cosa,xtime,gmt,radt,declin,dx,dy,height,local,1,mask,shadlen, &
      1,nx+1,1,ny+1,1,1,-1,nx+2,-1,ny+2,1,1,1,nx,1,ny,1,1,1,nx,1,ny,1,1)
end subroutine

subroutine solar_probe(nx,ny,lat,lon,julian,xtime,gmt,radfrq,declin,coszen,angles,shade_coszen,shade_angles) bind(C)
  use iso_c_binding
  use pristine_toposhad
  use module_model_constants
  implicit none
  integer(c_int), value :: nx,ny
  real(c_float), intent(in) :: lat(nx,ny),lon(nx,ny)
  real(c_float), value :: julian,xtime,gmt,radfrq
  real(c_float), intent(out) :: declin,coszen(nx,ny),angles(nx,ny),shade_coszen(nx,ny),shade_angles(nx,ny)
  real :: solcon,xt24,tloctm,hrang,xxlat,csza,xlat(nx,ny),xlong(nx,ny)
  integer :: i,j
  xlat=lat
  xlong=lon
  call radconst(xtime,declin,solcon,julian,degrad,360./365.)
  call calc_coszen(1,nx,1,ny,1,nx,1,ny,julian,xtime+radfrq*.5,gmt,declin,degrad,lon,lat,coszen,angles)
  XT24=MOD(XTIME+RADFRQ*0.5,1440.)
  do j=1,ny
   do i=1,nx
! PRISTINE_TOPOSOLAR
    shade_coszen(i,j)=csza
    shade_angles(i,j)=hrang
   enddo
  enddo
end subroutine

subroutine flux_probe(n,lat,lon,coszen,diffuse,declin,hrang,slope,aspect,mask,flux) bind(C)
  use iso_c_binding
  use pristine_toposhad
  implicit none
  integer(c_int), value :: n
  real(c_float), intent(in) :: lat(n),lon(n),coszen(n),diffuse(n),declin(n),hrang(n),slope(n),aspect(n)
  integer(c_int), intent(in) :: mask(n)
  real(c_float), intent(out) :: flux(n)
  integer :: i
  do i=1,n
    call TOPO_RAD_ADJ(lat(i),lon(i),coszen(i),diffuse(i),declin(i),3.14159265358979323846/180., &
        600.,1370.,hrang(i),flux(i),1,1,slope(i),aspect(i),mask(i),i,1)
  enddo
end subroutine
