module module_state_description
end module
module module_configure
 type grid_config_rec_type
  logical :: periodic_x=.false.
 end type
end module
module pristine_boundary
 use module_configure
 contains
! PRISTINE_BDY
! PRISTINE_SPEC
end module
subroutine nl_get_spec_zone(id,value)
 integer :: id,value
 value=1
end subroutine
subroutine nl_get_relax_zone(id,value)
 integer :: id,value
 value=4
end subroutine

subroutine caller_probe(px,py,nx,ny,ipos,jpos,ratio,parent,child,after,delta) bind(C)
 use iso_c_binding
 use pristine_boundary
 implicit none
 integer(c_int),value :: px,py,nx,ny,ipos,jpos,ratio
 real(c_float),intent(in) :: parent(-2:px+3,1,-2:py+3),child(nx+1,1,ny+1)
 real(c_float),intent(out) :: after(nx+1,1,ny+1),delta(4)
 real :: p(-2:px+3,1,-2:py+3),q(nx+1,1,ny+1)
 real :: xs(ny+1,1,5),xe(ny+1,1,5),ys(nx+1,1,5),ye(nx+1,1,5)
 real :: txs(ny+1,1,5),txe(ny+1,1,5),tys(nx+1,1,5),tye(nx+1,1,5),dummy(1)
 integer :: mask(nx+1,ny+1)
 type(grid_config_rec_type) :: config
 p=parent
 q=child
 xs=0.;xe=0.;ys=0.;ye=0.;txs=0.;txe=0.;tys=0.;tye=0.;dummy=0.
 mask=1
 call bdy_interp1(p, &
  1,px+1,1,1,1,py+1,-2,px+3,1,1,-2,py+3,1,px,1,1,1,py, &
  q,1,max(nx+1,ny+1),5, &
  1,nx+1,1,1,1,ny+1,1,nx+1,1,1,1,ny+1,1,nx,1,1,1,ny, &
  3,mask,.false.,.false.,ipos,jpos,ratio,ratio, &
  dummy,xs,dummy,xe,dummy,ys,dummy,ye,dummy,txs,dummy,txe,dummy,tys,dummy,tye,54.,18.)
 after=q
 call spec_bdyfield(after,xs,xe,ys,ye,'m',config,5,2, &
  1,nx+1,1,ny+1,1,1,1,nx+1,1,ny+1,1,1,1,nx,1,ny,1,1,1,nx,1,ny,1,1)
 delta=[maxval(abs(txs)),maxval(abs(txe)),maxval(abs(tys)),maxval(abs(tye))]
end subroutine
