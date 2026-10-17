! Literal solve_em TKE_advance caller; all numerical routines are pristine libwrflib.a.
program tke_stage_driver
  use module_configure, only: grid_config_rec_type
  use module_em, only: rk_scalar_tend, rk_update_scalar, rk_update_scalar_pd, bound_tke
  use module_bc, only: set_physical_bc3d, set_physical_bc2d, flow_dep_bdy
  implicit none
  type(grid_config_rec_type) :: cf
  integer, parameter :: h=5
  integer :: nx,ny,nz,per,diff6,rk,ims,ime,jms,jme,kme,i
  real :: dx,dy,dt,dt_rk,upper,factor
  character(512) :: fin,fout
  real, allocatable :: q(:,:,:),old(:,:,:),sc(:,:,:),adv(:,:,:),ht(:,:,:),zt(:,:,:),rq(:,:,:), &
    ru(:,:,:),rv(:,:,:),ww(:,:,:),we(:,:,:),wi(:,:,:),zero(:,:,:), &
    mo(:,:),mn(:,:),mb(:,:),mx(:,:),my(:,:),ux(:,:),uy(:,:),vx(:,:),vy(:,:),vxi(:,:), &
    c1(:),c2(:),c1f(:),c2f(:),fnm(:),fnp(:),rdn(:),rdnw(:),base(:)
  call get_command_argument(1,fin)
  call get_command_argument(2,fout)
  open(10,file=trim(fin),access='stream',form='unformatted',status='old')
  read(10) nx,ny,nz,per,diff6
  read(10) dx,dy,dt,upper,factor
  ims=1-h; ime=nx+1+h; jms=1-h; jme=ny+1+h; kme=nz+1
  cf%specified=per==0; cf%nested=.false.; cf%periodic_x=per==1; cf%periodic_y=per==1
  cf%open_xs=.false.; cf%open_xe=.false.; cf%open_ys=.false.; cf%open_ye=.false.
  cf%symmetric_xs=.false.; cf%symmetric_xe=.false.; cf%symmetric_ys=.false.; cf%symmetric_ye=.false.
  cf%polar=.false.; cf%rk_ord=3; cf%zadvect_implicit=0; cf%diff_opt=2; cf%km_opt=5
  cf%h_sca_adv_order=5; cf%v_sca_adv_order=3; cf%cu_physics=0; cf%bl_pbl_physics=0
  cf%spec_bdy_width=5; cf%diff_6th_slopeopt=0
  call nl_set_time_step(1,nint(dt))
  call a3(q); call a3(old); call a3(sc); call a3(adv); call a3(ht); call a3(zt); call a3(rq)
  call a3(ru); call a3(rv); call a3(ww); call a3(we); call a3(wi); call a3(zero)
  call a2(mo); call a2(mn); call a2(mb); call a2(mx); call a2(my)
  call a2(ux); call a2(uy); call a2(vx); call a2(vy); call a2(vxi)
  allocate(c1(kme),c2(kme),c1f(kme),c2f(kme),fnm(kme),fnp(kme),rdn(kme),rdnw(kme),base(kme))
  c1f=1.; c2f=0.; base=0.
  read(10) q(1:nx,1:nz,1:ny)
  read(10) sc(1:nx,1:nz,1:ny)
  read(10) mo(1:nx,1:ny); read(10) mn(1:nx,1:ny)
  read(10) ru(1:nx+1,1:nz,1:ny); read(10) rv(1:nx,1:nz,1:ny+1)
  read(10) ww(1:nx,1:nz+1,1:ny)
  read(10) mx(1:nx,1:ny); read(10) my(1:nx,1:ny)
  read(10) ux(1:nx+1,1:ny); read(10) uy(1:nx+1,1:ny)
  read(10) vx(1:nx,1:ny+1); read(10) vy(1:nx,1:ny+1)
  read(10) c1,c2,fnm,fnp,rdn,rdnw
  close(10)
  call bc2(mo,'t'); call bc2(mn,'t'); call bc2(mx,'t'); call bc2(my,'t')
  call bc2(ux,'u'); call bc2(uy,'u'); call bc2(vx,'v'); call bc2(vy,'v')
  vxi=1./vx
  call bc3(ru,'u'); call bc3(rv,'v'); call bc3(ww,'w'); call bc3(q,'p')
  old=q
  open(20,file=trim(fout),access='stream',form='unformatted',status='replace')
  do rk=1,3
    dt_rk=dt/real(4-rk)
    if(rk==3) then
      call rk_update_scalar_pd(1,1,old,sc,c1,c2,mo,mo,mb,rk,dt_rk,1,cf, &
        1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,1,kme,1,nx,1,ny,1,nz+1)
      call bc3(old,'p')
    endif
    call rk_scalar_tend(1,1,cf,.false.,rk,dt_rk,ru,rv,ww,we,wi,zero,zero,mn,mb,mo, &
      c1,c2,c1f,c2f,zero,old,q,sc,adv,ht,zt,rq,base,.false.,fnm,fnp, &
      ux,uy,vx,vxi,vy,mx,my,1./dx,1./dy,rdn,rdnw,0.,0.,zero,diff6,factor,1, &
      zero,zero,.false.,.false., &
      1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,1,kme,1,nx,1,ny,1,nz+1)
    call rk_update_scalar(scs=1,sce=1,scalar_1=old,scalar_2=q,sc_tend=sc,advect_tend=adv, &
      h_tendency=ht,z_tendency=zt,msftx=mx,msfty=my,c1=c1,c2=c2,mu_old=mo,mu_new=mn,mu_base=mb, &
      rk_step=rk,dt=dt_rk,spec_zone=1,config_flags=cf,tenddec=.false., &
      ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1,ims=ims,ime=ime,jms=jms,jme=jme,kms=1,kme=kme, &
      its=1,ite=nx,jts=1,jte=ny,kts=1,kte=nz+1)
    call bound_tke(q,upper,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,1,kme,1,nx,1,ny,1,nz+1)
    if(per==0) call flow_dep_bdy(q,ru,rv,cf,1, &
      1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,1,kme, &
      1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
    write(20) q(1:nx,1:nz,1:ny),adv(1:nx,1:nz,1:ny),sc(1:nx,1:nz,1:ny)
    call bc3(q,'p')
  enddo
  close(20)
contains
  subroutine a3(a)
    real,allocatable,intent(out)::a(:,:,:)
    allocate(a(ims:ime,1:kme,jms:jme)); a=0.
  end subroutine
  subroutine a2(a)
    real,allocatable,intent(out)::a(:,:)
    allocate(a(ims:ime,jms:jme)); a=0.
  end subroutine
  subroutine bc3(a,stag)
    real,intent(inout)::a(ims:ime,1:kme,jms:jme)
    character,intent(in)::stag
    call set_physical_bc3d(a,stag,cf,1,nx+1,1,ny+1,1,nz+1,ims,ime,jms,jme,1,kme, &
      1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
  end subroutine
  subroutine bc2(a,stag)
    real,intent(inout)::a(ims:ime,jms:jme)
    character,intent(in)::stag
    call set_physical_bc2d(a,stag,cf,1,nx+1,1,ny+1,ims,ime,jms,jme,1,nx+1,1,ny+1,1,nx+1,1,ny+1)
  end subroutine
end program
