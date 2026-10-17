program oracle
use module_stoch
use module_configure
use module_domain
implicit none
integer,parameter :: nx=18, ny=16, nz=5
type(grid_config_rec_type) :: config
type(domain) :: grid
integer :: seed_dim, seed(8),before(8),after(8),i,j,k,n,ikmax,ilmax
real :: amp(nx,ny),s(nx,ny),c(nx,ny),zeros(nx,ny),alpha,z
real :: vc(nx,nz,ny),vs(nx,nz,ny),va(nz)
real :: noise_s(nx,ny),noise_c(nx,ny),field(nx,nz,ny)
real :: u(nx,nz,ny),v(nx,nz,ny),t(nx,nz,ny),q(nx,nz,ny),r(nx,nz,ny)
real :: pu(nx,nz,ny),pv(nx,nz,ny),pt(nx,nz,ny)
real :: mu(nx,ny),mub(nx,ny),c1(nz),c2(nz)
character :: component
character(7),parameter :: variables='WTPRSQO'
open(20,file='oracle.bin',access='stream',form='unformatted',status='replace')
config%start_year=2026
config%start_month=10
config%start_day=13
config%start_hour=18
config%nens=7
call random_seed(size=seed_dim)
if(seed_dim/=8) stop 8
call rand_seed(config,811,seed,1,seed_dim)
write(20) seed
grid%i_start=1
grid%i_end=nx
grid%j_start=1
grid%j_end=ny
allocate(grid%rand_real(nx,nz,ny),grid%rand_imag(nx,nz,ny))
vc=1.
vs=0.
call setup_rand_perturb('P',1,.false.,amp,c,s,alpha,vc,vs,va, &
  1,1000000,1,1000000,ikmax,ilmax,54,9000.,9000.,0.5,15000.,21600.,1.e-6,10800.,-1.83, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
write(20) amp,alpha,vc,vs,va
do j=1,ny
do i=1,nx
  s(i,j)=real(2*i-j)*0.003
  c(i,j)=real(i+3*j)*0.002
enddo
enddo
write(20) s,c
zeros=0.
do n=1,4
  before=seed
  call random_seed(put=before)
  do j=1,ny
  do i=1,nx
    do
      call gauss_noise(z)
      if(abs(z)<3.) exit
    enddo
    noise_s(i,j)=z
    do
      call gauss_noise(z)
      if(abs(z)<3.) exit
    enddo
    noise_c(i,j)=z
  enddo
  enddo
  call update_stoch(s,c,amp,alpha,.false.,seed,seed_dim, &
    1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
  after=seed
  write(20) noise_s,noise_c,s,c,seed
  do k=0,1
  do i=1,3
    if(i==1) component='T'
    if(i==2) component='U'
    if(i==3) component='V'
    call rand_pert_update(grid,component,s,c,zeros,0., &
      1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nz, &
      1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz, &
      nz,nz,nz,nz,.false.,seed,seed_dim,9000.,9000.,k,field,1.e10,1.,vc,vs,va)
    write(20) field
    seed=after
  enddo
  enddo
enddo
do n=1,7
  call setup_rand_perturb(variables(n:n),1,.false.,amp,c,s,alpha,vc,vs,va, &
    1,1000000,1,1000000,ikmax,ilmax,54,9000.,9000.,0.5,15000.,21600.,1.e-6,10800.,-1.83, &
    1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
  write(20) amp,alpha,vc,vs,va
enddo
do k=1,nz
  c1(k)=real(nz-k)*0.17
  c2(k)=real(k)*2000.
  do j=1,ny
  do i=1,nx
    u(i,k,j)=real(i-j)*0.013
    v(i,k,j)=real(i+j)*0.021
    t(i,k,j)=real(k-i)*0.007
    q(i,k,j)=real(j-k)*0.0003
    pu(i,k,j)=real(i-2*j)*0.00001
    pv(i,k,j)=real(j-2*i)*0.00002
    pt(i,k,j)=real(i+j-k)*0.000001
    r(i,k,j)=real(mod(i+2*j+k,7)-3)*0.6
    mu(i,j)=real(i-j)*1.25
    mub(i,j)=75000.+real(2*i+j)*17.
  enddo
  enddo
enddo
write(20) u,v,t,q,pu,pv,pt,r,mu,mub,c1,c2
call update_stoch_ten(u,v,t,pu,pv,pt,mu,mub,c1,c2, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,nz,nz)
write(20) u,v,t
call perturb_physics_tend(0.5,2.,r,u,v,t,q, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,nz,nz)
write(20) u,v,t,q,r
close(20)
end program
