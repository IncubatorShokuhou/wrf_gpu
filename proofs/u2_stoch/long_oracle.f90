program oracle
use module_stoch
use module_configure
use module_domain
implicit none
integer,parameter::nx=18,ny=16,nz=5
type(grid_config_rec_type)::config
type(domain)::grid
integer::seed_dim,seed(8),n,m,ikmax,ilmax
real::amp(nx,ny),s(nx,ny),c(nx,ny),alpha
real::vc(nx,nz,ny),vs(nx,nz,ny),va(nz)
real::field(nx,nz,ny),secondary(nx,nz,ny)
character::component
character(7),parameter::variables='WTPRSQO'
config%start_year=2026;config%start_month=10;config%start_day=13;config%start_hour=18;config%nens=7
grid%i_start=1;grid%i_end=nx;grid%j_start=1;grid%j_end=ny
allocate(grid%rand_real(nx,nz,ny),grid%rand_imag(nx,nz,ny))
call random_seed(size=seed_dim)
if(seed_dim/=8)stop 8
open(20,file='oracle.bin',access='stream',form='unformatted',status='replace')
do m=1,7
vc=1.;vs=0.
s=0.;c=0.
call setup_rand_perturb(variables(m:m),1,.false.,amp,c,s,alpha,vc,vs,va, &
  1,1000000,1,1000000,ikmax,ilmax,54,9000.,9000.,0.5,15000.,21600.,1.e-6,10800.,-1.83, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz)
call rand_seed(config,811,seed,1,seed_dim)
component='T'
if(m==1)component='U'
do n=1,1000
call rand_pert_update(grid,component,s,c,amp,alpha, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nz, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz, &
  nz,nz,nz,nz,.false.,seed,seed_dim,9000.,9000.,1,field,2.,.5,vc,vs,va)
if(m==1)then
call rand_pert_update(grid,'V',s,c,amp,alpha, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nz, &
  1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz, &
  nz,nz,nz,nz,.false.,seed,seed_dim,9000.,9000.,1,secondary,2.,.5,vc,vs,va)
else
secondary=field
endif
if(mod(n,10)==0)write(20)s,c,field,secondary,seed
enddo
enddo
close(20)
end program
