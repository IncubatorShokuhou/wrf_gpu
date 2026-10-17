! Literal WRF microphysics-driver mediation; all constants from module_model_constants.
program wrapper
 use module_mp_milbrandt2mom, only: mp_milbrandt2mom_driver
 implicit none
 integer :: n,k,u
 real :: dt,rvord
 real,allocatable :: a(:,:,:,:),ps(:),p8w(:,:,:),dz(:,:,:),refl(:,:,:),rain(:,:,:)
 real,allocatable :: thold(:,:,:),qold(:,:,:)
 character(len=1024) :: input,output
 call get_command_argument(1,input)
 call get_command_argument(2,output)
 open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
 read(u) n,k,dt
 allocate(a(n,k,1,17),ps(n),p8w(n,k,1),dz(n,k,1),refl(n,k,1),rain(n,1,9),thold(n,k,1),qold(n,k,1))
 read(u) a,ps
 close(u)
 rvord=461.6/287.
 ! Input theta is the full moist prognostic; prep passes dry theta to MP.
 a(:,:,:,1)=a(:,:,:,1)/(1.+rvord*a(:,:,:,2))
 thold=a(:,:,:,1)
 qold=a(:,:,:,2)
 p8w=0.
 p8w(:,1,1)=ps
 dz=100.
 rain=0.
 refl=0.
 call mp_milbrandt2mom_driver(qv=a(:,:,:,2),qc=a(:,:,:,3),qr=a(:,:,:,4),qi=a(:,:,:,5), &
    qs=a(:,:,:,6),qg=a(:,:,:,7),qh=a(:,:,:,8),nc=a(:,:,:,9),nr=a(:,:,:,10),ni=a(:,:,:,11), &
    ns=a(:,:,:,12),ng=a(:,:,:,13),nh=a(:,:,:,14),th=a(:,:,:,1),pii=a(:,:,:,17), &
    p=a(:,:,:,16),w=a(:,:,:,15),dz=dz,dt_in=dt,itimestep=1,p8w=p8w, &
    rainnc=rain(:,:,1),rainncv=rain(:,:,2),snownc=rain(:,:,3),snowncv=rain(:,:,4), &
    grplnc=rain(:,:,5),grplncv=rain(:,:,6),hailnc=rain(:,:,7),hailncv=rain(:,:,8),sr=rain(:,:,9),zet=refl, &
    ids=1,ide=n+1,jds=1,jde=2,kds=1,kde=k+1,ims=1,ime=n,jms=1,jme=1,kms=1,kme=k, &
    its=1,ite=n,jts=1,jte=1,kts=1,kte=k)
 ! module_big_step_utilities_em.F:5736-5738 (full theta interface retains +T0).
 a(:,:,:,1)=thold*(1.+rvord*qold)+(a(:,:,:,1)-thold)*(1.+rvord*a(:,:,:,2)) &
            +rvord*(a(:,:,:,2)-qold)*a(:,:,:,1)
 open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
 write(u) a(:,:,:,1:14),rain,refl
 close(u)
end program
