program error_driver
 use obs_full_oracle
 implicit none
 integer nx,ny,nz,nobs,ncalls,c,wind,temp,mois
 real,allocatable :: u(:,:,:),v(:,:,:),tb(:,:,:),qv(:,:,:),pb(:,:,:),pp(:,:,:),z(:,:,:)
 real,allocatable :: ur(:,:),vr(:,:),tr(:,:),varobs(:,:),errf(:,:),rko(:),rio(:),rjo(:),timeob(:)
 integer,allocatable :: kpbl(:,:)
 integer lev(1),par(1),obsprt(1),stnid(40,1)
 real diagnostic(1)
 character(1024) infile,outfile
 call get_command_argument(1,infile)
 call get_command_argument(2,outfile)
 open(10,file=trim(infile),access='stream',form='unformatted',status='old')
 read(10) nx,ny,nz,nobs,ncalls,wind,temp,mois
 allocate(u(nx+1,nz+1,ny+1),v(nx+1,nz+1,ny+1),tb(nx+1,nz+1,ny+1),qv(nx+1,nz+1,ny+1))
 allocate(pb(nx+1,nz+1,ny+1),pp(nx+1,nz+1,ny+1),z(nx+1,nz+1,ny+1))
 allocate(ur(nx+1,ny+1),vr(nx+1,ny+1),tr(nx+1,ny+1),kpbl(nx+1,ny+1))
 allocate(varobs(6,nobs),errf(10,nobs),rko(nobs),rio(nobs),rjo(nobs),timeob(nobs))
 u=0.;v=0.;tb=0.;qv=0.;pb=0.;pp=0.;z=0.;ur=0.;vr=0.;tr=0.;kpbl=0
 read(10) u(:,1:nz,1:ny),v(1:nx,1:nz,:),tb(1:nx,1:nz,1:ny),qv(1:nx,1:nz,1:ny)
 read(10) pb(1:nx,1:nz,1:ny),pp(1:nx,1:nz,1:ny),z(1:nx,1:nz,1:ny)
 read(10) ur(1:nx,1:ny),vr(1:nx,1:ny),tr(1:nx,1:ny),kpbl(1:nx,1:ny)
 read(10) rio,rjo,rko,varobs
 lev=0;par=0;diagnostic=0.;stnid=0;obsprt=-999;timeob=0.
 open(11,file=trim(outfile),access='stream',form='unformatted',status='replace')
 do c=1,ncalls
  call errob(1,u,v,tb,300.,qv,pb,pp,287./1004.5,z,ur,vr,tr,kpbl,6,10,nobs,1,lev,par,nobs,nobs, &
   wind,temp,mois,0,timeob,rio,rjo,rko,varobs,errf,c,0.,1,0,1,1,.false.,obsprt,stnid, &
   diagnostic,diagnostic,diagnostic,diagnostic,1,nx+1,1,ny+1,1,nz+1,1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz)
  write(11) varobs,rko,errf
 enddo
 close(10);close(11)
end program
