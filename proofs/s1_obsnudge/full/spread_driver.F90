! Original NUDOB with grouped soundings and optional negative-QV scaling.
program spread_driver
 use obs_full_oracle
 implicit none
 integer nx,nz,ny,nobs,ncalls,ivar,j,ionf,enabled,nudge,horiz,vert,scale,c,ktau,ktaur,rest
 integer levidn(1),parid(1)
 real dx,rinxy,twindo,dtmin,coef,sfcfact,sfcfacr,dpsmx,xtime,tfaci,maxgap
 real,allocatable :: rio(:),rjo(:),rko(:),timeob(:),varobs(:,:),errf(:,:),lev(:),plfo(:),nlevs(:)
 real,allocatable :: mu(:),msfx(:),msfy(:),c1h(:),c2h(:),pb(:,:),pp(:,:),z(:,:),qv(:,:,:),qslab(:,:)
 real,allocatable :: terrain(:),vih1(:),vih2(:),aten(:,:),saved(:,:,:)
 integer,allocatable :: kpbl(:)
 character(1024) inpath,outpath
 call get_command_argument(1,inpath);call get_command_argument(2,outpath)
 open(10,file=trim(inpath),access='stream',form='unformatted',status='old')
 read(10) nx,nz,ny,nobs,ncalls,ivar,j,ionf,enabled,nudge,horiz,vert,scale
 read(10) dx,rinxy,twindo,dtmin,coef,sfcfact,sfcfacr,dpsmx,maxgap
 allocate(rio(nobs),rjo(nobs),rko(nobs),timeob(nobs),varobs(6,nobs),errf(10,nobs))
 allocate(lev(nobs),plfo(nobs),nlevs(nobs),mu(nx),msfx(nx),msfy(nx),c1h(nz),c2h(nz))
 allocate(pb(nx,nz),pp(nx,nz),z(nx,nz),qv(nx,nz,ny),qslab(nx,nz),terrain(nx),vih1(nx),vih2(nx))
 allocate(aten(nx,nz),saved(6,nx,nz),kpbl(nx))
 read(10) rio,rjo,rko,timeob,varobs,errf,lev,nlevs
 read(10) mu,c1h,c2h,msfx,msfy,pb,pp,kpbl,vih1,vih2,terrain,z,qslab
 read(10) aten,saved
 do c=1,ny
  qv(:,:,c)=qslab
 enddo
 plfo=1.;levidn=0;parid=0
 open(11,file=trim(outpath),access='stream',form='unformatted',status='replace')
 do c=1,ncalls
  read(10) ktau,ktaur,rest
  read(10) xtime,tfaci
  call nudob(j,ivar,aten,1,rest==1,ktau,ktaur,xtime,mu,c1h,c2h,msfx,msfy, &
   6,10,nobs,1,0,ionf,rinxy,twindo,nudge==1,sfcfact,sfcfacr,levidn,parid,nobs, &
   1.,2.,50.,1./dpsmx,tfaci,horiz,vert,maxgap,lev,plfo,nlevs,1,dx,dtmin, &
   rio,rjo,rko,timeob,varobs,errf,pb,5000.,pp,enabled,enabled,enabled,coef,coef,coef, &
   saved,kpbl,0,vih1,vih2,terrain,z,.false., &
   1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz,1,nx,1,ny,1,nz,qv,scale)
  write(11) aten,saved(ivar,:,:)
 enddo
 close(10);close(11)
end program
