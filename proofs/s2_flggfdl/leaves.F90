! Independent leaf savepoints from unchanged WRF procedures, all 121 g-points.
program flg_leaves
  use module_ra_flg
  implicit none
  integer :: nc,nz,u,v,i,b,ig,n,j,ip
  integer,parameter :: kg(18)=[10,8,12,7,12,5,2,3,4,4,3,5,2,10,12,7,7,8]
  integer,allocatable :: jd(:)
  real :: ptop,gmt,radt,dt,hk
  integer :: yr
  character(len=1024) :: input,output
  real,allocatable :: znw(:),znu(:),junk(:),sfc(:,:),fields(:,:,:)
  real,allocatable :: pp(:),pt(:),ph(:),po(:),dz(:),trp(:),tg(:),op(:),fkg(:),coef(:,:)
  real,allocatable :: dzall(:,:),trpall(:,:),ozall(:,:,:),conall(:,:,:),gasall(:,:,:),weights(:)
  real,allocatable :: rr(:),gg(:),tr(:),wr(:),ww(:,:),rainall(:,:,:,:),graupall(:,:,:,:),rayall(:,:,:,:)
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u) nc,nz,ptop,gmt,radt,dt,yr
  allocate(znw(nz+1),znu(nz+1),junk(nc),jd(nc),sfc(nc,11),fields(nc,nz+1,15))
  read(u) znw,znu,junk,junk,junk,junk,jd,sfc,fields
  close(u)
  allocate(pp(nz+1),pt(nz+1),ph(nz+1),po(nz+1),dz(nz),trp(nz),tg(nz),op(nz+1))
  allocate(dzall(nc,nz),trpall(nc,nz),ozall(nc,nz+1,5),conall(nc,nz,18),gasall(nc,nz,121),weights(121))
  allocate(rr(nz),gg(nz),tr(nz),wr(nz),ww(nz,4),rainall(nc,nz,6,18),graupall(nc,nz,6,18),rayall(nc,nz,6,18))
  do i=1,nc
    pp=fields(i,nz+1:1:-1,13)/100.
    pt=fields(i,nz+1:1:-1,5)
    ph=fields(i,nz+1:1:-1,6)
    do ip=1,5
      call o3prof(ip,1,nz+1,fields(i,:,13)/100.,op)
      ozall(i,:,ip)=op
    enddo
    po=ozall(i,nz+1:1:-1,5)
    call thicks(nz,nz+1,pp,pt,ph,po,dz)
    call rayle2(nz,nz+1,pp,pt,ph,po,trp)
    dzall(i,:)=dz;trpall(i,:)=trp
    rr=fields(i,nz:1:-1,8)*fields(i,nz:1:-1,4)*1000.
    gg=fields(i,nz:1:-1,11)*fields(i,nz:1:-1,4)*1000.
    ! Missing graupel in the real census: direct leaf boundary stress operands.
    gg(1)=0.;gg(2)=1.e-6;gg(3)=1.e-5;gg(4)=1.e-4;gg(5)=.01
    n=0
    do b=1,18
      call gascon(nz,nz+1,b,tg,pp,pt,ph,po);conall(i,:,b)=tg
      call rain(nz,nz+1,b,rr,dz,tr,wr,ww)
      rainall(i,:,1,b)=tr;rainall(i,:,2,b)=wr;rainall(i,:,3:6,b)=ww
      call graup(nz,nz+1,b,gg,dz,tr,wr,ww)
      graupall(i,:,1,b)=tr;graupall(i,:,2,b)=wr;graupall(i,:,3:6,b)=ww
      call rayle(nz,nz+1,b,trp,tr,wr,ww,.6)
      rayall(i,:,1,b)=tr;rayall(i,:,2,b)=wr;rayall(i,:,3:6,b)=ww
      do ig=1,kg(b)
        n=n+1
        call gases(nz,nz+1,b,ig,hk,pp,pt,ph,po,tg)
        gasall(i,:,n)=tg;weights(n)=hk
      enddo
    enddo
  enddo
  open(newunit=v,file=trim(output),access='stream',form='unformatted',status='replace')
  write(v) nc,nz,dzall,trpall,ozall,conall,gasall,weights,rainall,graupall,rayall
  close(v)
end program
