! Matched-input pristine transfer replay for source FMA-vs-no-FMA bounds.
program transport_replay
  use module_ra_flg, only: qfts,qftisf
  use control_para, only: quadra,edding,hemisp
  implicit none
  integer :: nr,nz,i,u
  integer,allocatable :: bands(:)
  real,allocatable :: parameters(:,:),bf(:,:),optics(:,:,:),flux(:,:,:)
  real,allocatable :: up(:),down(:),direct(:),diffuse(:)
  character(len=1024) :: input,output
  call get_command_argument(1,input);call get_command_argument(2,output)
  open(newunit=u,file=trim(input),access='stream',form='unformatted',status='old')
  read(u) nr,nz
  allocate(bands(nr),parameters(nr,5),bf(nr,nz+1),optics(nr,nz,6),flux(nr,nz+1,2))
  allocate(up(nz+1),down(nz+1),direct(nz+1),diffuse(nz+1))
  read(u) bands,parameters,bf,optics
  close(u)
  do i=1,nr
    if(bands(i)<=6)then
      call qfts(nz,nz+1,bands(i),parameters(i,1),parameters(i,2),parameters(i,3), &
          optics(i,:,1),optics(i,:,2),optics(i,:,3),optics(i,:,4),optics(i,:,5),optics(i,:,6), &
          up,down,direct,diffuse)
    else
      quadra=.false.;edding=.false.;hemisp=.true.
      call qftisf(nz,nz+1,bands(i),parameters(i,4),bf(i,:),parameters(i,5), &
          optics(i,:,1),optics(i,:,2),optics(i,:,3),optics(i,:,4),optics(i,:,5),optics(i,:,6),up,down)
    endif
    flux(i,:,1)=up;flux(i,:,2)=down
  enddo
  open(newunit=u,file=trim(output),access='stream',form='unformatted',status='replace')
  write(u) nr,nz,flux
  close(u)
end program
