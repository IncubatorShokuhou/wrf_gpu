program helpers
  use obs_oracle
  implicit none
  character(1024) :: path,outpath
  integer :: n,i
  real,allocatable :: args(:,:),full(:),ramp(:),pblh(:),h1(:),h2(:),tw(:)
  real :: hmax,fullmin,rampmin
  call get_command_argument(1,path)
  call get_command_argument(2,outpath)
  open(10,file=trim(path),access='stream',form='unformatted',status='old')
  read(10) n
  allocate(args(5,n),full(n),ramp(n),pblh(n),h1(n),h2(n),tw(n))
  read(10) args,full,ramp,pblh,hmax,fullmin,rampmin
  close(10)
  do i=1,n
    tw(i)=get_timewt(args(1,i),args(2,i),args(3,i),args(4,i),args(5,i))
  enddo
  call get_vif_hts_slab(full,ramp,pblh,hmax,fullmin,rampmin,h1,h2,1,n,1,n)
  open(11,file=trim(outpath),access='stream',form='unformatted',status='replace')
  write(11) tw,h1,h2
  close(11)
end program
