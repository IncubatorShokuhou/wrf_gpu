program gfdl_emission
  use module_ra_gfdleta, only: table,e2spec,e3v88,clo89
  implicit none
  integer :: u,i,j,ncloud(1),top(1,9),bottom(1,9)
  real :: amount(1,9),emiss(1,9),fxo(1,2),dt(1,2)
  real :: tv(1,17),av(1,17),emv(1,17),camt(1,9),cldfac(1,9,9)
  character(len=1024) :: path
  call table
  call get_command_argument(1,path)
  open(newunit=u,file=trim(path),access='stream',form='unformatted',status='replace')
  do i=1,64
    do j=1,17
      tv(1,j)=180.25+real(mod(i*7+j*3,160))
      av(1,j)=10.**(-14.+real(mod(i+j*3,140))*.1)
    enddo
    amount=1.;emiss=0.
    do j=1,2
      amount(1,j)=av(1,j)
      fxo(1,j)=aint(tv(1,j)*.1)-9.
      dt(1,j)=tv(1,j)-10.*aint(tv(1,j)*.1)
    enddo
    call e2spec(emiss,amount,fxo,dt,16.,10.,28.,.1, &
      1,2,1,2,1,9, 1,1,1,1,1,9, 1,1,1,1,1,8)
    call e3v88(emv,tv,av,10.,.1,28.,16., &
      1,2,1,2,1,9, 1,1,1,1,1,9, 1,1,1,1,1,8)
    write(u) amount(1,1:2),fxo,dt,emiss(1,1:2),tv,av,emv
  enddo
  do i=0,8
    ncloud=i;camt=0.;top=1;bottom=1
    do j=2,9
      camt(1,j)=.05*real(j)
      top(1,j)=1+mod(i+3*j,8)
      bottom(1,j)=1+mod(i+5*j,8)
    enddo
    call clo89(cldfac,camt,ncloud,bottom,top, &
      1,2,1,2,1,9, 1,1,1,1,1,9, 1,1,1,1,1,8)
    write(u) ncloud,camt,bottom,top,cldfac
  enddo
  close(u)
end program
