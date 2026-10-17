program cloud_oracle
  implicit none
  real :: qv(1,1,1),qc(1,1,1),qi(1,1,1),qs(1,1,1),t(1,1,1),p(1,1,1),cf(1,1,1),zero(1,1,1)
  integer :: flag(1,1,1),mp,ios
  logical :: fqi,fqs
  zero=0.
  do
    read(*,*,iostat=ios) mp,t,p,qv,qc,qi,qs
    if(ios/=0)exit
    fqi=mp/=1.and.mp/=3
    fqs=fqi.and.mp/=50.and.mp/=51
    call cal_cldfra1(cf,qv,qc,qi,qs,.true.,.true.,fqi,fqs,t,p,zero,zero,mp,flag, &
      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    write(*,'(es25.17,1x,i2)') cf,flag
  enddo
end program
