! Read-only post-RK observer. Never writes the caller's arrays.
subroutine f2_observe(a, fid, step, ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme)
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  integer, intent(in) :: fid,step,ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme
  real, intent(in) :: a(ims:ime,kms:kme,jms:jme)
  integer :: i,j,k,n,nf,u
  integer :: samples(4,(ide-ids)*(jde-jds)*(kde-kds))
  logical, save :: opened(10)=.false.
  character(32) :: filename
  u=9200+fid
  if (.not.opened(fid)) then
    write(filename,'("f2_field_",I2.2,".bin")') fid
    open(unit=u,file=trim(filename),access='stream',form='unformatted',status='replace')
    opened(fid)=.true.
  endif
  n=0
  nf=0
  do k=kds,kde-1
    do j=jds,jde-1
      do i=ids,ide-1
        if (.not.ieee_is_finite(a(i,k,j))) nf=nf+1
        if (a(i,k,j)<0.) then
          n=n+1
          samples(:,n)=[i-ids,j-jds,k-kds,transfer(a(i,k,j),0)]
        endif
      enddo
    enddo
  enddo
  write(u) step,n,nf
  if(n>0) write(u) samples(:,1:n)
end subroutine f2_observe
