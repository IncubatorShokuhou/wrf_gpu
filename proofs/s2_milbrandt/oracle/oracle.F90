program oracle
 use my2_mod, only: mp_milbrandt2mom_main
 implicit none
 integer :: nc,nk,i,j,unit
 real :: dt
 real,allocatable :: a(:,:,:),ps(:),sigma(:,:),diam(:,:,:),zet(:,:),zec(:),ss(:,:,:),rates(:,:)
 character(len=1024) :: input,output,stage
 logical :: sedi
 call get_command_argument(1,input)
 call get_command_argument(2,output)
 call get_command_argument(3,stage)
 sedi=trim(stage)/='local'
 open(newunit=unit,file=trim(input),access='stream',form='unformatted',status='old')
 read(unit) nc,nk,dt
 allocate(a(nc,nk,16),ps(nc),sigma(nc,nk),diam(nc,nk,6),zet(nc,nk),zec(nc),ss(nc,nk,20),rates(nc,11))
 read(unit) a,ps
 close(unit)
 do i=1,nc
   sigma(i,:)=a(i,:,16)/ps(i)
 enddo
 call mp_milbrandt2mom_main(a(:,:,15),a(:,:,1),a(:,:,2),a(:,:,3),a(:,:,4),a(:,:,5), &
      a(:,:,6),a(:,:,7),a(:,:,8),a(:,:,9),a(:,:,10),a(:,:,11),a(:,:,12),a(:,:,13),a(:,:,14), &
      ps,sigma,rates(:,1),rates(:,2),rates(:,3),rates(:,4),rates(:,5),rates(:,6),rates(:,7), &
      rates(:,8),rates(:,9),rates(:,10),rates(:,11),dt,nc,nk,1,1,2, &
      .true.,sedi,.true.,.true.,.true.,.true., &
      diam(:,:,1),diam(:,:,2),diam(:,:,3),diam(:,:,4),diam(:,:,5),diam(:,:,6),zet,zec,ss,.false.)
 open(newunit=unit,file=trim(output),access='stream',form='unformatted',status='replace')
 write(unit) a(:,:,1:14),rates,diam,zet,zec
 close(unit)
end program
