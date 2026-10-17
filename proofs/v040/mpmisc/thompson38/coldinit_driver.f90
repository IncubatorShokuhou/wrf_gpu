! Original module body is linked unchanged. Three terrain/spacing columns.
program coldinit38
 use module_mp_thompson, only: thompson_init
 implicit none
 integer, parameter :: nz=40
 real :: hgt(4,nz,2),orho(4,nz,2),nwfa(4,nz,2),nifa(4,nz,2)
 real :: ng(4,nz,2),nbca(4,nz,2),nwfa2d(4,2),frc(4,2)
 real :: terrain(4),spacing(4)
 integer :: i,j,k
 terrain=[0.,1500.,3000.,0.]
 spacing=[50.,125.,200.,50.]
 do j=1,2
 do k=1,nz
 do i=1,4
  hgt(i,k,j)=terrain(i)+real(k-1)*spacing(i)
  orho(i,k,j)=(287.04*(290.-real(k))*(.008+.622))/(.622*(100000.-real(k)*1000.))
 enddo
 enddo
 enddo
 nwfa=0.;nifa=0.;ng=0.;nbca=0.;nwfa2d=0.;frc=0.
 call initialize
 do i=1,3
 write(*,'(A,I0,A,ES25.16E3)') 'EMISSION[',i,']=',nwfa2d(i,1)
 do k=1,nz
 write(*,'(A,I0,A,I0,A,ES25.16E3)') 'HGT[',i,',',k,']=',hgt(i,k,1)
 write(*,'(A,I0,A,I0,A,ES25.16E3)') 'ORHO[',i,',',k,']=',orho(i,k,1)
 write(*,'(A,I0,A,I0,A,ES25.16E3)') 'NWFA[',i,',',k,']=',nwfa(i,k,1)
 write(*,'(A,I0,A,I0,A,ES25.16E3)') 'NIFA[',i,',',k,']=',nifa(i,k,1)
 enddo
 enddo
 ! Positive data leaves the provided emission untouched on a later init.
 nwfa2d=123.25
 call initialize
 do i=1,3
 write(*,'(A,I0,A,ES25.16E3)') 'HELD_EMISSION[',i,']=',nwfa2d(i,1)
 enddo
contains
 subroutine initialize
 call thompson_init(HGT=hgt,ORHO=orho,NWFA2D=nwfa2d,NWFA=nwfa,NIFA=nifa,NBCA=nbca,NG=ng, &
 wif_input_opt=1,FRC_URB2D=frc,DX=3000.,DY=3000.,is_start=.true., &
 ids=1,ide=4,jds=1,jde=2,kds=1,kde=nz+1,ims=1,ime=4,jms=1,jme=2,kms=1,kme=nz, &
 its=1,ite=4,jts=1,jte=2,kts=1,kte=nz)
 end subroutine
end program
