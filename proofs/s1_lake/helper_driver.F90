! Unmodified module's explicit DOUBLE helpers, independent of the JAX port.
program lake_helper_oracle
  use module_sf_lake, only: r8, QSat, Combo, MoninObukIni, Tridiagonal
  implicit none
  real(r8) :: t,p,es,est,qs,qst, dz,wliq,wice,temp, dz2,wliq2,wice2,temp2
  real(r8) :: ur,thv,dthv,zldis,z0m,um,obu
  real(r8) :: a(3,25),b(3,25),c(3,25),r(3,25),u(3,25),seed(3,25)
  real(r8) :: temps(14),combo_inputs(8,4), monin_inputs(5,7)
  integer :: j,k,ci,tops(3),filter(2)
  temps=[150._r8,185._r8,198.16_r8,230._r8,260._r8,273.159999_r8, &
         273.16_r8,273.160001_r8,290._r8,320._r8,360._r8,373.16_r8,400._r8,450._r8]
  do k=1,size(temps)
    t=temps(k); p=90000._r8
    call QSat(t,p,es,est,qs,qst)
    write(*,'(A,6(1X,ES26.17E3))') 'QSAT',t,p,es,est,qs,qst
  enddo
  ! Negative enthalpy, mixed phase, positive excess, very small layer mass.
  combo_inputs(:,1)=[.02_r8,0._r8,4._r8,250._r8,.04_r8,0._r8,9._r8,265._r8]
  combo_inputs(:,2)=[.02_r8,1._r8,4._r8,270._r8,.04_r8,2._r8,9._r8,273.16_r8]
  combo_inputs(:,3)=[.02_r8,1._r8,0._r8,290._r8,.04_r8,2._r8,0._r8,280._r8]
  combo_inputs(:,4)=[.0001_r8,.0001_r8,.0002_r8,260._r8,.0002_r8,0._r8,.0001_r8,270._r8]
  do k=1,4
    dz=combo_inputs(1,k); wliq=combo_inputs(2,k); wice=combo_inputs(3,k); temp=combo_inputs(4,k)
    dz2=combo_inputs(5,k); wliq2=combo_inputs(6,k); wice2=combo_inputs(7,k); temp2=combo_inputs(8,k)
    call Combo(dz,wliq,wice,temp,dz2,wliq2,wice2,temp2)
    write(*,'(A,12(1X,ES26.17E3))') 'COMBO',combo_inputs(:,k),dz,wliq,wice,temp
  enddo
  monin_inputs(:,1)=[4._r8,290._r8,2._r8,20._r8,.001_r8]
  monin_inputs(:,2)=[4._r8,290._r8,-2._r8,20._r8,.001_r8]
  monin_inputs(:,3)=[4._r8,290._r8,0._r8,20._r8,.001_r8]
  monin_inputs(:,4)=[0._r8,290._r8,2._r8,20._r8,.001_r8]
  monin_inputs(:,5)=[0._r8,290._r8,-20._r8,20._r8,.001_r8]
  monin_inputs(:,6)=[50._r8,290._r8,0.000001_r8,20._r8,.001_r8]
  monin_inputs(:,7)=[50._r8,290._r8,-0.000001_r8,20._r8,.001_r8]
  do k=1,7
    ur=monin_inputs(1,k); thv=monin_inputs(2,k); dthv=monin_inputs(3,k)
    zldis=monin_inputs(4,k); z0m=monin_inputs(5,k)
    call MoninObukIni(ur,thv,dthv,zldis,z0m,um,obu)
    write(*,'(A,7(1X,ES26.17E3))') 'MONIN',monin_inputs(:,k),um,obu
  enddo
  do j=1,25
    do ci=1,3
      a(ci,j)=-.1_r8*(j+ci); c(ci,j)=-.15_r8*(j+ci)
      b(ci,j)=10._r8+j+ci; r(ci,j)=3._r8*j-ci
      seed(ci,j)=-777._r8+j
    enddo
  enddo
  tops=[1,6,11]; filter=[1,2]; u=seed
  call Tridiagonal(1,3,1,25,tops,2,filter,a,b,c,r,u)
  write(*,'(A,*(1X,ES26.17E3))') 'TRI_A',a
  write(*,'(A,*(1X,ES26.17E3))') 'TRI_B',b
  write(*,'(A,*(1X,ES26.17E3))') 'TRI_C',c
  write(*,'(A,*(1X,ES26.17E3))') 'TRI_R',r
  write(*,'(A,*(1X,ES26.17E3))') 'TRI_SEED',seed
  write(*,'(A,*(1X,ES26.17E3))') 'TRI_U',u
end program
