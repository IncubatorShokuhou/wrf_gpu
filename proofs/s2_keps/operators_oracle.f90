! Caller-explicit deterministic subroutines, not the undefined full PBL drivers.
program operators_oracle
 use module_bl_keps, only: shear,buoy,cdtur_ke,invert,diff,terms_tpe, &
                          calc_countergradient,thetaphi_alphabeta
 use module_bl_eeps, only: tridiag
 implicit none
 integer :: n,m,count,j,k,first,case_index
 real :: dt,z0,ustar,pe,pm
 real,allocatable :: f(:,:),sh(:),bu(:),km(:),kh(:),a(:,:),rhs(:),sol(:),fc(:)
 real,allocatable :: at(:),bt(:),coeff(:),te(:),ep(:),src(:),eec(:,:),eed(:,:)
 read(*,*) count
 do case_index=1,count
   read(*,*) n,dt,z0,ustar,pe,pm
   m=n+1
   allocate(f(m,26),sh(m),bu(m),km(m),kh(m),a(m,3),rhs(m),sol(m),fc(m), &
            at(m),bt(m),coeff(m),te(m),ep(m),src(m),eec(1,n),eed(1,n))
   do j=1,26
     read(*,*) f(:,j)
   enddo
   sh=0.;bu=0.;km=0.;kh=0.;a=0.;sol=0.;fc=0.;at=0.;bt=0.;coeff=0.
   call shear(1,m,1,n,f(:,1),f(:,2),f(:,5),sh,f(:,20))
   call emit('shear',sh(1:n))
   call buoy(9.81,1,m,1,n,f(:,3),f(:,4),f(:,5),bu,f(:,7),f(:,9),f(:,10))
   call emit('buoyancy',bu(1:n))
   call cdtur_ke(1,m,1,n,f(:,7),f(:,8),f(:,24),f(:,5),km,kh,f(:,21))
   call emit('exchange_m',km)
   call emit('exchange_h',kh)
   a(:,1)=f(:,15);a(:,2)=f(:,16);a(:,3)=f(:,17);rhs=f(:,18)
   call invert(1,m,1,n,a,rhs,sol)
   call emit('invert',sol(1:n))
   do first=1,2
     sol=f(:,18);fc=0.
     call diff(1,m,1,n,first,1,dt,sol,f(:,6),f(:,22),f(:,23), &
               f(:,14),f(:,11),f(:,20),f(:,19),f(:,5),fc)
     if(first==1)then
       call emit('diffusion_1',sol(1:n));call emit('flux_1',fc(1:n))
     else
       call emit('diffusion_2',sol(1:n));call emit('flux_2',fc(1:n))
     endif
   enddo
   call terms_tpe(f(:,11),9.81,1,m,1,n,f(:,4),f(:,7),f(:,8),f(:,9),at,bt,f(:,10),coeff)
   call emit('tpe_implicit',at(1:n));call emit('tpe_explicit',bt(1:n));call emit('tpe_coeff',coeff(1:n))
   bt=f(:,11)
   call calc_countergradient(f(:,5),9.81,1,m,1,n,f(:,4),f(:,7),f(:,8),f(:,9),f(:,10),bt)
   call emit('countergradient',bt(1:n))
   te=f(:,7);ep=f(:,8);src=f(:,14)
   call thetaphi_alphabeta(1,m,1,n,z0,f(:,5),f(:,25),f(:,26),te,ep,dt,f(:,10), &
                         f(:,24),ustar,pe,pm,f(:,13),f(:,12),src)
   call emit('analytic_tke',te(1:n));call emit('analytic_epsilon',ep(1:n));call emit('analytic_source',src(1:n))
   eec(1,:)=f(1:n,15);eed(1,:)=f(1:n,18)
   ! EEPS a=upper/c=lower; deliberately nonzero c(1) exercises overwrite.
   call tridiag(reshape(f(1:n,17),[1,n]),reshape(f(1:n,16),[1,n]),eec,eed,1,n)
   call emit('eeps_solution',eed(1,:));call emit('eeps_modified_c',eec(1,:))
   deallocate(f,sh,bu,km,kh,a,rhs,sol,fc,at,bt,coeff,te,ep,src,eec,eed)
 enddo
contains
 subroutine emit(label,x)
   character(*),intent(in)::label
   real,intent(in)::x(:)
   write(*,'(A)') label
   write(*,'(*(ES24.16E3,1X))') x
 end subroutine
end program
