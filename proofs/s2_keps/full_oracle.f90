! Full unchanged EEPS/KEPS drivers; caller owns every input/output buffer.
program full_oracle
 use module_bl_keps, only: keps
 use module_bl_eeps, only: eeps
 implicit none
 integer :: count,c,n,m,scheme,steps,first,s,k,j,kp(1,1)
 real :: dt,dx,ust,rmol,wspd,hfx,qfx,tsk,qsfc,psfc,br,znt,fm,fhh
 real,allocatable :: f(:,:,:,:),o(:,:,:,:),zero(:,:,:),one(:,:,:),kh(:,:,:),km(:,:,:)
 real,allocatable :: wu(:,:,:),wv(:,:,:),wt(:,:,:),wq(:,:,:)
 real :: height(1,1)
 character(12),parameter :: names(23)=[character(12)::'u','v','theta','temperature','pressure','qv','qc','qi','qr','qs','qg', &
                  'rho','dz','raten','exner','p8w','tke','epsilon','tpe','prandtl','tke_adv','epsilon_adv','tpe_adv']
 read(*,*) count
 do c=1,count
   read(*,*) scheme,n,steps,first,dt,dx
   m=n+4
   read(*,*) ust,rmol,wspd,hfx,qfx,tsk,qsfc,psfc,br,znt,fm,fhh
   allocate(f(1,m,1,23),o(1,m,1,6),zero(1,m,1),one(1,m,1),kh(1,m,1),km(1,m,1), &
            wu(1,m,1),wv(1,m,1),wt(1,m,1),wq(1,m,1))
   f=0.;o=0.;zero=0.;one=1.;kh=0.;km=0.;wu=0.;wv=0.;wt=0.;wq=0.;height=0.;kp=0
   do j=1,23
     read(*,*) f(1,:,1,j)
   enddo
   do s=first,first+steps-1
     write(*,*) 'RECORD',c,s,scheme,n
     do j=1,23
       call emit('I_'//trim(names(j)),f(1,:,1,j))
     enddo
     o=0.
     if(scheme==16)then
       call eeps(u3d=f(:,:,:,1),v3d=f(:,:,:,2),t3d=f(:,:,:,4),qv3d=f(:,:,:,6),qc3d=f(:,:,:,7), &
         qi3d=f(:,:,:,8),qr3d=f(:,:,:,9),qs3d=f(:,:,:,10),qg3d=f(:,:,:,11),p3d=f(:,:,:,5),pi3d=f(:,:,:,15), &
         rho3d=f(:,:,:,12),rthraten=f(:,:,:,14),rublten=o(:,:,:,1),rvblten=o(:,:,:,2),rthblten=o(:,:,:,3), &
         rqvblten=o(:,:,:,4),rqcblten=o(:,:,:,5),rqiblten=o(:,:,:,6),pek=f(:,:,:,17),pep=f(:,:,:,18), &
         dz8w=f(:,:,:,13),psfc=reshape([psfc],[1,1]),qsfc=reshape([qsfc],[1,1]),tsk=reshape([tsk],[1,1]), &
         ust=reshape([ust],[1,1]),rmol=reshape([rmol],[1,1]),wspd=reshape([wspd],[1,1]), &
         xland=reshape([1.],[1,1]),hfx=reshape([hfx],[1,1]),qfx=reshape([qfx],[1,1]),dt=dt,dx=dx,itimestep=s, &
         exch_h=kh,exch_m=km,pblh=height,kpbl=kp,pek_adv=f(:,:,:,21),pep_adv=f(:,:,:,22), &
         ids=1,ide=2,jds=1,jde=2,kds=1,kde=n+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=m, &
         its=1,ite=1,jts=1,jte=1,kts=1,kte=n)
     else
       call keps(mol=reshape([rmol],[1,1]),tsk=reshape([tsk],[1,1]),xtime=s,frc_urb2d=reshape([0.],[1,1]), &
         flag_bep=.false.,dz8w=f(:,:,:,13),dt=dt,u_phy=f(:,:,:,1),v_phy=f(:,:,:,2),th_phy=f(:,:,:,3), &
         pi_phy=f(:,:,:,15),rthraten=f(:,:,:,14),p8w=f(:,:,:,16),rho=f(:,:,:,12),qv_curr=f(:,:,:,6), &
         qc_curr=f(:,:,:,7),hfx=reshape([hfx],[1,1]),qfx=reshape([qfx],[1,1]),ustar=reshape([ust],[1,1]), &
         cp=1004.5,g=9.81,rublten=o(:,:,:,1),rvblten=o(:,:,:,2),rthblten=o(:,:,:,3), &
         rqvblten=o(:,:,:,4),rqcblten=o(:,:,:,5),tke_pbl=f(:,:,:,17),diss_pbl=f(:,:,:,18), &
         tpe_pbl=f(:,:,:,19),tke_adv=f(:,:,:,21),diss_adv=f(:,:,:,22),tpe_adv=f(:,:,:,23),pr_pbl=f(:,:,:,20), &
         wu=wu,wv=wv,wt=wt,wq=wq,exch_h=kh,exch_m=km,pblh=height, &
         a_u_bep=zero,a_v_bep=zero,a_t_bep=zero,a_q_bep=zero,b_u_bep=zero,b_v_bep=zero, &
         b_t_bep=zero,b_q_bep=zero,b_e_bep=zero,sf_bep=one,vl_bep=one,br=reshape([br],[1,1]), &
         znt=reshape([znt],[1,1]),psim=reshape([fm],[1,1]),psih=reshape([fhh],[1,1]), &
         ids=1,ide=2,jds=1,jde=2,kds=1,kde=n+1,ims=1,ime=1,jms=1,jme=1,kms=1,kme=m, &
         its=1,ite=1,jts=1,jte=1,kts=1,kte=n)
     endif
     call emit('O_rublten',o(1,1:n,1,1));call emit('O_rvblten',o(1,1:n,1,2))
     call emit('O_rthblten',o(1,1:n,1,3));call emit('O_rqvblten',o(1,1:n,1,4))
     call emit('O_rqcblten',o(1,1:n,1,5));call emit('O_rqiblten',o(1,1:n,1,6))
     do j=17,23
       call emit('O_'//trim(names(j)),f(1,1:n,1,j))
     enddo
     call emit('O_exch_h',kh(1,1:n,1));call emit('O_exch_m',km(1,1:n,1))
     call emit('O_wu',wu(1,1:n,1));call emit('O_wv',wv(1,1:n,1));call emit('O_wt',wt(1,1:n,1));call emit('O_wq',wq(1,1:n,1))
     call emit('O_pblh',[height(1,1)]);call emit('O_kpbl',[real(kp(1,1))])
     ! Independent mean-column caller for carried-step oracle inputs.
     f(:,:,:,1)=f(:,:,:,1)+dt*o(:,:,:,1);f(:,:,:,2)=f(:,:,:,2)+dt*o(:,:,:,2)
     f(:,:,:,3)=f(:,:,:,3)+dt*o(:,:,:,3);f(:,:,:,4)=f(:,:,:,3)*f(:,:,:,15)
     f(:,:,:,6)=f(:,:,:,6)+dt*o(:,:,:,4);f(:,:,:,7)=f(:,:,:,7)+dt*o(:,:,:,5)
     f(:,:,:,8)=f(:,:,:,8)+dt*o(:,:,:,6)
   enddo
   deallocate(f,o,zero,one,kh,km,wu,wv,wt,wq)
 enddo
contains
 subroutine emit(label,x)
   character(*),intent(in)::label
   real,intent(in)::x(:)
   write(*,'(A)') label
   write(*,'(*(ES24.16E3,1X))') x
 end subroutine
end program
