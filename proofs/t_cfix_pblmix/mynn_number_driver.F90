! =====================================================================
! t-cfix-pblmix F8: MYNN number-scalar mixing oracle (QNC, QNI, QNWFA, QNIFA)
! on REAL CPU-WRF mp=28 columns.
!
! Calls the REAL WRF DMP_mf + mynn_tendencies of module_bl_mynnedmf with the
! caller flags of an mp=28 run (FLAG_QC/QI/QNC/QNI/QNWFA/QNIFA, mixscalars=1,
! cloudmix=1, mixqt=0, edmf=1, edmf_mom=1; module_pbl_driver.F:1715-1722,
! mynnedmf_driver.F:456-475). Linked either against the pristine build objects
! (bytes of the WRF tendencies) or, with -DCAPTURE, against a private
! WRITE-only copy that records the internal post-solve sqc2/sqi2/q*2 arrays.
! dfh/dfm/dfq: the deterministic kh_profile of proofs/mynn_edmf/fortran_oracle
! (both sides consume the dumped dfh).  argv(1) = column text file.
! =====================================================================
program mynn_number_oracle
  use module_bl_mynnedmf_common, only: kind_phys, p608, xlvcp
  use module_bl_mynnedmf, only: dmp_mf, mynn_tendencies
#ifdef CAPTURE
  use module_bl_mynnedmf, only: cap_sqc2, cap_sqi2, cap_qnc2, cap_qni2, cap_qnwfa2, cap_qnifa2
#endif
  implicit none
  integer, parameter :: NZ = 200
  integer :: ncol, kx, c, k, kts, kte, kpbl, ktop_plume, spp_pbl, nchem
  real(kind_phys) :: s(11), row(17)
  real(kind_phys) :: ust, pblh, psfc, xland, flt, flq, flqv, fltv, th_sfc, dx, delt, wspd, flqc
  real(kind_phys) :: uoce, voce, maxmf, ztop_plume, maxwidth, psig_shcu, dzk
  real(kind_phys), dimension(NZ) :: u,v,w,th,tk,p,exner,rho,dz,qv,qc,qi,qs,qke,qnc,qni,qnwfa,qnifa,qnbca,ozone
  real(kind_phys), dimension(NZ) :: thl,thv,sqv,sqc,sqi,sqs,sqw,tsq,qsq,cov,tcd,qcd,dfm,dfh,dfq,diss_heat
  real(kind_phys), dimension(NZ) :: cldfra_bl1,qc_bl1,qi_bl1,cldfra_bl1_old,qc_bl1_old,vt1,vq1,sgm1,el1
  real(kind_phys), dimension(NZ) :: tkeprod_up,pattern_spp_pbl
  real(kind_phys), dimension(NZ) :: Du,Dv,Dth,Dqv,Dqc,Dqi,Dqs,Dqnc,Dqni,Dqnwfa,Dqnifa,Dqnbca,Dozone
  real(kind_phys), dimension(NZ+1) :: zw,s_aw,s_awthl,s_awqt,s_awqv,s_awqc,s_awu,s_awv,s_awqke
  real(kind_phys), dimension(NZ+1) :: s_awqnc,s_awqni,s_awqnwfa,s_awqnifa,s_awqnbca
  real(kind_phys), dimension(NZ+1) :: sd_aw,sd_awthl,sd_awqt,sd_awqv,sd_awqc,sd_awqi,sd_awqnc,sd_awqni
  real(kind_phys), dimension(NZ+1) :: sd_awqnwfa,sd_awqnifa,sd_awu,sd_awv
  real(kind_phys), dimension(NZ) :: sub_thl,sub_sqv,sub_u,sub_v,det_thl,det_sqv,det_sqc,det_u,det_v
  real(kind_phys), dimension(NZ) :: edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc
  real(kind_phys), dimension(NZ,1) :: chem1
  real(kind_phys), dimension(NZ+1,1) :: s_awchem1
  logical :: mix_chem
  character(len=512) :: path

  call get_command_argument(1, path)
  open(10, file=trim(path), status='old', action='read')
  read(10,*) ncol, kx
  kts = 1; kte = kx; nchem = 1
  write(*,'(A,I0,1X,I0)') 'NCOL_KX ', ncol, kx
  do c = 1, ncol
    read(10,*) s
    ust=s(1); pblh=s(2); psfc=s(3); xland=s(4); flt=s(5); flq=s(6); flqv=s(7); fltv=s(8)
    th_sfc=s(9); dx=s(10); delt=s(11)
    do k = 1, kx
      read(10,*) row
      u(k)=row(1); v(k)=row(2); w(k)=row(3); th(k)=row(4); tk(k)=row(5); p(k)=row(6)
      exner(k)=row(7); rho(k)=row(8); dz(k)=row(9); qv(k)=row(10); qc(k)=row(11); qi(k)=row(12)
      qke(k)=row(13); qnc(k)=row(14); qni(k)=row(15); qnwfa(k)=row(16); qnifa(k)=row(17)
    end do
    wspd = sqrt(u(1)**2 + v(1)**2)
    zw(kts) = 0.0_kind_phys
    do k = kts, kte   ! module_bl_mynnedmf.F:828-840 / mynnedmf_pre_run
      zw(k+1) = zw(k) + dz(k)
      sqv(k) = qv(k)/(1.0_kind_phys + qv(k)); sqc(k) = qc(k)/(1.0_kind_phys + qv(k))
      sqi(k) = qi(k)/(1.0_kind_phys + qv(k)); sqs(k) = 0.0_kind_phys; qs(k) = 0.0_kind_phys
      sqw(k) = sqv(k) + sqc(k) + sqi(k)
      thl(k) = th(k) - xlvcp/exner(k)*sqc(k)
      thv(k) = th(k)*(1.0_kind_phys + p608*sqv(k))
      qnbca(k)=0.; ozone(k)=0.; tsq(k)=0.; qsq(k)=0.; cov(k)=0.; cldfra_bl1(k)=0.; qc_bl1(k)=0.
      qi_bl1(k)=0.; cldfra_bl1_old(k)=0.; qc_bl1_old(k)=0.; vt1(k)=0.; vq1(k)=0.; sgm1(k)=0.
      diss_heat(k)=0.; el1(k)=50.; tkeprod_up(k)=0.; pattern_spp_pbl(k)=0.
    end do
    dfm(kts)=0.; dfh(kts)=0.; dfq(kts)=0.
    do k = kts+1, kte
      dzk = 0.5_kind_phys*(dz(k)+dz(k-1))
      call kh_profile(zw(k), pblh, ust, dfh(k))
      dfh(k) = dfh(k)/dzk; dfm(k) = dfh(k); dfq(k) = dfh(k)
    end do
    s_aw=0.; s_awthl=0.; s_awqt=0.; s_awqv=0.; s_awqc=0.; s_awu=0.; s_awv=0.; s_awqke=0.
    s_awqnc=0.; s_awqni=0.; s_awqnwfa=0.; s_awqnifa=0.; s_awqnbca=0.
    sd_aw=0.; sd_awthl=0.; sd_awqt=0.; sd_awqv=0.; sd_awqc=0.; sd_awqi=0.; sd_awqnc=0.; sd_awqni=0.
    sd_awqnwfa=0.; sd_awqnifa=0.; sd_awu=0.; sd_awv=0.
    sub_thl=0.; sub_sqv=0.; sub_u=0.; sub_v=0.; det_thl=0.; det_sqv=0.; det_sqc=0.; det_u=0.; det_v=0.
    edmf_a=0.; edmf_w=0.; edmf_qt=0.; edmf_thl=0.; edmf_ent=0.; edmf_qc=0.; tcd=0.; qcd=0.
    s_awchem1=0.; chem1=0.; ktop_plume=0; maxmf=0.; ztop_plume=0.; maxwidth=0.; psig_shcu=1.
    spp_pbl=0; mix_chem=.false.; kpbl=2; uoce=0.; voce=0.; flqc=0.
    call DMP_mf(1,1, kts,kte, delt, zw, dz, p, rho, 1, 0, 1, &
         u,v,w,th,thl,thv,tk, sqw,sqv,sqc,qke, &
         qnc,qni,qnwfa,qnifa,qnbca, exner,vt1,vq1,sgm1, &
         ust,flt,fltv,flq,flqv, pblh,kpbl,dx, xland,th_sfc, &
         edmf_a,edmf_w,edmf_qt,edmf_thl,edmf_ent,edmf_qc, &
         s_aw,s_awthl,s_awqt,s_awqv,s_awqc, s_awu,s_awv,s_awqke, s_awqnc,s_awqni, &
         s_awqnwfa,s_awqnifa,s_awqnbca, sub_thl,sub_sqv,sub_u,sub_v, &
         det_thl,det_sqv,det_sqc,det_u,det_v, nchem,chem1,s_awchem1, mix_chem, &
         qc_bl1,cldfra_bl1, qc_bl1_old,cldfra_bl1_old, &
         .true.,.true., .true.,.true., .true.,.true.,.false., psig_shcu, &
         maxwidth,ktop_plume,maxmf,ztop_plume, spp_pbl,pattern_spp_pbl, tkeprod_up,el1)
    call mynn_tendencies(kts,kte,1, delt,dz,zw,xland,rho, &
         u,v,th,tk,qv,qc,qi,qs,qnc,qni, psfc,p,exner, &
         thl,sqv,sqc,sqi,sqs,sqw, qnwfa,qnifa,qnbca,ozone, &
         ust,flt,flq,flqv,flqc,wspd, uoce,voce, tsq,qsq,cov, tcd,qcd, &
         dfm,dfh,dfq, Du,Dv,Dth,Dqv,Dqc,Dqi,Dqs,Dqnc,Dqni, Dqnwfa,Dqnifa,Dqnbca,Dozone, diss_heat, &
         s_aw,s_awthl, s_awqt,s_awqv,s_awqc, s_awu,s_awv, s_awqnc,s_awqni, s_awqnwfa,s_awqnifa,s_awqnbca, &
         sd_aw,sd_awthl,sd_awqt,sd_awqv, sd_awqc,sd_awqi, sd_awqnc,sd_awqni, sd_awqnwfa,sd_awqnifa, &
         sd_awu,sd_awv, sub_thl,sub_sqv, sub_u,sub_v, det_thl,det_sqv,det_sqc, det_u,det_v, &
         .true.,.true.,.true.,.true., .false., .true.,.true.,.false., .false., &
         cldfra_bl1, 1, 0, 1, 1, 1)
    do k = 1, kx+1
      write(*,'(A,I0,1X,I0,7(1X,ES26.16E3))') 'MF ', c, k, s_aw(k), s_awqc(k), s_awqnc(k), s_awqni(k), &
            s_awqnwfa(k), s_awqnifa(k), s_awqv(k)
    end do
    do k = 1, kx
      write(*,'(A,I0,1X,I0,7(1X,ES26.16E3))') 'TEN ', c, k, dfh(k), sqc(k), sqi(k), Dqnc(k), Dqni(k), &
            Dqnwfa(k), Dqnifa(k)
#ifdef CAPTURE
      write(*,'(A,I0,1X,I0,6(1X,ES26.16E3))') 'CAP ', c, k, cap_sqc2(k), cap_sqi2(k), cap_qnc2(k), &
            cap_qni2(k), cap_qnwfa2(k), cap_qnifa2(k)
#endif
    end do
  end do
contains
  subroutine kh_profile(z, pblh, ust, kh)   ! proofs/mynn_edmf/fortran_oracle/oracle.f90
    real(kind_phys), intent(in) :: z, pblh, ust
    real(kind_phys), intent(out) :: kh
    real(kind_phys) :: zr
    if (z < pblh) then
      zr = z/pblh
      kh = 0.4_kind_phys*max(ust,0.1_kind_phys)*z*(1.0_kind_phys-zr)*(1.0_kind_phys-zr)
    else
      kh = 0.1_kind_phys
    end if
    kh = max(min(kh, 500.0_kind_phys), 0.01_kind_phys)
  end subroutine kh_profile
end program mynn_number_oracle
