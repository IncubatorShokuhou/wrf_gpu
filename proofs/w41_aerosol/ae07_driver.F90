! AE07: pristine WRF V4.7.1 RRTMG SW with aer_opt = 0 / 3 (Thompson water/ice-friendly aerosols) on the RE01 columns.
!
! Unchanged pristine routines (libwrflib.a), the AE01/RE01 arm-A caller wiring (has_reqc/i/s = 1 Thompson radii,
! cldovrlp = 2, icloud = 1, ghg_input = 1, o3input = 2 on the d01 parent column, P = p_hyd, P8W = p_hyd_w, T = t_phy)
! plus the aer_opt = 3 chain of module_radiation_driver.F: gt_aod(p_hyd, dz8w, t_phy, qv, QNWFA, QNIFA, QNBCA,
! wif_input_opt = 1) -> taod5502d = sum_k taod5503d (:1652-1694) -> calc_aerosol_rrtmg_sw(taer_type 1, aod550_opt 2,
! angexp_opt 3, ssa_opt 3, asy_opt 3, namelist default *_val) (:2374-2392) -> RRTMG_SWRAD(aer_opt = 3, tauaer3d_sw,
! ssaaer3d_sw, asyaer3d_sw).  QNWFA/QNIFA columns (number per kg) come from the input.
! Arms (julian, aer_opt, QNWFA/QNIFA scale): julian < 0 -> the column's own radiation-call julian.
! Input = RE01 layout followed by qnwfa(ncol,nz), qnifa(ncol,nz), narm, narm x (julian, aer_opt, scale).
! Output per column and arm: solcon, declin, coszen, julian, taod5503d(nz), taod5502d, tauaer/ssaaer/asyaer(nz,14),
!   hsw(nz), hswc(nz), swu/swuc/swd/swdc(nz+2), scalars as AE01.
program ae07_driver
  use module_mp_thompson, only: thompson_init, calc_effectRad
  use module_ra_rrtmg_sw, only: rrtmg_swinit, rrtmg_swrad
  use module_radiation_driver, only: radconst, calc_coszen, ozn_time_int, ozn_p_int, gt_aod
  use module_ra_aerosol, only: calc_aerosol_rrtmg_sw
  use module_ra_cam_support, only: oznini
  use module_model_constants, only: RE_QC_BG, RE_QI_BG, RE_QS_BG, DEGRAD, DPD, r_d, g
  implicit none

  integer, parameter :: levsiz = 59, n_ozmixm = 12, alevsiz = 12, nsrc = 6
  integer :: ncol, nz, ic, k, arm, narm, hdr(2), aopt
  real :: p_top, gmt_in
  real, allocatable :: julian(:), tsk(:), emiss(:), albedo(:), coszen_h(:), xland(:), xlat(:), xlong(:), snow(:), xice(:), lat01(:)
  integer, allocatable :: julday(:), arm_opt(:)
  real, allocatable :: arm_jul(:), arm_scale(:)
  real, allocatable :: t(:,:), p(:,:), pii(:,:), th(:,:), rho(:,:), qv(:,:), qc(:,:), qi(:,:), ni(:,:), qs(:,:), qr(:,:), qg(:,:)
  real, allocatable :: qcrad(:,:), qirad(:,:), cldfra(:,:), phyd(:,:), p01(:,:), dz8w(:,:), p8w(:,:), t8w(:,:)
  real, allocatable :: tau(:)
  character(len=256) :: fin, fout
  integer :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte
  real, allocatable :: hgt(:,:,:)
  real, allocatable :: t3d(:,:,:), t8w3(:,:,:), p3d(:,:,:), p8w3(:,:,:), pi3d(:,:,:), rho3d(:,:,:), dz3(:,:,:)
  real, allocatable :: cf3(:,:,:), qv3(:,:,:), qc3(:,:,:), qr3(:,:,:), qi3(:,:,:), qs3(:,:,:), qg3(:,:,:), o33(:,:,:)
  real, allocatable :: rec(:,:,:), rei(:,:,:), res(:,:,:), p013(:,:,:)
  real, allocatable :: hsw(:,:,:), hswc(:,:,:), swu(:,:,:), swuc(:,:,:), swd(:,:,:), swdc(:,:,:)
  real, allocatable :: ozmixm(:,:,:,:), ozmixt(:,:,:), pin(:)
  real, allocatable :: aerod(:,:,:,:), qnwfa(:,:), qnifa(:,:), nwfa3(:,:,:), nifa3(:,:,:), nbca3(:,:,:), taod3(:,:,:)
  real, dimension(1,1) :: taod2, angexp2d, aerssa2d, aerasy2d, ht2
  real, dimension(1,1) :: s_xlat, s_xlong, s_xland, s_xice, s_snow, s_tsk, s_alb, s_cosz, s_obscur, hrang
  real, dimension(1,1) :: gsw, swcf
  real, dimension(1,1) :: swupt, swuptc, swuptcln, swdnt, swdntc, swdntcln, swupb, swupbc, swupbcln, swdnb, swdnbc, swdnbcln
  real, dimension(1,1) :: swvisdir, swvisdif, swnirdir, swnirdif, swddir, swddni, swddif, swdownc, swddnic, swddirc
  real, dimension(1,1) :: alvisdir, alvisdif, alnirdir, alnirdif, coszr
  real, pointer :: tauaer(:,:,:,:), ssaaer(:,:,:,:), asyaer(:,:,:,:), tp(:,:,:,:), sp(:,:,:,:), ap(:,:,:,:)
  real :: re_c1(200), re_i1(200), re_s1(200), nc1(200), rec_a(200), rei_a(200), res_a(200)
  real :: declin, solcon, xtime, gmt, radt, jul

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), access='stream', form='unformatted', status='old', convert='native')
  read(10) hdr
  ncol = hdr(1); nz = hdr(2)
  allocate(julian(ncol), julday(ncol), tsk(ncol), emiss(ncol), albedo(ncol), coszen_h(ncol), xland(ncol), xlat(ncol), &
           xlong(ncol), snow(ncol), xice(ncol), lat01(ncol), tau(ncol))
  allocate(t(ncol,nz), p(ncol,nz), pii(ncol,nz), th(ncol,nz), rho(ncol,nz), qv(ncol,nz), qc(ncol,nz), qi(ncol,nz), &
           ni(ncol,nz), qs(ncol,nz), qr(ncol,nz), qg(ncol,nz), qcrad(ncol,nz), qirad(ncol,nz), cldfra(ncol,nz), phyd(ncol,nz), &
           p01(ncol,nz))
  allocate(dz8w(ncol,nz+1), p8w(ncol,nz+1), t8w(ncol,nz+1))
  read(10) p_top, gmt_in
  read(10) julian, julday, tsk, emiss, albedo, coszen_h, xland, xlat, xlong, snow, xice, lat01
  read(10) t, p, pii, th, rho, qv, qc, qi, ni, qs, qr, qg, qcrad, qirad, cldfra, phyd, p01
  read(10) dz8w, p8w, t8w
  read(10) tau
  allocate(qnwfa(ncol,nz), qnifa(ncol,nz))
  read(10) qnwfa, qnifa
  read(10) narm
  allocate(arm_jul(narm), arm_opt(narm), arm_scale(narm))
  read(10) arm_jul, arm_opt, arm_scale
  close(10)

  ids = 1; ide = 2; jds = 1; jde = 2; kds = 1; kde = nz + 1
  ims = 1; ime = 1; jms = 1; jme = 1; kms = 1; kme = nz + 1
  its = 1; ite = 1; jts = 1; jte = 1; kts = 1; kte = nz
  allocate(hgt(1,kms:kme,1)); hgt = 0.
  allocate(t3d(1,kms:kme,1), t8w3(1,kms:kme,1), p3d(1,kms:kme,1), p8w3(1,kms:kme,1), pi3d(1,kms:kme,1), rho3d(1,kms:kme,1), &
           dz3(1,kms:kme,1), cf3(1,kms:kme,1), qv3(1,kms:kme,1), qc3(1,kms:kme,1), qr3(1,kms:kme,1), qi3(1,kms:kme,1), &
           qs3(1,kms:kme,1), qg3(1,kms:kme,1), o33(1,kms:kme,1), rec(1,kms:kme,1), rei(1,kms:kme,1), res(1,kms:kme,1), &
           p013(1,kms:kme,1), hsw(1,kms:kme,1), hswc(1,kms:kme,1))
  allocate(swu(1,kms:kme+2,1), swuc(1,kms:kme+2,1), swd(1,kms:kme+2,1), swdc(1,kms:kme+2,1))
  allocate(ozmixm(1,levsiz,1,n_ozmixm), ozmixt(1,levsiz,1), pin(levsiz))
  allocate(aerod(1,kms:kme,1,nsrc), nwfa3(1,kms:kme,1), nifa3(1,kms:kme,1), nbca3(1,kms:kme,1), taod3(1,kms:kme,1))
  allocate(tauaer(1,kms:kme,1,14), ssaaer(1,kms:kme,1,14), asyaer(1,kms:kme,1,14))
  aerod = 0.; ht2 = 0.

  call thompson_init(HGT=hgt, IDS=ids, IDE=ide, JDS=jds, JDE=jde, KDS=kds, KDE=kde, IMS=ims, IME=ime, JMS=jms, JME=jme, &
       KMS=kms, KME=kme, ITS=its, ITE=ite, JTS=jts, JTE=jte, KTS=kts, KTE=kte)
  call rrtmg_swinit(.true., ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)

  open(20, file=trim(fout), access='stream', form='unformatted', status='replace', convert='native')
  write(20) ncol, nz, narm
  gmt = gmt_in; radt = 30.0
  do ic = 1, ncol
    do k = 1, nz
      re_c1(k) = RE_QC_BG; re_i1(k) = RE_QI_BG; re_s1(k) = RE_QS_BG
      nc1(k) = 100.E6 / rho(ic,k)
    end do
    call calc_effectRad(t(ic,1:nz), p(ic,1:nz), qv(ic,1:nz), qc(ic,1:nz), nc1(1:nz), qi(ic,1:nz), ni(ic,1:nz), qs(ic,1:nz), &
                        re_c1(1:nz), re_i1(1:nz), re_s1(1:nz), 1, nz)
    do k = 1, nz
      rec_a(k) = MAX(RE_QC_BG, MIN(re_c1(k), 50.E-6))
      rei_a(k) = MAX(RE_QI_BG, MIN(re_i1(k), 125.E-6))
      res_a(k) = MAX(RE_QS_BG, MIN(re_s1(k), 999.E-6))
    end do
    do arm = 1, narm
      jul = julian(ic)
      if (arm_jul(arm) >= 0.) jul = arm_jul(arm)
      aopt = arm_opt(arm)
      xtime = tau(ic) * 60.0
      call radconst(xtime, declin, solcon, jul, DEGRAD, DPD)
      s_xlat(1,1) = xlat(ic); s_xlong(1,1) = xlong(ic)
      call calc_coszen(ims, ime, jms, jme, its, ite, jts, jte, jul, xtime + radt*0.5, gmt, declin, DEGRAD, &
                       s_xlong, s_xlat, s_cosz, hrang)
      s_xlat(1,1) = lat01(ic)
      call oznini(ozmixm, pin, levsiz, n_ozmixm, s_xlat, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                  its, ite, jts, jte, kts, kte)
      call ozn_time_int(julday(ic), jul, ozmixm, ozmixt, levsiz, n_ozmixm, ids, ide, jds, jde, kds, kde, &
                        ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
      p013 = 0.; p013(1,1:nz,1) = p01(ic,1:nz)
      o33 = 0.
      call ozn_p_int(p013, pin, levsiz, ozmixt, o33, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                     its, ite, jts, jte, kts, kte)
      s_xlat(1,1) = xlat(ic)
      t3d = 0.; p3d = 0.; pi3d = 0.; rho3d = 0.; dz3 = 0.; t8w3 = 0.; p8w3 = 0.
      cf3 = 0.; qv3 = 0.; qc3 = 0.; qr3 = 0.; qi3 = 0.; qs3 = 0.; qg3 = 0.
      t3d(1,1:nz,1) = t(ic,1:nz); p3d(1,1:nz,1) = phyd(ic,1:nz); pi3d(1,1:nz,1) = pii(ic,1:nz); rho3d(1,1:nz,1) = rho(ic,1:nz)
      dz3(1,1:nz+1,1) = dz8w(ic,1:nz+1); t8w3(1,1:nz+1,1) = t8w(ic,1:nz+1); p8w3(1,1:nz+1,1) = p8w(ic,1:nz+1)
      cf3(1,1:nz,1) = cldfra(ic,1:nz); qv3(1,1:nz,1) = qv(ic,1:nz); qc3(1,1:nz,1) = qcrad(ic,1:nz)
      qr3(1,1:nz,1) = qr(ic,1:nz); qi3(1,1:nz,1) = qirad(ic,1:nz); qs3(1,1:nz,1) = qs(ic,1:nz); qg3(1,1:nz,1) = qg(ic,1:nz)
      rec = 0.; rei = 0.; res = 0.
      rec(1,1:nz,1) = rec_a(1:nz); rei(1,1:nz,1) = rei_a(1:nz); res(1,1:nz,1) = res_a(1:nz)
      ! radiation_driver Radiation_step, aer_opt = 3: gt_aod + calc_aerosol_rrtmg_sw (taer_* driver PARAMETERs)
      tauaer = 0.; ssaaer = 0.; asyaer = 0.; taod3 = 0.; taod2 = 0.
      angexp2d = 0.; aerssa2d = 0.; aerasy2d = 0.
      if (aopt == 3) then
        nwfa3 = 0.; nifa3 = 0.; nbca3 = 0.
        nwfa3(1,1:nz,1) = qnwfa(ic,1:nz) * arm_scale(arm); nifa3(1,1:nz,1) = qnifa(ic,1:nz) * arm_scale(arm)
        call gt_aod(p3d, dz3, t3d, qv3, nwfa3, nifa3, nbca3, taod3, 1, ims, ime, jms, jme, kms, kme, &
                    its, ite, jts, jte, kts, kte)
        do k = kts, kte
          taod2(1,1) = taod2(1,1) + taod3(1,k,1)
        end do
        call calc_aerosol_rrtmg_sw(ht2, dz3, p3d, t3d, qv3, 1, 2, 3, 3, 3, 0.12, 1.3, 0.85, 0.9, &
                                   taod2, angexp2d, aerssa2d, aerasy2d, ims, ime, jms, jme, kms, kme, &
                                   its, ite, jts, jte, kts, kte, tauaer, ssaaer, asyaer, taod3)
        tp => tauaer; sp => ssaaer; ap => asyaer   ! radiation_driver allocates them only for aer_opt 2/3
      else
        nullify(tp, sp, ap)
      end if
      s_xland(1,1) = xland(ic); s_xice(1,1) = xice(ic); s_snow(1,1) = snow(ic); s_tsk(1,1) = tsk(ic)
      s_alb(1,1) = albedo(ic); s_obscur(1,1) = 0.
      alvisdir = albedo(ic); alvisdif = albedo(ic); alnirdir = albedo(ic); alnirdif = albedo(ic)
      coszr = 0.
      hsw = 0.; hswc = 0.; swu = 0.; swuc = 0.; swd = 0.; swdc = 0.
      gsw = 0.; swcf = 0.
      swupt = 0.; swuptc = 0.; swuptcln = 0.; swdnt = 0.; swdntc = 0.; swdntcln = 0.
      swupb = 0.; swupbc = 0.; swupbcln = 0.; swdnb = 0.; swdnbc = 0.; swdnbcln = 0.
      swvisdir = 0.; swvisdif = 0.; swnirdir = 0.; swnirdif = 0.
      swddir = 0.; swddni = 0.; swddif = 0.; swdownc = 0.; swddnic = 0.; swddirc = 0.
      call rrtmg_swrad( &
           rthratensw=hsw, rthratenswc=hswc, &
           swupt=swupt, swuptc=swuptc, swuptcln=swuptcln, swdnt=swdnt, swdntc=swdntc, swdntcln=swdntcln, &
           swupb=swupb, swupbc=swupbc, swupbcln=swupbcln, swdnb=swdnb, swdnbc=swdnbc, swdnbcln=swdnbcln, &
           swcf=swcf, gsw=gsw, xtime=xtime, gmt=gmt, xlat=s_xlat, xlong=s_xlong, radt=radt, degrad=DEGRAD, declin=declin, &
           coszr=coszr, julday=julday(ic), solcon=solcon, albedo=s_alb, t3d=t3d, t8w=t8w3, tsk=s_tsk, p3d=p3d, p8w=p8w3, &
           pi3d=pi3d, rho3d=rho3d, dz8w=dz3, cldfra3d=cf3, lradius=rec, iradius=rei, is_cammgmp_used=.false., r=r_d, g=g, &
           re_cloud=rec, re_ice=rei, re_snow=res, has_reqc=1, has_reqi=1, has_reqs=1, &
           icloud=1, warm_rain=.false., cldovrlp=2, idcor=0, f_ice_phy=qi3, f_rain_phy=qr3, &
           xland=s_xland, xice=s_xice, snow=s_snow, qv3d=qv3, qc3d=qc3, qr3d=qr3, qi3d=qi3, qs3d=qs3, qg3d=qg3, &
           o3input=2, o33d=o33, aer_opt=aopt, aerod=aerod, no_src=nsrc, &
           alswvisdir=alvisdir, alswvisdif=alvisdif, alswnirdir=alnirdir, alswnirdif=alnirdif, &
           swvisdir=swvisdir, swvisdif=swvisdif, swnirdir=swnirdir, swnirdif=swnirdif, sf_surface_physics=4, &
           f_qv=.true., f_qc=.true., f_qr=.true., f_qi=.true., f_qs=.true., f_qg=.true., aer_ra_feedback=0, progn=0, &
           calc_clean_atm_diag=0, mp_physics=8, ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
           ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, &
           swupflx=swu, swupflxc=swuc, swdnflx=swd, swdnflxc=swdc, tauaer3d_sw=tp, ssaaer3d_sw=sp, asyaer3d_sw=ap, &
           swddir=swddir, swddni=swddni, swddif=swddif, swdownc=swdownc, swddnic=swddnic, swddirc=swddirc, &
           xcoszen=s_cosz, yr=2026, julian=jul, ghg_input=1, obscur=s_obscur, proceed_cmaq_sw=.false.)
      write(20) solcon, declin, s_cosz(1,1), jul, taod3(1,1:nz,1), taod2(1,1)
      write(20) tauaer(1,1:nz,1,:), ssaaer(1,1:nz,1,:), asyaer(1,1:nz,1,:)
      write(20) hsw(1,1:nz,1), hswc(1,1:nz,1)
      write(20) swu(1,1:nz+2,1), swuc(1,1:nz+2,1), swd(1,1:nz+2,1), swdc(1,1:nz+2,1)
      write(20) gsw, swcf, swupt, swuptc, swdnt, swdntc, swupb, swupbc, swdnb, swdnbc, swddir, swddni, swddif, swdownc, coszr
    end do
  end do
  close(20)
end program ae07_driver
