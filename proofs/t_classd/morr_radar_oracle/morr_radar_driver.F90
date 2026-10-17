! Column driver for pristine WRF 4.7.1 refl10cm_hm (module_mp_morr_two_moment[_aero]).
! Only caller code is project-authored: the WRF init (MORR_TWO_MOMENT_INIT[_AERO](0), which
! sets xam_*/xbm_*/xmu_* and calls radar_init) followed by refl10cm_hm on each input column
! and the caller's MAX(-35., dBZ) clamp (module_mp_morr_two_moment.F:913-916).
! Input (stream, native REAL): nz, ncol, then per column qv qr nr qs ns qg ng t p (each nz).
! Output: radar_init constants and per-column dBZ (raw) as hex bit patterns + values.
program morr_radar_driver
#ifdef AERO
  use module_mp_morr_two_moment_aero, only: init => MORR_TWO_MOMENT_INIT_AERO, refl10cm_hm
#else
  use module_mp_morr_two_moment, only: init => MORR_TWO_MOMENT_INIT, refl10cm_hm
#endif
  use module_mp_radar
  implicit none
  integer :: nz, ncol, c, k, u
  real, allocatable :: qv(:), qr(:), nr(:), qs(:), ns(:), qg(:), ng(:), t(:), p(:), dbz(:)
  character(len=512) :: path
  call get_command_argument(1, path)
  call init(0)
  open(newunit=u, file=trim(path), access='stream', form='unformatted', status='old')
  read(u) nz, ncol
  allocate(qv(nz), qr(nz), nr(nz), qs(nz), ns(nz), qg(nz), ng(nz), t(nz), p(nz), dbz(nz))
  write(*,'(A,4ES26.17E3)') 'XCRG ', xcrg
  write(*,'(A,4ES26.17E3)') 'XCSG ', xcsg
  write(*,'(A,4ES26.17E3)') 'XCGG ', xcgg
  write(*,'(A,4ES26.17E3)') 'XCRE ', xcre
  write(*,'(A,4ES26.17E3)') 'XCSE ', xcse
  write(*,'(A,4ES26.17E3)') 'XCGE ', xcge
  write(*,'(A,12ES26.17E3)') 'SCAL ', xam_r, xbm_r, xmu_r, xobmr, xam_s, xbm_s, xobms, xocms, &
       xam_g, xbm_g, xobmg, xocmg
  write(*,'(A,3ES26.17E3)') 'XORG ', xorg2, xosg2, xogg2
  write(*,'(A,3ES26.17E3)') 'KPL ', K_w, PI5, lamda4
  write(*,'(A,4ES26.17E3)') 'MWI ', real(m_w_0, 8), aimag(m_w_0), real(m_i_0, 8), aimag(m_i_0)
  do c = 1, ncol
     read(u) qv, qr, nr, qs, ns, qg, ng, t, p
     call refl10cm_hm(qv, qr, nr, qs, ns, qg, ng, t, p, dbz, 1, nz, c, 1)
     do k = 1, nz
        write(*,'(A,I8,I4,Z20,ES26.17E3)') 'DBZ ', c, k, dbz(k), dbz(k)
     end do
  end do
  close(u)
end program morr_radar_driver
