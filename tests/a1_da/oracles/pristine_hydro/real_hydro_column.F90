! Pristine-WRF oracle (lane a1-io, DESIGN_v2 V3b): the real.exe final hydrostatic block, column form.
! Verbatim transcription of dyn_em/module_initialize_real.F (WRF 4.7.1, <USER_HOME>/src/wrf_pristine/WRF)
!   :3923-3958 top-down perturbation pressure / alt / al (moist loading summed over ALL moist species),
!   :3974-3987 hypsometric_opt == 2 geopotential, :4002-4011 al re-diagnosis (hypsometric_opt == 2),
!   :4031-4036 p from the equation of state;
! one pass (flag_metgrid adjust_heights off -> dpmu = 0 after the first pass). grid%x(i,k,j) -> x(k);
! constants = share/module_model_constants.F. Default REAL = REAL*4 like WRF (RWORDSIZE 4).
! I/O: stream binary. Input: nz, ncol, nmoist (int32); c1f c2f c3f c4f (nz+1) c1h c2h c3h c4h rdnw rdn dnw (nz)
! p_top (real32); per column: mub, mu_2, pb(nz), alb(nz), t_2(nz) [theta - t0], moist(nz,nmoist), phb(nz+1).
! Output per column: p(nz), ph_2(nz+1), al(nz), alt(nz).
program real_hydro_column
  implicit none
  real, parameter :: r_d = 287., cp = 7.*r_d/2., r_v = 461.6, cv = cp-r_d, cvpm = -cv/cp
  real, parameter :: p1000mb = 100000., t0 = 300., cpovcv = cp/(cp-r_d), rvovrd = r_v/r_d
  integer :: nz, ncol, nmoist, kte, k, kk, im, col, u
  real, allocatable :: c1f(:), c2f(:), c3f(:), c4f(:), c1h(:), c2h(:), c3h(:), c4h(:), rdnw(:), rdn(:), dnw(:)
  real, allocatable :: pb(:), alb(:), t_2(:), moist(:,:), phb(:), p(:), ph_2(:), al(:), alt(:), p_hyd(:)
  real :: p_top, mub, mu_2, mu0, qtot, qvf, qvf1, qvf2, pfu, pfd, phm
  character(len=512) :: fin, fout
  integer, parameter :: p_qv = 1

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(newunit=u, file=trim(fin), access='stream', form='unformatted', status='old')
  read(u) nz, ncol, nmoist
  kte = nz + 1
  allocate(c1f(kte), c2f(kte), c3f(kte), c4f(kte), c1h(nz), c2h(nz), c3h(nz), c4h(nz), rdnw(nz), rdn(nz), dnw(nz))
  allocate(pb(nz), alb(nz), t_2(nz), moist(nz, nmoist), phb(kte), p(nz), ph_2(kte), al(nz), alt(nz), p_hyd(nz))
  read(u) c1f, c2f, c3f, c4f, c1h, c2h, c3h, c4h, rdnw, rdn, dnw, p_top
  open(unit=20, file=trim(fout), access='stream', form='unformatted', status='replace')
  do col = 1, ncol
    read(u) mub, mu_2, pb, alb, t_2, moist, phb
    mu0 = mub + mu_2
    ! ---- :3923-3940 top level
    kk = kte-1
    k = kk+1
    qtot = 0.
    do im = 1, nmoist
      qtot = qtot + moist(kk,im)
    enddo
    qvf2 = 1./(1.+qtot)
    qvf1 = qtot*qvf2
    p(kk) = - 0.5*((c1f(k)*mu_2)+qvf1*(c1f(k)*mub+c2f(k)))/rdnw(kk)/qvf2
    qvf = 1. + rvovrd*moist(kk,p_qv)
    alt(kk) = (r_d/p1000mb)*(t_2(kk)+t0)*qvf*(((p(kk)+pb(kk))/p1000mb)**cvpm)
    al(kk) = alt(kk) - alb(kk)
    p_hyd(kk) = p(kk) + pb(kk)
    ! ---- :3945-3958 downward
    do kk = kte-2, 1, -1
      k = kk + 1
      qtot = 0.
      do im = 1, nmoist
        qtot = qtot + 0.5*(moist(kk,im)+moist(kk+1,im))
      enddo
      qvf2 = 1./(1.+qtot)
      qvf1 = qtot*qvf2
      p(kk) = p(kk+1) - ((c1f(k)*mu_2) + qvf1*(c1f(k)*mub+c2f(k)))/qvf2/rdn(kk+1)
      qvf = 1. + rvovrd*moist(kk,p_qv)
      alt(kk) = (r_d/p1000mb)*(t_2(kk)+t0)*qvf*(((p(kk)+pb(kk))/p1000mb)**cvpm)
      al(kk) = alt(kk) - alb(kk)
      p_hyd(kk) = p(kk) + pb(kk)
    end do
    ! ---- :3974-3987 hypsometric_opt == 2
    ph_2(1) = phb(1)
    do k = 2, kte
      pfu = c3f(k  )*mu0+c4f(k  ) + p_top
      pfd = c3f(k-1)*mu0+c4f(k-1) + p_top
      phm = c3h(k-1)*mu0+c4h(k-1) + p_top
      ph_2(k) = ph_2(k-1) + alt(k-1)*phm*log(pfd/pfu)
    end do
    do k = 1, kte
      ph_2(k) = ph_2(k) - phb(k)
    end do
    ! ---- :4002-4011 al re-diagnosis (hypsometric_opt == 2)
    do k = 1, kte-1
      pfu = c3f(k+1)*(mub+mu_2)+c4f(k+1)+p_top
      pfd = c3f(k  )*(mub+mu_2)+c4f(k  )+p_top
      phm = c3h(k  )*(mub+mu_2)+c4h(k  )+p_top
      al(k) = (ph_2(k+1)-ph_2(k)+phb(k+1)-phb(k))/phm/log(pfd/pfu)-alb(k)
    end do
    ! ---- :4031-4036 p from the equation of state
    do k = 1, kte-1
      qvf = 1.+rvovrd*moist(k,p_qv)
      p(k) = p1000mb*( (r_d*(t0+t_2(k))*qvf)/(p1000mb*(al(k)+alb(k))) )**cpovcv - pb(k)
      p_hyd(k) = p(k) + pb(k)
      alt(k) = al(k) + alb(k)
    end do
    write(20) p, ph_2, al, alt
  end do
  close(20)
  close(u)
end program real_hydro_column
