"""WRF caller operands for the PBL / surface-layer oracles (critic cr-w43-conv VERDICT P1, 2026-10-20):
module_first_rk_step_part1.F:637/1380 pass P = grid%p_hyd (HYDROSTATIC pressure, history P_HYD) and PI = pi_phy (dynamic
Exner from P+PB, so t3d = th*pi_phy stays dynamic); RHO = grid%rho = (1+qv)/alt (phy_prep :4856) with alt = al+alb from
calc_p_rho_phi hypsometric_opt=2 (Registry default; namelist unset) module_big_step_utilities_em.F:1045-1048:
alt = (PHt(k+1)-PHt(k)) / phm / LOG(pfd/pfu), pf = c3f*MUTS + c4f + p_top, phm = c3h*MUTS + c4h + p_top."""
import numpy as np


def caller_fields(z):
    g = lambda v: np.asarray(z[v], np.float32)  # noqa: E731
    qv = g("QVAPOR")
    p_dyn = g("P") + g("PB")
    exner = (p_dyn / np.float32(1.0e5)) ** (np.float32(287.0) / np.float32(7.0 * 287.0 / 2.0))
    muts = g("MU") + g("MUB")
    ptop = np.float32(z["P_TOP"])
    c3f, c4f, c3h, c4h = (g(n)[:, None, None] for n in ("C3F", "C4F", "C3H", "C4H"))
    pf = c3f * muts[None] + c4f + ptop
    phm = c3h * muts[None] + c4h + ptop
    pht = g("PH") + g("PHB")
    alt = (pht[1:] - pht[:-1]) / phm / np.log(pf[:-1] / pf[1:])
    rho = (np.float32(1.0) + qv) / alt
    return dict(p=g("P_HYD"), exner=exner, rho=rho.astype(np.float32), p_dyn=p_dyn)
