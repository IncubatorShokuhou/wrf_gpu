"""Genuine mutants for the t-cfix-phyprep caller fixes (E39/E202).

usage: python mutants.py <repo_root> <out.json>
Each mutant restores the pre-fix (or a plausible wrong) caller behaviour in place, runs the selected tests and
counts a KILL only for a genuine assertion failure with zero setup/collection errors; the file is restored
byte-for-byte afterwards (verified by SHA). Run with JAX_PLATFORMS=cpu on lane cores.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MYJ = "src/gpuwrf/physics/myj_adapters.py"
SA = "src/gpuwrf/coupling/scan_adapters.py"
PREP = "src/gpuwrf/coupling/wrf_phy_prep.py"
T = "tests/test_t_cfix_phyprep.py"
MUTANTS = [
    ("M01 MYJ/Janjic theta moist", MYJ, '"theta_cols": _cols(prep["th_phy"])', '"theta_cols": _cols(state.theta)', "myj_pbl_and_janjic"),
    ("M02 MYJ T moist", MYJ, 'T = prep["t_phy"]', 'T = state.theta * prep["pi_phy"]', "myj_pbl_and_janjic or two_step and myj"),
    ("M03 MYJ dry rate on theta_m", MYJ, "theta_new = (state.theta + dt_f * _theta_m_tendency_from_dry(",
     "theta_new = (state.theta + dt_f * (lambda a, b, c, d, e: a)(", "converts_dry_rthblten or two_step and myj"),
    ("M04 MYJ nonhydrostatic PMID", MYJ, 'p, p_int, psfc = prep["p_hyd"], prep["p_hyd_w"], prep["psfc"]',
     'p, p_int, psfc = state.p, prep["p_hyd_w"], prep["psfc"]', "myj_pbl_and_janjic"),
    ("M05 MYJ PSFC lowest level", MYJ, 'p, p_int, psfc = prep["p_hyd"], prep["p_hyd_w"], prep["psfc"]',
     'p, p_int, psfc = prep["p_hyd"], prep["p_hyd_w"], state.p[0]', "myj_pbl_and_janjic"),
    ("M06 GFS-PBL T moist", SA, 'T = prep["t_phy"]', 'T = _temperature_from_theta(state.theta, state.p)', "gfs_legacy or two_step and gfs"),
    ("M07 GFS-PBL dry rate on theta_m", SA, '"theta_m_cols": _cols(state.theta),', '', "gfs_legacy or two_step and gfs"),
    ("M08 cumulus nonhydrostatic P", SA, 'p, p8w, rho = prep["p_hyd"], prep["p_hyd_w"], prep["rho"]',
     'p, p8w, rho = state.p, prep["p_hyd_w"], prep["rho"]', "bmj or ntiedtke or tiedtke or gf_receives"),
    ("M09 cumulus dry rho", SA, 'p, p8w, rho = prep["p_hyd"], prep["p_hyd_w"], prep["rho"]',
     'p, p8w, rho = prep["p_hyd"], prep["p_hyd_w"], _rho_from_state(state)', "bmj or ntiedtke or gf_receives"),
    ("M10 cumulus T moist", SA, 'return dict(T=prep["t_phy"]', 'return dict(T=_temperature_from_theta(state.theta, state.p)',
     "bmj or ntiedtke or gf_receives or two_step and (bmj or gf or ntiedtke)"),
    ("M11 BMJ dry rate", SA, '''        theta=(state.theta + dt_f * _theta_m_tendency_from_dry(
            _back(rth), _back(rqv), state.theta, state.qv, state.theta.dtype
        )).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        rainc_acc''', '''        theta=(state.theta + dt_f * _back(rth)).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        rainc_acc''', "bmj"),
    ("M12 GF dry rate", SA, '''theta=(state.theta + dt_f * _theta_m_tendency_from_dry(
            rth, rqv, state.theta, state.qv, state.theta.dtype)).astype(_output_dtype(state, "theta")),''',
     '''theta=(state.theta + dt_f * rth).astype(_output_dtype(state, "theta")),''', "gf_receives or two_step and gf"),
    ("M13 NTiedtke dry rate", SA, '''        theta=(state.theta + dt_f * _theta_m_tendency_from_dry(
            _back(rth), _back(rqv), state.theta, state.qv, state.theta.dtype
        )).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * _back(rqc)).astype(_output_dtype(state, "qc")),
        qi=''', '''        theta=(state.theta + dt_f * _back(rth)).astype(_output_dtype(state, "theta")),
        qv=(state.qv + dt_f * _back(rqv)).astype(_output_dtype(state, "qv")),
        qc=(state.qc + dt_f * _back(rqc)).astype(_output_dtype(state, "qc")),
        qi=''', "ntiedtke"),
    ("M14 BMJ PSFC lowest mass level", SA, 'else jnp.asarray(prep["psfc"], jnp.float64).reshape(ny * nx))',
     'else jnp.asarray(prep["p"][0], jnp.float64).reshape(ny * nx))', "bmj_receives"),
    ("M15 phy_prep g=9.80665", PREP, "jnp.asarray(WRF_PHYSICS_G_M_S2, dtype)", "jnp.asarray(9.80665, dtype)",
     "helper or gfs_legacy or gf_receives"),
    ("M16 phy_prep moist th", PREP, "th_phy = _dry_theta_view(state)", "th_phy = jnp.asarray(state.theta)", "helper"),
    ("M17 Tiedtke mean-of-levels P8W", SA, 'dz_full, p, p8w = prep["dz"], prep["p"], prep["p8w"]',
     'dz_full, p, p8w = prep["dz"], prep["p"], jnp.concatenate([prep["p"][:1], 0.5 * (prep["p"][:-1] + prep["p"][1:]), prep["p"][-1:]], axis=0)',
     "tiedtke_receives"),
    # ---- series 2 (F15 class B)
    ("M18 G3/GD nonhydrostatic P", SA, "        p8w = jnp.asarray(p8w, real)\n    else:\n        # No metrics (isolated tests): total",
     "        p8w = jnp.asarray(p8w, real); p_hyd = jnp.asarray(state.p, real)\n    else:\n        # No metrics (isolated tests): total",
     "grell_receives"),
    ("M19 QNSE nonhydrostatic PMID", "src/gpuwrf/coupling/qnse_adapter.py", "p_mid=p_hyd,p_int=pint", "p_mid=pressure,p_int=pint", "qnse"),
    ("M20 TEMF nonhydrostatic P3D", "src/gpuwrf/coupling/temf.py", "pressure=cols(p_hyd),", "pressure=cols(p),", "temf"),
    ("M21 EEPS nonhydrostatic P3D", "src/gpuwrf/coupling/epsilon_pbl.py", "pressure=cols(p_hyd),", "pressure=cols(p),", "epsilon"),
    ("M22 KEPS phy_prep-extrapolated P8W", "src/gpuwrf/coupling/epsilon_pbl.py",
     "common.update(p8w=cols(jnp.asarray(p8w,jnp.float32),nz+1),",
     "common.update(p8w=cols(_wrf_phy_prep_p8w(p,z,grid.metrics.fnm,grid.metrics.fnp),nz+1),", "epsilon"),
    # ---- series 3 (F4/F15, API-only LSMs)
    ("M23 SLAB T moist", "src/gpuwrf/coupling/slab_surface_hook.py", 't_bottom = prep["t_phy"][0]',
     't_bottom = jnp.asarray(state.theta)[0] * prep["pi_phy"][0]', "slab"),
    ("M24 Noah TH2=theta_m", "src/gpuwrf/coupling/noahclassic_surface_hook.py",
     "th2 = t_air * (1.0e5 / sfcprs) ** capa / (1.0e5 / psfc) ** capa", "th2 = jnp.asarray(state.theta[0], dtype=jnp.float64)",
     "noahclassic"),
    ("M25 PX THETA1 moist", "src/gpuwrf/coupling/pleim_xiu_surface_hook.py",
     'theta1_2d = jnp.asarray(prep["th_phy"][0], dtype=jnp.float64)', "theta1_2d = jnp.asarray(state.theta[0], dtype=jnp.float64)",
     "pleim"),
    ("M26 PX PSFC lowest level", "src/gpuwrf/coupling/pleim_xiu_surface_hook.py",
     'jnp.asarray(state.p[0] if prep["psfc"] is None else prep["psfc"],', "jnp.asarray(state.p[0],", "pleim"),
]


def main():
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    only = set(sys.argv[3:])
    env = dict(os.environ, JAX_PLATFORMS="cpu", GPUWRF_JAX_CACHE="0")
    results = []
    for name, rel, old, new, select in MUTANTS:
        if only and name.split()[0] not in only:
            continue
        path = root / rel
        original = path.read_bytes()
        text = original.decode()
        count = text.count(old)
        if count != 1:
            results.append(dict(mutant=name, status="NOT_APPLIED", occurrences=count)); continue
        try:
            path.write_text(text.replace(old, new))
            proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "-W", "ignore",
                                   "--basetemp", str(out.parent / "pytest_mut"), str(root / T), "-k", select],
                                  cwd=root, env=env, capture_output=True, text=True, timeout=1800)
        finally:
            path.write_bytes(original)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(original).hexdigest()
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        genuine = proc.returncode == 1 and "AssertionError" in proc.stdout and " error" not in tail
        results.append(dict(mutant=name, file=rel, select=select, rc=proc.returncode, summary=tail,
                            status="KILLED" if genuine else "SURVIVED_OR_ERROR"))
        print(name, results[-1]["status"], tail, flush=True)
    out.write_text(json.dumps(dict(mutants=results, killed=sum(r["status"] == "KILLED" for r in results),
                                   total=len(results)), indent=1) + "\n")


if __name__ == "__main__":
    main()
