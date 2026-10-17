"""Genuine-mutant check for the YSU WRF-oracle gates (lane t-ysuqc, E39/E202).

Each mutant reverts ONE fix in place, runs the gate tests, requires a genuine
assertion failure (junit <failure>, no <error>), then restores the file bytes.
Run sequentially in one worktree only (source-mutating, E125).
usage: JAX_PLATFORMS=cpu python mutants.py <repo> <out_json> [mutant-id ...]
"""
from pathlib import Path
import json, subprocess, sys, xml.etree.ElementTree as ET

repo, out_json = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
K = 'src/gpuwrf/physics/pbl_ysu.py'
A = 'src/gpuwrf/coupling/scan_adapters.py'
O = 'src/gpuwrf/runtime/operational_mode.py'
T = 'tests/v040/test_ysu_wrf_oracle.py'
LIT = T + '::test_literal_real4_ysu_is_byte_identical_to_pristine_wrf'
CMP = T + '::test_compiled_ysu_inside_wrf_input_ulp_envelope'
ADP = T + '::test_adapter_passes_wrf_caller_operands_and_applies_cloud_rates'
OPS = T + '::test_operational_ysu_call_passes_rthraten_and_namelist_topdown'
PBLH = T + '::test_get_pblh_vectorised_matches_literal_wrf_loops'
RAD = T + '::test_legacy_radiation_refresh_reaches_ysu_once_from_entry_state'
SSIB = T + '::test_ssib_ysu_rates_enter_rk_once_and_restore_entry_state'
S = 'src/gpuwrf/coupling/ssib_surface_hook.py'
MUTANTS = [
    ('M1 topdown branch off', K, '    if topdown:  # theta-li enhancement', '    if False:  # theta-li enhancement', [CMP]),
    ('M2 cloud-top entrainment off', K, '        cloudflg = jnp.logical_and(jnp.logical_and(pblflg, kpbl >= 2), (qcxl[ke] + qixl[ke]) > 0.01e-3)\n',
     '        cloudflg = jnp.logical_and(jnp.logical_and(pblflg, kpbl >= 2), (qcxl[ke] + qixl[ke]) > 1.e30)\n', [CMP]),
    ('M3 RQCBLTEN dropped', K, '    qc_tend = (qc_sol - qcxl) * rdt if f_qc else zeros\n', '    qc_tend = zeros\n', [CMP]),
    ('M4 in-cloud Ri dropped', K, '    ri = jnp.where(jnp.logical_and(cloud[:-1], cloud[1:]), ri_cloud, ri)\n', '', [CMP]),
    ('M5 thlix without cloud water', K, '    thlix = (tx - c.XLV * qcxl / CP - 2.834e6 * qixl / CP) / pi\n',
     '    thlix = (tx - 2.834e6 * qixl / CP) / pi\n', [CMP]),
    ('M6 RTHBLTEN not via TTEND/PI', K, '    theta_tend = ttend / pi\n', '    theta_tend = (theta_sol - thx + 300.0) * rdt\n', [LIT]),
    ('M7 ctopo/vconv diagonal as 1+fric', K, '    first_ad = 1.0 + fric * vconvlim + ctopo * fric * (1.0 - vconvlim)\n',
     '    first_ad = 1.0 + fric\n', [LIT]),
    ('M8 get_pblh safeguard overwrite dropped', K,
     '    zi = jnp.where(jnp.logical_and(jnp.any(found), jz < n - 3), zi_j[jz], zw[1])\n',
     '    zi = jnp.where(jnp.any(found), zi_j[jz], zw[1])\n', [PBLH]),
    ('M9 tridiag division form', K, '        f_k = fk * (rhs[..., k] - lower[k] * f_prev)\n',
     '        f_k = (rhs[..., k] - lower[k] * f_prev) / (diag[k] - lower[k] * au_prev)\n', [LIT]),
    ('M10 xkzm(kpbl-1)=0 cloud rule dropped', K,
     '    xkzm_p = jnp.where(jnp.logical_and(jnp.logical_and(kp1 == kpbl - 1, cloudflg), we < 0.0), 0.0, xkzm_p)\n',
     '', [CMP]),
    ('M11 adapter qc not passed', A, '            qc=real(cols(forcing_state.qc)), qi=real(cols(forcing_state.qi)),\n',
     '            qc=real(cols(forcing_state.qc) * 0.0), qi=real(cols(forcing_state.qi)),\n', [ADP]),
    ('M12 adapter cloud rates not applied', A,
     '    result = _apply_ysu_cloud_rates(result, state, dt, out, ny=f["ny"], nx=f["nx"], nz=f["nz"])\n', '', [ADP]),
    ('M13 operational rthraten not passed', O,
     "                **({'rthraten':next_carry.rthraten,'ysu_topdown_pblmix':int(namelist.ysu_topdown_pblmix)}\n                   if bl_opt==1 else {}),\n                # Original pbl_driver.F:1593",
     "                **({'ysu_topdown_pblmix':int(namelist.ysu_topdown_pblmix)}\n                   if bl_opt==1 else {}),\n                # Original pbl_driver.F:1593", [OPS]),
    ('M14 EP1 python double', K, '            ROVCP=rd / cp, EP1=rv / rd - F(1.0), EP2=rd / rv,\n',
     '            ROVCP=rd / cp, EP1=float(461.6 / 287.0 - 1.0), EP2=rd / rv,\n', [LIT]),
    # rv-t-ysuqc FIX round: legacy radiation order and SSiB once-RK seam.
    ('M15 YSU early radiation refresh removed', O,
     '    if int(namelist.bl_pbl_physics) == 1 and next_carry.radiation_diagnostics is None:\n',
     '    if False and next_carry.radiation_diagnostics is None:\n', [RAD]),
    ('M16 YSU radiation input is post-surface state', O,
     '        early_rthraten, next_carry = _legacy_radiation_refresh(before, next_carry)\n',
     '        early_rthraten, next_carry = _legacy_radiation_refresh(next_state, next_carry)\n', [RAD]),
    ('M17 radiation solved twice', O, '    if early_rthraten is not None:\n        # YSU already consumed',
     '    if False:\n        # YSU already consumed', [RAD]),
    ('M18 SSiB raw RK seam removed', O,
     '        if source_leaf_mode and _phys_tend_rk_wrf_enabled():\n            # Same once-RK seam as the generic SF1 branch',
     '        if False:\n            # Same once-RK seam as the generic SF1 branch', [SSIB]),
    ('M19 SSiB publishes theta_m instead of dry RTHBLTEN', S,
     '    return (updated,raw) if _with_sources else updated\n',
     '    return (updated,result) if _with_sources else updated\n', [SSIB]),
]
wanted = set(sys.argv[3:])
if wanted:
    MUTANTS = [m for m in MUTANTS if m[0].split()[0] in wanted]
results = []
for name, rel, old, new, tests in MUTANTS:
    path = repo / rel
    original = path.read_bytes()
    text = original.decode()
    assert text.count(old) == 1, (name, text.count(old))
    path.write_text(text.replace(old, new))
    try:
        tag = name.split()[0]
        xml = out_json.parent / f'mutant_{tag}.xml'
        proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '-W', 'ignore',
                               '--basetemp', str(out_json.parent / 'pytest' / tag), f'--junitxml={xml}', *tests],
                              cwd=repo, capture_output=True, text=True, timeout=1500)
        root = ET.parse(xml).getroot()
        fails = list(root.iter('failure'))
        errors = list(root.iter('error'))
        msgs = [f.get('message', '')[:160] for f in fails]
        killed = proc.returncode != 0 and len(fails) > 0 and not errors and any(
            'Assert' in (f.get('type') or '') + (f.get('message') or '') for f in fails)
        results.append(dict(mutant=name, killed=killed, failures=len(fails), errors=len(errors), messages=msgs))
    finally:
        path.write_bytes(original)
    assert path.read_bytes() == original
    print(results[-1], flush=True)
status = subprocess.run(['git', 'status', '--porcelain', '--', K, A, O, S], cwd=repo, capture_output=True, text=True).stdout
out_json.write_text(json.dumps(dict(results=results, all_killed=all(r['killed'] for r in results)), indent=1) + '\n')
print('ALL KILLED' if all(r['killed'] for r in results) else 'SURVIVOR')
