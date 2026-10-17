"""Genuine-mutant check for the Shin-Hong WRF-oracle gates (lane t-shinhong, E39/E202).

Each mutant reverts ONE fix in place, runs the gate tests, requires a genuine
assertion failure (junit <failure>, no <error>), then restores the file bytes.
Run sequentially in one worktree only (source-mutating, E125).
usage: JAX_PLATFORMS=cpu python mutants.py <repo> <out_json> [mutant-id ...]
"""
from pathlib import Path
import json, subprocess, sys, xml.etree.ElementTree as ET

repo, out_json = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
K = 'src/gpuwrf/physics/bl_shinhong.py'
A = 'src/gpuwrf/coupling/scan_adapters.py'
ORACLE = 'tests/v040/test_shinhong_wrf_oracle.py'
CALLER = 'tests/v040/test_shinhong_tke_caller.py'
COMPILED = ORACLE + '::test_compiled_shinhong_inside_wrf_input_ulp_envelope'
MUTANTS = [
    ('M1 pblflg gate dropped', K, '    in_pbl_face = jnp.logical_and(pblflg, kp1_face < kpbl)\n',
     '    in_pbl_face = kp1_face < kpbl\n', [ORACLE]),
    ('M2 mixlen cwm zero', K, '        u, v, tx, th, qv, qc, q2x0, zq,', '        u, v, tx, th, qv, jnp.zeros_like(qc), q2x0, zq,', [ORACLE]),
    ('M3 tflux without PI', K, '    tflux_e = _seq_cumsum(ttend * dz, reverse=True)\n',
     '    tflux_e = _seq_cumsum(theta_tend * dz, reverse=True)\n', [ORACLE]),
    ('M4 mixlen k+2 interface', K, '    rdz = 2.0 / jnp.maximum(zkp1 - zkm1, 1.0e-12)\n',
     '    rdz = 2.0 / jnp.maximum(jnp.concatenate([z[2:], z[-1:]]) - zkm1, 1.0e-12)\n', [ORACLE]),
    ('M5 RQCBLTEN dropped', K, '    qc_tend = (q3_sol[1] - qc) * rdt\n', '    qc_tend = jnp.zeros_like(qc)\n', [ORACLE]),
    ('M6 tridiag division form', K, '        f_k = fk * (rhs[..., k] - lower[k] * f_prev)\n',
     '        f_k = (rhs[..., k] - lower[k] * f_prev) / (diag[k] - lower[k] * au_prev)\n', [ORACLE]),
    ('M7 EP1 python double', K, 'EP1 = float(np.float32(R_V) / np.float32(R_D) - np.float32(1.0))\n',
     'EP1 = R_V / R_D - 1.0\n', [ORACLE]),
    ('M8 caller corf default', A, '    corf = 1.0e-4 if grid is None else grid.metrics.f.reshape(-1)\n',
     '    corf = 1.0e-4\n', [CALLER]),
    ('M9 caller qc not passed', A, '            qc=real(cols(state.qc)),\n', '            qc=real(cols(state.qc) * 0.0),\n', [CALLER]),
    # Review rv-t-shinhong: erased compiled driving outputs (TKE kept) must fail the
    # COMPILED gate on its own (the former AND mask let it pass).
    ('M10 erase driving outputs vs compiled gate', K,
     '    u_t, v_t, th_t, qv_t, exch_h, pblh, kpbl, wstar, delta, tke_out, el_pbl, qc_t, qi_t = out\n',
     '    u_t, v_t, th_t, qv_t, exch_h, pblh, kpbl, wstar, delta, tke_out, el_pbl, qc_t, qi_t = out\n'
     '    u_t, v_t, th_t, qv_t, exch_h, el_pbl, qc_t, qi_t = (x * 0.0 for x in (u_t, v_t, th_t, qv_t, exch_h, el_pbl, qc_t, qi_t))\n',
     [COMPILED]),
    ('M11 pblflg gate dropped vs compiled gate', K, '    in_pbl_face = jnp.logical_and(pblflg, kp1_face < kpbl)\n',
     '    in_pbl_face = kp1_face < kpbl\n', [COMPILED]),
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
        xml = out_json.parent / f'mutant_{name.split()[0]}.xml'
        proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', '-W', 'ignore',
                               '--basetemp', str(out_json.parent / 'pytest' / name.split()[0]), f'--junitxml={xml}', *tests],
                              cwd=repo, capture_output=True, text=True, timeout=900)
        root = ET.parse(xml).getroot()
        fails = [f for f in root.iter('failure')]
        errors = [e for e in root.iter('error')]
        msgs = [f.get('message', '')[:160] for f in fails]
        killed = proc.returncode != 0 and len(fails) > 0 and not errors and any('Assert' in (f.get('type') or '') + (f.get('message') or '') for f in fails)
        results.append(dict(mutant=name, killed=killed, failures=len(fails), errors=len(errors), messages=msgs))
    finally:
        path.write_bytes(original)
    assert path.read_bytes() == original
    print(results[-1], flush=True)
status = subprocess.run(['git', 'status', '--porcelain', '--', K, A], cwd=repo, capture_output=True, text=True).stdout
out_json.write_text(json.dumps(dict(results=results, all_killed=all(r['killed'] for r in results),
                                    tree_clean_after=status == ''), indent=1) + '\n')
print('ALL KILLED' if all(r['killed'] for r in results) else 'SURVIVOR', 'tree clean' if status == '' else status)
