"""Genuine-mutant harness (E202) for lane w41-shinhongtke.

Each mutant rewrites one source file in place, runs its test set with a junit report, restores
the original bytes in `finally` (SHA re-checked), and counts a kill only for failures that are
assertion failures (AssertionError / numpy 'Arrays are not equal' / 'DID NOT RAISE') with zero
setup/collection errors. Groups: M = history writer/adapter, S = cold seeds, DX = scale-aware
clamps vs the dx fixture (and, for disclosure, vs the 3 km v040 oracle test).
usage: python mutants.py <repo> <out.json> <basetemp>
"""
import hashlib, json, os, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path

REPO, OUT, BASE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
W = 'src/gpuwrf/io/wrfout_writer.py'
A = 'src/gpuwrf/coupling/scan_adapters.py'
N = 'src/gpuwrf/integration/nested_pipeline.py'
O = 'src/gpuwrf/runtime/operational_mode.py'
K = 'src/gpuwrf/physics/bl_shinhong.py'
HIST = ['tests/v041/test_shinhong_history.py', 'tests/v041/test_pbl_tke_history.py']
DXTEST = ['tests/v041/test_shinhong_dx_oracle.py']
V040 = ['tests/v040/test_shinhong_wrf_oracle.py']
CALL = "            _pbl_tke_history(fields,state,(nz,ny,nx),int(_lookup(namelist,'bl_pbl_physics')))\n"
MUTANTS = {
    'M1_writer_hunk_deleted': (W, CALL, "            pass\n", HIST),
    'M2_top_not_bottom_zero': (W, "        source[:-1]=np.float32(_QKE_TO_TKE_PBL[bl])", "        source[1:]=np.float32(_QKE_TO_TKE_PBL[bl])", HIST),
    'M3_every_pbl_publishes_qke': (W, "_QKE_TO_TKE_PBL = {2: 0.5, 4: 0.5, 8: 1.0, 11: 1.0, 12: 1.0}",
                                   "_QKE_TO_TKE_PBL = {1: 1.0, 2: 0.5, 4: 0.5, 5: 1.0, 8: 1.0, 9: 1.0, 11: 1.0, 12: 1.0}", HIST),
    'M4_cam_profile_ignored': (W, '    source=_lookup(state,"cam_tke_pbl",None) if bl == 11 else None\n', '    source=None\n', HIST),
    'M5_nonfinite_accepted': (W, '    if not np.isfinite(value).all(): raise ValueError(f"bl={bl} TKE_PBL history nonfinite")\n', '', HIST),
    'M6_el_never_published': (A, "              if int(shinhong_tke_diag)==1 else {})", "              if False else {})", HIST),
    'M7_el_published_shifted': (A, "{'el_pbl': _back3d(out[\"el_pbl\"])", "{'el_pbl': jnp.roll(_back3d(out[\"el_pbl\"]),1,axis=0)", HIST),
    'M8_q2_convention_ignored': (W, "_QKE_TO_TKE_PBL = {2: 0.5, 4: 0.5,", "_QKE_TO_TKE_PBL = {2: 1.0, 4: 1.0,", HIST),
    'M9_camuw_carry_not_passed': (N, "diagnostics=dict(diagnostics or {},TKE_PBL=carry.camuw_pbl.tke_pbl)", "diagnostics=diagnostics", HIST),
    'S1_myj_seed_without_q2_factor': (O, "(2 if int(namelist.bl_pbl_physics) == 2 else 1) * tke", "tke", HIST),
    'S2_boulac_seed_dropped': (O, "    elif int(namelist.bl_pbl_physics) in (2, 8, 9):", "    elif int(namelist.bl_pbl_physics) in (2, 9):", HIST),
    'S3_camuw_carry_seed_dropped': (O, "result = result.replace(camuw_pbl=camuw._replace(tke_pbl=camuw.tke_pbl.at[:-1].set(jnp.float32(.2))))",
                                    "result = result.replace(camuw_pbl=camuw)", HIST),
    'S4_cam_top_memory_overwritten': (O, "            updates['cam_tke_pbl'] = state.cam_tke_pbl.at[:-1].set(tke)\n",
                                      "            updates['cam_tke_pbl'] = state.cam_tke_pbl.at[:].set(tke)\n", HIST),
    'DX1_pthl_no_pmax_clamp': (K, "    val = 0.280 * num / den + (1.0 - 0.280)  # a7*num/den+(1.-a7), REAL 1.-.28 == .72\n    return jnp.clip(jnp.where(h != 0.0, val, 1.0), 0.0, 1.0)",
                               "    val = 0.280 * num / den + (1.0 - 0.280)  # a7*num/den+(1.-a7), REAL 1.-.28 == .72\n    return jnp.maximum(jnp.where(h != 0.0, val, 1.0), 0.0)", DXTEST),
    'DX2_pthnl_no_pmax_clamp': (K, "    val = 0.243 * num / den + _ONE_MINUS_A7_PTHNL  # a7*num/den+(1.-a7) (:2378)\n    return jnp.clip(jnp.where(h != 0.0, val, 1.0), 0.0, 1.0)",
                                "    val = 0.243 * num / den + _ONE_MINUS_A7_PTHNL  # a7*num/den+(1.-a7) (:2378)\n    return jnp.maximum(jnp.where(h != 0.0, val, 1.0), 0.0)", DXTEST),
}
DISCLOSURE = {f'{t}_vs_v040_3km': (*MUTANTS[t][:3], V040) for t in ('DX1_pthl_no_pmax_clamp', 'DX2_pthnl_no_pmax_clamp')}
env = dict(os.environ, JAX_PLATFORMS='cpu', GPUWRF_JAX_CACHE='0')


HANGS = []


def run(tag, tests):
    """One pytest per tag; a finished junit file is reused. CPU JAX can freeze in futex waits
    under core contention (FINDINGS E105/E119): a timed-out run is killed and retried once."""
    xml = BASE / f'{tag}.xml'
    rc = 'reused'
    for attempt in (1, 2):
        if xml.exists():
            break
        try:
            rc = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests, '-p', 'no:cacheprovider', '-W', 'ignore',
                                 '--basetemp', str(BASE / f'{tag}_{attempt}'), f'--junitxml={xml}'], cwd=REPO, env=env,
                                capture_output=True, text=True, timeout=420).returncode
        except subprocess.TimeoutExpired:
            HANGS.append(dict(tag=tag, attempt=attempt)); rc = 'timeout'
    assert xml.exists(), f'{tag}: no junit report after 2 attempts'
    fails, errors, passed = [], [], 0
    for tc in ET.parse(xml).getroot().iter('testcase'):
        f = tc.find('failure'); e = tc.find('error')
        if e is not None: errors.append(tc.get('name'))
        elif f is not None: fails.append((tc.get('name'), (f.get('message') or '')[:160]))
        elif tc.find('skipped') is None: passed += 1
    return dict(rc=rc, passed=passed, failures=fails, errors=errors)


BASE.mkdir(parents=True, exist_ok=True)
rows = {'control_hist': run('control_hist', HIST), 'control_dx': run('control_dx', DXTEST),
        'control_v040': run('control_v040', V040)}
for tag, (path, old, new, tests) in {**MUTANTS, **DISCLOSURE}.items():
    target = REPO / path
    original = target.read_bytes()
    text = original.decode()
    assert text.count(old) == 1, (tag, text.count(old))
    try:
        target.write_text(text.replace(old, new))
        r = run(tag, tests)
    finally:
        target.write_bytes(original)
    assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(original).digest()
    genuine = [n for n, m in r['failures'] if m.startswith(('AssertionError', 'assert', 'Failed: DID NOT RAISE'))
               or 'AssertionError' in m or 'Arrays are not equal' in m or 'DID NOT RAISE' in m]
    r.update(killed=bool(genuine) and not r['errors'], genuine_failures=genuine)
    rows[tag] = r
summary = dict(controls={k: dict(passed=rows[k]['passed'], failures=rows[k]['failures'], errors=rows[k]['errors'])
                         for k in ('control_hist', 'control_dx', 'control_v040')},
               killed=sum(rows[t]['killed'] for t in MUTANTS), total=len(MUTANTS),
               v040_3km_also_kills={t: rows[t]['killed'] for t in DISCLOSURE}, hangs_retried=HANGS, rows=rows)
OUT.write_text(json.dumps(summary, indent=1) + '\n')
print(json.dumps({k: (v['killed'] if 'killed' in v else v['passed'], len(v['failures']), v['errors']) for k, v in rows.items()}))
