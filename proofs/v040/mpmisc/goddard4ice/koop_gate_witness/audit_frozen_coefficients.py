"""Read selected pristine initialized frozen coefficients; never import JAX."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

CACHE = Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38/H38-03')
SOURCE_SHA = 'fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--tree', type=Path, default=Path.cwd())
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table = CACHE / 'qr_acr_qg_mp38V1.dat'
    before = table.stat()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(CACHE / 'module_mp_thompson.F') == SOURCE_SHA
    expressions = ([f'{name}({i})' for name, n in [('cie', 7), ('cig', 7), ('cre', 13), ('crg', 13)] for i in range(1, n + 1)]
                   + ['am_i', 'bm_i', 'av_i', 'bv_i', 'oig1', 'oig2', 'obmi', 'ogg2', 'ogg3', 'obmg', 'rho_not', 'a_coeff', 'b_coeff', 't2_qr_qi']
                   + [f'am_g({j})' for j in range(1, 10)]
                   # Only the distribution/fall-speed/sublimation rows used by mp38.
                   + [f'{name}({i},{j})' for name in ['cge', 'cgg'] for j in range(1, 10) for i in [2, 3, 6, 7, 9, 11, 12]])
    code = ('import gdb,json; expressions=' + repr(expressions) + '; '
            'values={name:{"value":float(gdb.parse_and_eval(name)),"type":str(gdb.parse_and_eval(name).type),"bytes":int(gdb.parse_and_eval(name).type.sizeof)} for name in expressions}; '
            'open(' + repr(str(out / 'original_values.json')) + ',"w").write(json.dumps(values,indent=2))')
    script = out / 'frozen_coefficients.gdb'
    script.write_text('\n'.join(['set pagination off', 'set confirm off', 'break module_mp_thompson.F:1648', 'run 4', 'set language fortran', 'python ' + code, 'quit']) + '\n')
    proc = subprocess.run(['gdb', '-q', '-batch', '-x', str(script), str(CACHE / 'oracle_debug')], cwd=CACHE, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
    (out / 'gdb.log').write_text(proc.stdout)
    assert proc.returncode == 0, proc.stdout[-3000:]
    values = json.loads((out / 'original_values.json').read_text())
    assert all(v['bytes'] == 4 for v in values.values())
    constants_file = args.tree / 'src/gpuwrf/physics/thompson_constants.py'
    constants = runpy.run_path(str(constants_file))  # standalone math-only constants module
    hail_file = args.tree / 'src/gpuwrf/physics/thompson_hail/_coefficients.py'
    hail = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(hail_file.read_text()).body if isinstance(node, ast.Assign)}
    wrf = constants['WRF_REAL_CONSTANTS']
    rows = []

    def compare(expression, value, mapping, used=True):
        original = np.float32(values[expression]['value'])
        bound = np.float32(value)
        bits = lambda v: int(np.asarray(v).view(np.uint32))
        rows.append({'original_expression': expression, 'original_kind': values[expression]['type'], 'original_bytes': 4,
                     'mapping': mapping, 'used_binding': used, 'original_value': float(original), 'bound_REAL_value': float(bound),
                     'original_u32_hex': f'{bits(original):08x}', 'bound_u32_hex': f'{bits(bound):08x}',
                     'BYTE': bits(original) == bits(bound), 'integer_bit_distance': abs(bits(original) - bits(bound))})

    for name in ['AM_I', 'BM_I', 'AV_I', 'BV_I', 'OIG1', 'OIG2', 'OBMI']:
        compare(name.lower(), wrf.get(name, constants[name]), 'TC._am_i WRF_REAL_CONSTANTS' if name == 'AM_I' else 'TC.' + name)
    for name in ['CIE2', 'CIE3', 'CIE6', 'CIE7', 'CIG1', 'CIG2', 'CIG3', 'CIG5', 'CIG6', 'CIG7', 'CRG2', 'CRG3', 'CRG6', 'CRG7', 'CRG8', 'CRG9', 'CRG10', 'CRG11', 'CRG12']:
        base, index = name[:3], name[3:]
        # CRG8 is only a legacy initializer dependency of T2_QR_QI;
        # active native code uses the original folded prefix below.
        compare(f'{base.lower()}({index})', wrf.get(name, constants[name]), 'TC._wrf ' + name if name in ['CIG6', 'CRG7'] else 'TC.' + name, used=name != 'CRG8')
    compare('t2_qr_qi', wrf['T2_QR_QI'], 'TC._wrf("T2_QR_QI", legacy prefix)')
    for j in range(1, 10):
        compare(f'am_g({j})', hail['AM_G'][j-1], 'hail._coefficients.AM_G[' + str(j-1) + ']')
        for name in ['CGE', 'CGG']:
            for i in [2, 3, 6, 7, 9, 11, 12]:
                compare(f'{name.lower()}({i},{j})', hail[name][j-1][i-1], f'hail._coefficients.{name}[{j-1}][{i-1}]')
    compare('ogg2', np.float32(1) / np.float32(hail['CGG'][0][1]), 'hail.OGG2')
    compare('ogg3', np.float32(1) / np.float32(hail['CGG'][0][2]), 'hail.OGG3')
    compare('obmg', np.float32(1) / np.float32(3), 'hail.OBMG')
    compare('rho_not', np.float32(101325) / (np.float32(287.05) * np.float32(298)), 'hail.RHO_NOT')
    compare('a_coeff', .47244157, 'hail.A_COEFF')
    compare('b_coeff', .54698726, 'hail.B_COEFF')
    # Complete initialized gamma arrays retained as evidence; math.gamma is a
    # diagnostic baseline only for coefficients not bound by the listed code.
    diagnostic = []
    for name, n in [('cig', 7), ('crg', 13)]:
        for i in range(1, n+1):
            expr = f'{name}({i})'
            exponent = values[f'{"cie" if name == "cig" else "cre"}({i})']['value']
            before_count = len(rows)
            compare(expr, math.gamma(exponent), 'diagnostic math.gamma(original REAL exponent)', used=False)
            diagnostic.append(rows.pop())
            assert len(rows) == before_count
    after = table.stat()
    unchanged = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert unchanged
    assert 'jax' not in sys.modules
    receipt = {'source_sha256': SOURCE_SHA, 'debug_object_sha256': sha(CACHE / 'thompson_debug.o'),
               'constants_source_sha256': sha(constants_file), 'hail_coefficients_sha256': sha(hail_file),
               'original_build': 'cached pristine O0-g object, zero init, no FMA; production aerosol and hail initialization',
               'breakpoint': 'module_mp_thompson.F:1648 run4 after original thompson_init', 'JAX_imported': False,
               'cached_table_unchanged': unchanged, 'original_values': values, 'bindings': rows,
               'binding_count': len(rows), 'binding_BYTE_count': sum(r['BYTE'] for r in rows),
               'effective_binding_count': sum(r['used_binding'] for r in rows),
               'effective_binding_BYTE_count': sum(r['BYTE'] and r['used_binding'] for r in rows),
               'diagnostic_math_gamma_only': diagnostic,
               'precision_scope': 'All selected initialized coefficients are REAL(kind=4). Original lami/ilami/lamg and rate DOUBLE islands are separate and not changed by this audit.'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['binding_count', 'binding_BYTE_count', 'effective_binding_count', 'effective_binding_BYTE_count', 'cached_table_unchanged', 'JAX_imported']}))
    print(json.dumps([r for r in rows if not r['BYTE']], indent=2))


if __name__ == '__main__':
    main()
