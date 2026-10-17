"""Capture original REAL sublimation scalars and their DOUBLE promotions."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys

import numpy as np

CACHE = Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38/H38-03')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table = CACHE / 'qr_acr_qg_mp38V1.dat'
    before = table.stat()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source_sha = sha(CACHE / 'module_mp_thompson.F')
    assert source_sha == 'fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'
    code = ('import gdb,json; names=["t1_qg_sd","t1_qs_sd","t2_qs_sd"]; '
            'values={name:{"value":float(gdb.parse_and_eval(name)),"kind":str(gdb.parse_and_eval(name).type),"bytes":int(gdb.parse_and_eval(name).type.sizeof)} for name in names}; '
            'open(' + repr(str(out / 'original_values.json')) + ',"w").write(json.dumps(values,indent=2))')
    commands = ['set pagination off', 'set confirm off', 'break module_mp_thompson.F:1648', 'run 4', 'set language fortran', 'python ' + code, 'quit']
    script = out / 'sublimation_scalars.gdb'
    script.write_text('\n'.join(commands) + '\n')
    process = subprocess.run(['gdb', '-q', '-batch', '-x', str(script), str(CACHE / 'oracle_debug')], cwd=CACHE,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
    (out / 'gdb.log').write_text(process.stdout)
    assert process.returncode == 0, process.stdout[-3000:]
    values = json.loads((out / 'original_values.json').read_text())
    constants_file = Path('src/gpuwrf/physics/thompson_constants.py')
    constants = runpy.run_path(str(constants_file))  # standalone math-only module
    rows = []
    for original, binding in [('t1_qg_sd', 'T1_SUBL_QG'), ('t1_qs_sd', 'T1_SUBL_QS'), ('t2_qs_sd', 'T2_SUBL_QS')]:
        captured = values[original]
        assert captured['bytes'] == 4
        reference = float(np.float32(captured['value']))
        legacy = constants[binding]
        rows.append({'original': original, 'binding': binding, 'kind': captured['kind'], 'bytes': captured['bytes'],
                     'original_REAL': reference, 'original_REAL_hex': struct.pack('>f', reference).hex(),
                     'legacy_python_float': legacy, 'legacy_cast_REAL_hex': struct.pack('>f', legacy).hex(),
                     'REAL_BYTE': struct.pack('>f', reference) == struct.pack('>f', legacy),
                     'original_REAL_promoted_DOUBLE': reference, 'original_promoted_DOUBLE_hex': struct.pack('>d', reference).hex(),
                     'legacy_DOUBLE_hex': struct.pack('>d', legacy).hex(), 'effective_DOUBLE_BYTE': reference == legacy,
                     'original_minus_legacy_DOUBLE': reference-legacy})
    after = table.stat()
    unchanged = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert unchanged and 'jax' not in sys.modules
    receipt = {'source_sha256': source_sha, 'debug_object_sha256': sha(CACHE / 'thompson_debug.o'),
               'constants_source_sha256': sha(constants_file), 'breakpoint': 'original1648/run4 after production thompson_init',
               'original_source': {'t1_qg_sd': ':812 REAL assignment .86*cgg(10,1)', 't1_qs_sd': ':804 REAL assignment .86',
                                   't2_qs_sd': ':805 REAL assignment .28*Sc3*SQRT(av_s)'},
               'promotion_scope': 'Original :2699/:2829 multiplies REAL t1_qg_sd with DOUBLE ilamg**REAL cge(10,1), promoting the already-rounded REAL coefficient to DOUBLE. TC :3205 uses Python T1_SUBL_QG with DOUBLE _dpow(ilamg,...). Snow scalar promotion depends on its actual operand kinds; no snow causality claimed.',
               'JAX_imported': False, 'cached_table_unchanged': unchanged, 'rows': rows}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
