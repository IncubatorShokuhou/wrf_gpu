"""Standalone unchanged WRF glacier model, with an ISO C buffer wrapper."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_noahmp_glacier.F')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'


def layout():
    source = WRF.read_text()
    body = source[source.index('  SUBROUTINE NOAHMP_GLACIER ('):source.index('  END SUBROUTINE NOAHMP_GLACIER')]
    variables = []
    for line in body.splitlines():
        line = line.split('!')[0]
        if '::' not in line or 'INTENT' not in line:
            continue
        head, name = line.split('::')
        name = name.strip().lower()
        if name == 'sfcheadrt':
            continue  # WRF_HYDRO disabled, as in the operational caller.
        match = re.search(r'DIMENSION\(([^)]+)\)', head)
        bounds = (match[1].strip().replace('NSNOW', '3').replace('NSOIL', '4') if match else None)
        extent = 1
        if bounds:
            lo, hi = bounds.split(':')
            extent = eval(hi, {'__builtins__': {}}, {}) - eval(lo, {'__builtins__': {}}, {}) + 1
        variables.append(dict(name=name, extent=extent, bounds=bounds,
                              integer=head.strip().startswith('INTEGER'),
                              intent=re.search(r'INTENT\((\w+)\)',head)[1]))
    return variables


def build(target):
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(WRF, target/WRF.name)
    fields = layout()
    offset = 0
    declaration, unpack, pack = [], [], []
    for field in fields:
        name, size = field['name'], field['extent']
        field['offset'] = offset
        kind = 'integer' if field['integer'] else 'real'
        declaration.append(f' {kind} :: {name}'+(f'({field["bounds"]})' if field['bounds'] else ''))
        if name in ('nsnow','nsoil'):
            unpack.append(f' {name} = {3 if name == "nsnow" else 4}')
        else:
            expression = f'input({offset+1}:{offset+size})' if size>1 else f'input({offset+1})'
            unpack.append(f' {name} = '+(f'int({expression})' if field['integer'] else expression))
        pack.append(f' output({offset+1}:{offset+size}) = {name}')
        offset += size
    source = WRF.read_text()
    signature = source.split('  SUBROUTINE NOAHMP_GLACIER (', 1)[1].split(')', 1)[0]
    signature = '\n'.join(line.split('!')[0] for line in signature.splitlines()
                          if not line.lstrip().startswith('#') and 'sfcheadrt' not in line)
    call = signature.replace('&', '').replace('\n', '').replace(' ', '').lower()
    wrapper = '\n'.join(['subroutine glacier_eval(input, output, options) bind(C)',
        ' use iso_c_binding', ' use module_sf_noahmp_glacier', ' implicit none',
        f' real(c_float), intent(in) :: input({offset})',
        f' real(c_float), intent(out) :: output({offset})',
        ' integer(c_int), intent(in) :: options(5)', *declaration,
        ' call noahmp_options_glacier(options(1), options(2), options(3), options(4), options(5))',
        *unpack, f' call noahmp_glacier({call})', *pack, 'end subroutine',
        'subroutine wrf_debug(level, message)', ' integer :: level', ' character(*) :: message', 'end subroutine',
        'subroutine wrf_message(message)', ' character(*) :: message', 'end subroutine',
        'subroutine wrf_error_fatal(message)', ' character(*) :: message',
        " print *, 'ORACLE_FATAL ', message", ' error stop 77', 'end subroutine', ''])
    (target/'wrapper.f90').write_text(wrapper)
    receipts = []
    for variant, init in [('zero', 'zero'), ('alternate', 'nan')]:
        command = [FC, '-O2', '-shared', '-fPIC', '-cpp', '-ffree-form', '-ffree-line-length-none',
            '-ffp-contract=off', '-finit-real='+init, '-finit-integer=0', '-finit-logical=false',
            'module_sf_noahmp_glacier.F', 'wrapper.f90', '-o', f'glacier_{variant}.so']
        subprocess.run(command, cwd=target, check=True, timeout=60)
        receipts.append(dict(variant=variant, command=command))
    manifest = dict(source=str(WRF), source_sha256=hashlib.sha256(WRF.read_bytes()).hexdigest(),
        fields=fields, width=offset, options=['opt_alb','opt_snf','opt_tbot','opt_stc','opt_gla'],
        compiler=subprocess.check_output([FC,'--version'],text=True).splitlines()[0], builds=receipts,
        caller='module_sf_noahmpdrv.F:1049-1063; Q_ML specific humidity, TBOT<=263.15, FICEOLD entry snow ice fraction',
        tolerances='literal binary32 byte identity; compiled within independently measured WRF FMA spread/input-ULP propagation; reject nonfinite')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(width=offset, fields=len(fields), source_sha256=manifest['source_sha256'])))


if __name__ == '__main__':
    build(Path(sys.argv[1]))
