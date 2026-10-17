"""Scan the paid c26 executables; retain exact native QNSE evidence."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tree', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert os.environ['JAX_PLATFORMS'] == 'cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    assert a.out.resolve().is_relative_to(Path('<USER_HOME>/wrf_gpu2_lanes/t-qnsecompile'))
    sys.path.insert(0, str(a.tree / 'src'))
    spec = importlib.util.spec_from_file_location('paid_wave_gate', a.tree / 'scripts/wave_cubin_gate.py')
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    a.out.mkdir(parents=True, exist_ok=False)
    pins = a.out / 'PIN'
    pins.mkdir()
    entries = gate.discover(a.cache)
    inventory = []
    for i, entry in enumerate(entries):
        target = pins / f'{i:02d}_{entry.kind}_{entry.path.name}'
        shutil.copy2(entry.path, target)
        data = entry.path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        inventory.append(dict(origin=str(entry.path), pin=str(target), kind=entry.kind,
                              program=entry.program, aliases=entry.aliases,
                              bytes=len(data), sha256=digest))
    (a.out / 'PIN.json').write_text(json.dumps(inventory, indent=2)+'\n')
    result = gate.run_gate(a.cache, a.out / 'e41')
    print('ACTUAL E41', result['e41_gate'], result['coverage'], flush=True)
    import zstandard
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    from gpuwrf.kernels import phys_qnse_columns  # noqa: F401
    native = {}
    for entry in entries:
        blob = entry.path.read_bytes()
        if entry.kind == 'jax_cache':
            blob = zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 33)[4:]
        raw = (gate._record(blob, 2, ax, asm)
               if ax._gpu_blob_format(blob) == ax.GPU_THUNK_FORMAT else blob)
        for cubin in gate._elfs(raw):
            symbols = [n for n in ('qnse_sfc_columns', 'qnse_pbl_columns', 'qnse_mf_columns')
                       if n.encode() in cubin]
            if symbols:
                digest = hashlib.sha256(cubin).hexdigest()
                target = a.out / f'qnse_{digest[:16]}.cubin'
                target.write_bytes(cubin)
                native[digest] = dict(path=str(target), bytes=len(cubin), symbols=symbols)
    assert native, 'Actual paid QNSE ELF missing from stepping executable'
    assert {'qnse_sfc_columns', 'qnse_pbl_columns'} <= {
        s for v in native.values() for s in v['symbols']}
    (a.out / 'QNSE_ACTUAL.json').write_text(json.dumps(native, indent=2)+'\n')
    print('Actual QNSE ELFs', native, flush=True)


if __name__ == '__main__':
    main()
