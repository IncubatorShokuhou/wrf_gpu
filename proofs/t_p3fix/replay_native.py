"""Replay the exact retained c21 P3 cubin, without compiling another kernel.

The 39-pointer ABI is checked against the source and cuobjdump before launch.
Inputs are the independent original-WRF 44-level oracle. Only eight columns
launch; full production-sized buffers preserve the original allocation bounds.
All transfers precede or follow the three device calls.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ('qv', 'qc', 'qr', 'qi', 'ni', 'nr', 'nc', 'qir', 'bir',
          'th_old', 'qv_old', 'th', 'p', 'dz')
OUTPUTS = ('qv', 'qc', 'qr', 'qi', 'ni', 'nr', 'qir', 'bir',
           'th_old', 'qv_old', 're_cloud', 're_ice', 'th', 'rainncv', 'snowncv')
ELF_SHA = '5beb6df6764c33ddf9003d01339316f2d08351545c4515b97fc75cc48cc9b2d1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_data(fixture):
    with np.load(fixture, allow_pickle=False) as f:
        reference = dict(f)
    arrays = {}
    for name in INPUTS:
        value = reference[name][..., 0]
        assert value.shape == (8, 44) and value.dtype == np.float32
        assert np.isfinite(value).all()
        array = np.broadcast_to(value[:1], (1764, 44)).copy()
        array[:8] = value
        arrays[name] = array
    for kind in ('2mom', 'warm'):
        with np.load(ROOT / f'src/gpuwrf/physics/p3/data/{kind}.npz', allow_pickle=False) as f:
            for name in f.files:
                arrays[f'table.{name}'] = np.ascontiguousarray(f[name], dtype=np.float32)
    arrays['table.gamma_factorials'] = np.array([float(math.factorial(k)) for k in range(35)], np.float32)
    spec = importlib.util.spec_from_file_location('p3_constants', ROOT / 'src/gpuwrf/physics/p3/constants.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arrays['constants'] = np.array(tuple(module.CONSTANTS.values()), np.float32)
    arrays['dt'] = np.array(18, np.float32)
    for step in (1, 2, 3):
        arrays[f'it{step}'] = np.array(step, np.int32)
    table_names = tuple(name for name in arrays if name.startswith('table.'))
    assert len(table_names) == 7
    return reference, arrays, table_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cubin', type=Path, required=True)
    p.add_argument('--elf-info', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    assert sha(a.cubin) == ELF_SHA
    text = a.elf_info.read_text()
    section = text.split('\n.nv.info.p3_mp50_column\n', 1)[1].split('\n.nv.info.', 1)[0]
    rows = re.findall(r'Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)', section)
    records = sorted((int(i, 16), int(off, 16), int(size, 16)) for i, off, size in rows)
    assert records == [(i, 8*i, 8) for i in range(39)], records
    reference, arrays, tables = host_data(a.fixture)
    abi = (*INPUTS, *tables, 'constants', 'dt', 'it', *('out.'+k for k in OUTPUTS))
    assert len(abi) == 39
    report = {'scope': 'actual c21 native P3 vs original WRF columns; no new compile/forecast',
              'cubin_sha256': ELF_SHA, 'fixture_sha256': sha(a.fixture), 'abi': abi,
              'grid': [8, 1, 1], 'block': [64, 1, 1], 'allocation_columns': 1764,
              'nz': 44, 'dt': 18, 'calls': 3, 'GPU_execution': False,
              'input_hashes': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    if a.prepare_only:
        return 0
    import os
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    import jax
    assert jax.devices()[0].platform == 'gpu'
    report.update(GPU_execution=True, device=str(jax.devices()[0]), device_kind=jax.devices()[0].device_kind)
    save()
    print('DEVICE', report['device'], report['device_kind'], flush=True)
    cuda = C.CDLL('libcuda.so.1')
    def function(name, argtypes):
        fn = getattr(cuda, name)
        fn.argtypes, fn.restype = argtypes, C.c_int
        def call(*args):
            rc = fn(*args)
            if rc:
                raise RuntimeError(f'{name}: CUDA error {rc}')
        return call
    ptr, u64, u32, integer = C.c_void_p, C.c_uint64, C.c_uint, C.c_int
    init = function('cuInit', [u32])
    getdev = function('cuDeviceGet', [C.POINTER(integer), integer])
    retain = function('cuDevicePrimaryCtxRetain', [C.POINTER(ptr), integer])
    setctx = function('cuCtxSetCurrent', [ptr])
    load = function('cuModuleLoadData', [C.POINTER(ptr), ptr])
    getfn = function('cuModuleGetFunction', [C.POINTER(ptr), ptr, C.c_char_p])
    attribute = function('cuFuncGetAttribute', [C.POINTER(integer), integer, ptr])
    alloc = function('cuMemAlloc_v2', [C.POINTER(u64), C.c_size_t])
    htod = function('cuMemcpyHtoD_v2', [u64, ptr, C.c_size_t])
    dtoh = function('cuMemcpyDtoH_v2', [ptr, u64, C.c_size_t])
    memset = function('cuMemsetD32_v2', [u64, u32, C.c_size_t])
    create_stream = function('cuStreamCreate', [C.POINTER(ptr), u32])
    synchronize = function('cuStreamSynchronize', [ptr])
    launch = function('cuLaunchKernel', [ptr, u32, u32, u32, u32, u32, u32, u32, ptr, C.POINTER(ptr), C.POINTER(ptr)])
    free = function('cuMemFree_v2', [u64])
    unload = function('cuModuleUnload', [ptr])
    destroy_stream = function('cuStreamDestroy_v2', [ptr])
    release = function('cuDevicePrimaryCtxRelease_v2', [integer])
    init(0)
    dev, ctx, mod, kernel, stream = integer(), ptr(), ptr(), ptr(), ptr()
    owned = []
    getdev(C.byref(dev), 0)
    retain(C.byref(ctx), dev)
    setctx(ctx)
    try:
        data = C.create_string_buffer(a.cubin.read_bytes())
        load(C.byref(mod), C.cast(data, ptr))
        getfn(C.byref(kernel), mod, b'p3_mp50_column')
        attrs = {}
        for name, number in (('max_threads', 0), ('static_shared', 1), ('local_bytes', 3), ('registers', 4)):
            value = integer()
            attribute(C.byref(value), number, kernel)
            attrs[name] = value.value
        assert attrs['registers'] == 159 and attrs['local_bytes'] == 0, attrs
        assert attrs['max_threads'] >= 64 and attrs['static_shared'] in (0, 1024), attrs
        # The retained ELF declares 1024 B shared scratch. CUDA 595.99 reports
        # zero static bytes for this Triton image, so allocate the remainder
        # dynamically; total launch scratch must still be exactly 1024 B.
        dynamic_shared = 1024 - attrs['static_shared']
        assert attrs['static_shared'] + dynamic_shared == 1024
        report['function_attributes'] = attrs
        report['shared_memory'] = dict(elf_bytes=1024, static_bytes=attrs['static_shared'],
                                     launch_dynamic_bytes=dynamic_shared)
        create_stream(C.byref(stream), 0)
        def memory(nbytes):
            result = u64()
            alloc(C.byref(result), max(nbytes, 256))
            owned.append(result)
            return result
        inputs = {}
        for name, value in arrays.items():
            inputs[name] = memory(value.nbytes)
            htod(inputs[name], ptr(value.ctypes.data), value.nbytes)
        outputs = []
        for step in (1, 2, 3):
            row = {}
            for name in OUTPUTS:
                shape = (1764,) if name in ('rainncv', 'snowncv') else (1764, 44)
                host = np.empty(shape, np.float32)
                device = memory(host.nbytes)
                memset(device, 0x7fc00000, host.size)
                row[name] = (device, host)
            outputs.append(row)
        current = {k: inputs[k] for k in INPUTS}
        started = time.monotonic()
        for step, row in enumerate(outputs, 1):
            args = [*(current[k] for k in INPUTS), *(inputs[k] for k in tables),
                    inputs['constants'], inputs['dt'], inputs[f'it{step}'], *(row[k][0] for k in OUTPUTS)]
            parameters = (ptr * len(args))(*(C.addressof(v) for v in args))
            launch(kernel, 8, 1, 1, 64, 1, 1, dynamic_shared, stream, parameters, None)
            current = {k: row[k][0] if k in row else v for k, v in current.items()}
        synchronize(stream)
        report['three_calls_s'] = time.monotonic()-started
        saved, scores = {}, []
        # Retain all raw results before any finite/numerical assertion, so a
        # genuine native failure remains diagnosable after the process exits.
        for step, row in enumerate(outputs, 1):
            for name, (device, host) in row.items():
                dtoh(ptr(host.ctypes.data), device, host.nbytes)
                saved[f'{name}__{step}'] = host[:8].copy()
        np.savez_compressed(a.out.with_suffix('.npz'), **saved)
        sys.path.insert(0, str(ROOT / 'proofs/s2_p3'))
        from comparison import compare_field
        for step, row in enumerate(outputs, 1):
            fields = {}
            for name in row:
                value = saved[f'{name}__{step}']
                fields[name] = compare_field(name, value, reference[name][..., step])
            scores.append({'step': step, 'fields': fields})
        report.update(scores=scores, strict_failures=sum(v['bad'] for row in scores for v in row['fields'].values()))
        save()
        print('SCORES', [{k: (v['bad'], v['max_abs']) for k, v in row['fields'].items()} for row in scores], flush=True)
        return 0
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    finally:
        for device in reversed(owned):
            free(device)
        if stream:
            destroy_stream(stream)
        if mod:
            unload(mod)
        release(dev)


if __name__ == '__main__':
    raise SystemExit(main())
