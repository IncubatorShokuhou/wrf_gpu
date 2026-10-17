"""Audit only actual case5 rain-freezing indices and selected DOUBLE entries."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import numpy as np

CACHE = Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38/H38-03')
STAGES = Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38/H38-46-literal6-stages.npz')
SHAPE = (37, 37, 45, 55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--reuse-capture', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table = CACHE / 'freezeH2O.dat'
    before = table.stat()
    source = CACHE / 'module_mp_thompson.F'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    source_sha = sha(source)
    assert source_sha == 'fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'
    # No body edits: the actual original lookup point supplies original indices.
    capture = ('k=int(gdb.parse_and_eval("k")); '
               'row={name:int(gdb.parse_and_eval(name)) for name in ["k","idx_r","idx_r1","idx_tc","idx_IN"]}; '
               'row["selected"]=(10<=k<=17); '
               'row["entries"]={name:{"value":float(gdb.parse_and_eval(name+"(idx_r,idx_r1,idx_tc,idx_IN)")),"kind":str(gdb.parse_and_eval(name+"(idx_r,idx_r1,idx_tc,idx_IN)").type),"bytes":int(gdb.parse_and_eval(name+"(idx_r,idx_r1,idx_tc,idx_IN)").type.sizeof)} for name in ["tpi_qrfz","tni_qrfz"]} if row["selected"] else {}; rows.append(row)')
    script = out / 'selected_qrfz.gdb'
    script.write_text('\n'.join(['set pagination off', 'set confirm off', 'set debuginfod enabled off',
                                  'set language fortran', 'python import gdb,json; rows=[]',
                                  'break module_mp_thompson.F:2598', 'commands', 'silent', 'python ' + capture,
                                  'continue', 'end', 'run 5',
                                  'python open(' + repr(str(out / 'original_lookups.json')) + ',"w").write(json.dumps(rows,indent=2))', 'quit']) + '\n')
    if not args.reuse_capture:
        proc = subprocess.run(['gdb', '-q', '-batch', '-x', str(script), str(CACHE / 'oracle_debug')], cwd=CACHE,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, check=False)
        (out / 'gdb.log').write_text(proc.stdout)
        assert proc.returncode == 0, proc.stdout[-3000:]
    rows = json.loads((out / 'original_lookups.json').read_text())
    assert len(rows) > 0
    index_checks = []
    with np.load(STAGES, allow_pickle=False) as stages:
        for row in rows:
            k = row['k']
            original = [row[n] for n in ['idx_r', 'idx_r1', 'idx_tc', 'idx_IN']]
            candidate = [int(stages['freezing_' + n][4, k-1]) + 1 for n in ['idx_r', 'idx_r1', 'idx_tc', 'idx_in']]
            index_checks.append({'fortran_level': k, 'original_indices_1based': original,
                                 'candidate_indices_1based': candidate, 'BYTE': original == candidate})
    assert all(row['BYTE'] for row in index_checks), index_checks
    selected = [row for row in rows if row['selected']]
    assert len(selected) == 8, len(selected)
    asset = Path('data/fixtures/thompson-tables-v1.npz')
    comparisons = []
    # Read selected NPY element offsets from the compressed member. No whole
    # table allocation/dump; seek/decompression is limited to the target member.
    with zipfile.ZipFile(asset) as archive, table.open('rb') as raw:
        record_size = int(np.prod(SHAPE)) * 8
        first_header = struct.unpack('>i', raw.read(4))[0]
        assert first_header == record_size, (first_header, record_size)
        for record_index, name in enumerate(['tpi_qrfz', 'tni_qrfz']):
            with archive.open(name + '.npy') as member:
                version = np.lib.format.read_magic(member)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran_order, dtype = reader(member)
                assert shape == SHAPE and dtype == np.dtype('float64'), (shape, dtype)
                header_bytes = member.tell()
                for row in selected:
                    index = tuple(row[n] - 1 for n in ['idx_r', 'idx_r1', 'idx_tc', 'idx_IN'])
                    offset = int(np.ravel_multi_index(index, SHAPE, order='F' if fortran_order else 'C')) * 8
                    member.seek(header_bytes + offset)
                    legacy = float(np.frombuffer(member.read(8), dtype=dtype)[0])
                    raw_offset = record_index * (record_size + 8) + 4 + int(np.ravel_multi_index(index, SHAPE, order='F')) * 8
                    raw.seek(raw_offset)
                    original_file_value = struct.unpack('>d', raw.read(8))[0]
                    entry = row['entries'][name]
                    assert entry['bytes'] == 8
                    original = entry['value']
                    assert struct.pack('>d', original) == struct.pack('>d', original_file_value)
                    comparisons.append({'table': name, 'fortran_level': row['k'], 'fortran_indices_1based': [i+1 for i in index],
                                        'original_kind': entry['kind'], 'original_bytes': entry['bytes'],
                                        'legacy_asset_dtype': str(dtype), 'legacy_asset_fortran_order': fortran_order,
                                        'original_value': original, 'legacy_value': legacy,
                                        'original_hex': struct.pack('>d', original).hex(), 'legacy_hex': struct.pack('>d', legacy).hex(),
                                        'BYTE': original == legacy, 'original_cached_file_BYTE': True,
                                        'relative_difference': abs(original - legacy) / abs(original) if original else 0.0,
                                        'cached_file_element_offset': raw_offset})
    after = table.stat()
    unchanged = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert unchanged and 'jax' not in sys.modules
    rate_checks = []
    odts = np.float32(1.) / np.float32(90.)  # original H38-03 caller DT=90
    with np.load(STAGES, allow_pickle=False) as stages:
        for row in comparisons:
            name = 'pri_rfz' if row['table'] == 'tpi_qrfz' else 'pni_rfz'
            index = row['fortran_level'] - 1
            reference_path = CACHE.parent / 'H38-48' / ('case5_' + name + '.bin')
            original_rate = np.fromfile(reference_path, dtype='<f8')[index]
            candidate_rate = stages['freezing_' + name][4, index]
            original_projected = np.float64(row['original_value']) * np.float64(odts)
            candidate_projected = np.float64(row['legacy_value']) * np.float64(odts)
            rate_checks.append({'fortran_level': index+1, 'rate': name,
                                'original_table_times_REAL_odts_BYTE': bool(original_projected == original_rate),
                                'legacy_table_times_REAL_odts_BYTE': bool(candidate_projected == candidate_rate),
                                'original_rate_hex': struct.pack('>d', original_rate).hex(),
                                'candidate_rate_hex': struct.pack('>d', candidate_rate).hex()})
    assert all(r['legacy_table_times_REAL_odts_BYTE'] for r in rate_checks), rate_checks
    # H48:2960 observes mass after the original rain-availability limiter
    # at2904-2910. Number is unchanged there and directly proves projection.
    assert all(r['original_table_times_REAL_odts_BYTE'] for r in rate_checks if r['rate'] == 'pni_rfz'), rate_checks
    receipt = {'source_sha256': source_sha, 'debug_object_sha256': sha(CACHE / 'thompson_debug.o'),
               'legacy_asset_path': str(asset), 'legacy_asset_sha256': sha(asset), 'stage_path': str(STAGES), 'stage_sha256': sha(STAGES),
               'cached_freeze_table_path': str(table.resolve()), 'cached_freeze_table_size': before.st_size,
               'cached_freeze_table_inode': before.st_ino, 'cached_freeze_table_mtime_ns': before.st_mtime_ns,
               'cached_table_unchanged': unchanged, 'JAX_imported': False, 'index_checks': index_checks,
               'index_count': len(index_checks) * 4, 'index_BYTE_count': sum(r['BYTE'] for r in index_checks) * 4,
               'selected_entries': comparisons, 'selected_entry_count': len(comparisons), 'selected_entry_BYTE': sum(r['BYTE'] for r in comparisons),
               'paired_rate_projection': rate_checks, 'original_odts_REAL_hex': struct.pack('>f', odts).hex(),
               'projection_stage_scope': 'Candidate freezing captures are pre-limiter, all16 projected entries BYTE. Original H48:2960 PNI_RFZ is unchanged and all8 projected entries BYTE. Original PRI_RFZ has already been rain-limited at2904-2910; 5/8 levels do not equal the pre-limiter table product and are not claimed as direct projection matches.',
               'source_recipe': 'Original freezeH2O reads tpi_qrfz then tni_qrfz DOUBLE records at4630/4631. Calculation branch4710-4731 uses REAL fold inputs with DOUBLE lam_exp/lamr/N0_r, descending DOUBLE rain-bin accumulation and rounded REAL T_adjust. Runtime original2597/2598 multiplies selected DOUBLE entry by REAL odts. Legacy extraction script uses nvfortran and a patched dumper, but exact historical producer/math cause is not established by this selected audit.',
               'scope': 'Actual case5 original active lookup indices all checked before table comparison. Only selected8 levels x2 DOUBLE entries retained; no table rebuild, full-table dump or physics production edits.'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['index_count', 'index_BYTE_count', 'selected_entry_count', 'selected_entry_BYTE', 'cached_table_unchanged']}))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
