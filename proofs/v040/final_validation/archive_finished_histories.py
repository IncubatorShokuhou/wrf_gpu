"""Retain finished GPU evidence while meeting scratch and ROOT storage floors."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

assert os.environ['JAX_PLATFORMS'] == 'cpu'
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
DATA = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep')
DEST = ROOT / 'cold_gpu_archives/20261015_50GiB'
FLOOR = 45 * 1024**3
for case in ('c04', 'c05', 'c06'):
    q = json.loads((ROOT / case / 'QUALIFICATION.json').read_text())
    assert q['cache_remaining'] is False, 'Finish WSM01 consumers first'


def guard():
    while Path('/tmp/wrf_gpu2_quiet').exists():
        time.sleep(3)
    assert shutil.disk_usage(ROOT).free >= FLOOR, 'ROOT45GiB floor'
    assert shutil.disk_usage(DATA).free >= 120 * 1024**3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            guard()
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


guard()
DEST.mkdir(parents=True, exist_ok=True)
# Only historical GPU wrfouts: preserve receipts/PINs and every CPU/input file.
old = []
for case in ('c01', 'c02', 'c04', 'c07', 'c08', 'c10', 'c11', 'c12',
             'c14', 'c21', 'c22', 'c23', 'c24', 'c26', 'c28', 'c29',
             'c42', 'c45', 'c64'):
    old += sorted((DATA / 'arms' / case / 'attempts').glob('*/gpu/wrfout*'))
assert old and all(p.is_file() and not p.is_symlink() for p in old)
archive = DEST / 'historical_gpu_wrfouts.zip'
assert not archive.exists()
members = []
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                     compresslevel=1, allowZip64=True) as z:
    for p in old:
        member = str(p.relative_to(DATA))
        original_sha = digest(p)
        with p.open('rb') as src, z.open(member, 'w', force_zip64=True) as dst:
            for block in iter(lambda: src.read(1024**2), b''):
                guard()
                dst.write(block)
        members.append(dict(original_path=str(p), member=member,
                            bytes=p.stat().st_size, original_sha256=original_sha))
archive_sha = digest(archive)
with zipfile.ZipFile(archive) as z:
    for row in members:
        h = hashlib.sha256()
        with z.open(row['member']) as f:
            for block in iter(lambda: f.read(1024**2), b''):
                guard()
                h.update(block)
        assert h.hexdigest() == row['original_sha256']
        assert digest(Path(row['original_path'])) == row['original_sha256']
zip_receipt = dict(utc=datetime.now(timezone.utc).isoformat(), archive=str(archive),
    compressed_sha256=archive_sha, compressed_bytes=archive.stat().st_size,
    members=members, verification='Decompressed original FILE BYTES SHA-identical; all values and metadata therefore identical',
    authority='Manager storage approval with ROOT>=45GiB; E165 copy+receipt chain',
    scope='Historical GPU wrfouts only. Existing original-SHA manifests preserved; archived bytes recoverable by named ZIP member.')
save(DEST / 'HISTORICAL_WRFOUTS_RECEIPT.json', zip_receipt)
for row in members:
    p = Path(row['original_path'])
    guard()
    # Receipt is sealed BEFORE removing this exact verified scratch duplicate.
    assert p.is_relative_to(DATA / 'arms') and '/attempts/' in str(p) and p.name.startswith('wrfout')
    p.unlink()
for parent in {Path(r['original_path']).parent for r in members}:
    save(parent / 'ARCHIVED_WRFOUTS.json', dict(archive=str(archive),
        compressed_sha256=archive_sha, receipt=str(DEST / 'HISTORICAL_WRFOUTS_RECEIPT.json'),
        members=[r for r in members if Path(r['original_path']).parent == parent]))

relocations = []
for source in (DATA / 'arms/WN3-smoke/gpu', DATA / 'arms/c42/gpu'):
    guard()
    assert source.is_dir() and not source.is_symlink()
    files = sorted(p for p in source.rglob('*') if p.is_file())
    assert files and not any(p.is_symlink() for p in files)
    required = sum(p.stat().st_size for p in files)
    assert shutil.disk_usage(ROOT).free - required >= FLOOR + 512 * 1024**2, 'Raw copy cannot fit ROOT floor; retain originals'
    target = DEST / source.parent.name / 'gpu'
    assert not target.exists()
    target.mkdir(parents=True)
    manifest = []
    for p in files:
        out = target / p.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        original_sha = digest(p)
        with p.open('rb') as src, out.open('xb') as dst:
            for block in iter(lambda: src.read(1024**2), b''):
                guard()
                dst.write(block)
        shutil.copystat(p, out)
        assert digest(out) == original_sha == digest(p)
        manifest.append(dict(original_path=str(p), retained_path=str(out),
                             bytes=p.stat().st_size, sha256=original_sha))
    receipt = dict(original_directory=str(source), retained_directory=str(target),
                   bytes=required, files=manifest, verification='All original bytes SHA-identical; logical paths unchanged through symlink')
    save(target.parent / 'RAW_RELOCATION_RECEIPT.json', receipt)
    staged = source.with_name('gpu.verified_archive_duplicate')
    assert not staged.exists()
    source.rename(staged)
    source.symlink_to(target, target_is_directory=True)
    assert source.resolve() == target.resolve()
    guard()
    shutil.rmtree(staged)
    relocations.append(receipt)
    print('relocated', source.parent.name, required, flush=True)
guard()
result = dict(utc=datetime.now(timezone.utc).isoformat(), zip_receipt=str(DEST / 'HISTORICAL_WRFOUTS_RECEIPT.json'),
    removed_scratch_wrfout_bytes=sum(r['bytes'] for r in members), raw_relocations=relocations,
    root_free_GiB=round(shutil.disk_usage(ROOT).free/1024**3, 3),
    data_free_GiB=round(shutil.disk_usage(DATA).free/1024**3, 3),
    scope='CPU references and case inputs untouched; only finished GPU evidence relocated/losslessly archived')
save(ROOT / 'GPU23_STORAGE_TRIM.json', result)
print('ROOT freeGiB', result['root_free_GiB'], flush=True)
