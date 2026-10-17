"""Recover ROOT headroom using verified copies and explicit E165 SHA chains."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

assert os.environ['JAX_PLATFORMS'] == 'cpu'
ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
ARCHIVE = ROOT / 'cold_gpu_archives/20261015_50GiB'
spec = importlib.util.spec_from_file_location('checked_deflate',
    Path(__file__).resolve().parents[3] / 'scripts/compress_wrfout.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
TARGET = 48 * 1024**3


def quiet():
    while Path('/tmp/wrf_gpu2_quiet').exists():
        time.sleep(3)


def save(path, obj):
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


changed = []
before = shutil.disk_usage('/').free
for case in ('WN3-smoke', 'c42'):
    raw = ARCHIVE / case / 'gpu'
    original = json.loads((raw.parent / 'RAW_RELOCATION_RECEIPT.json').read_text())
    delivered = {Path(x['retained_path']).name: x['sha256'] for x in original['files']}
    dest = raw.parent / 'deflate_gpu'
    dest.mkdir(exist_ok=True)
    manifest_path = ROOT / case / 'GPU_HISTORY_STORAGE.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        schema='e165-post-validation-storage-v1', case=case, files={},
        original_delivery_receipt=str(raw.parent / 'RAW_RELOCATION_RECEIPT.json'),
        authority='Manager: deflate copies permitted; ROOT>=45GiB, scratch<=50GiB',
        scope='Post-validation storage only; unchanged model, CPU references, inputs, values and attributes. Original delivery SHA retained beside current compressed-copy SHA.')
    for p in sorted(raw.glob('wrfout*')):
        if shutil.disk_usage('/').free >= TARGET:
            break
        quiet()
        if p.is_symlink():
            rec = manifest['files'][p.name]
            assert helper.sha256(p) == rec['comp_sha256']
            continue
        orig_sha = helper.sha256(p)
        assert orig_sha == delivered[p.name], 'Original bytes differ from sealed delivery'
        out = dest / p.name
        assert not out.exists()
        subprocess.run(['nccopy', '-d1', '-s', str(p), str(out)], check=True, timeout=90)
        assert not helper.mismatches(p, out), 'Values/attributes changed'
        # Stronger than NaN-aware numerical equality: preserve every array bit,
        # including signed zero, char storage and NaN payloads.
        with helper.Dataset(p) as src, helper.Dataset(out) as dst:
            src.set_auto_maskandscale(False)
            dst.set_auto_maskandscale(False)
            src.set_auto_chartostring(False)
            dst.set_auto_chartostring(False)
            for name in src.variables:
                assert src[name][:].tobytes() == dst[name][:].tobytes(), name
        assert helper.sha256(p) == orig_sha
        rec = dict(original_path=str(p), compressed_copy=str(out),
            orig_sha256=orig_sha, comp_sha256=helper.sha256(out),
            orig_bytes=p.stat().st_size, comp_bytes=out.stat().st_size,
            values_bitwise_identical=True, dimensions_and_attributes_identical=True,
            utc=datetime.now(timezone.utc).isoformat())
        manifest['files'][p.name] = rec
        # Reissue CURRENT manifest BEFORE swapping its historical logical path.
        # The manifest explicitly points at the already-verified new copy.
        save(manifest_path, manifest)
        with (dest.parent / 'DEFLATE_COPY_RECEIPTS.jsonl').open('a') as f:
            f.write(json.dumps(rec) + '\n')
        link = raw / (p.name + '.verified-copy-link')
        link.symlink_to(out)
        link.replace(p)
        assert helper.sha256(p) == rec['comp_sha256']
        changed.append(rec)
        print(case, p.name, rec['orig_bytes'], '->', rec['comp_bytes'],
              'ROOTfreeGiB', round(shutil.disk_usage('/').free/1024**3, 3), flush=True)
    if shutil.disk_usage('/').free >= TARGET:
        break
result = dict(utc=datetime.now(timezone.utc).isoformat(), changed_files=len(changed),
    original_bytes=sum(r['orig_bytes'] for r in changed), compressed_bytes=sum(r['comp_bytes'] for r in changed),
    root_free_before_GiB=round(before/1024**3, 3), root_free_after_GiB=round(shutil.disk_usage('/').free/1024**3, 3),
    per_file_receipts=changed, scope='E165 verified compressed COPIES, reissued current SHA manifest, logical paths retained. CPU references/inputs untouched; all field arrays BIT-identical, all metadata identical; no numerical rerun required.')
save(ROOT / 'ROOT_HEADROOM_RECOVERY.json', result)
assert shutil.disk_usage('/').free >= 45 * 1024**3, 'ROOT floor still needs recovery'
print('recovered', result['root_free_after_GiB'], 'GiB', flush=True)
