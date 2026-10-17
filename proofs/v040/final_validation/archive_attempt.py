"""Preserve an owned GPU attempt before a changed-source retry; CPU is read-only."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep')
DATA = Path('<DATA_ROOT>/wrf_gpu2/scratch/v-prep/arms')
QUIET = Path('/tmp/wrf_gpu2_quiet')
assert os.environ.get('JAX_PLATFORMS') == 'cpu'
reason, *cases = sys.argv[1:]
assert re.fullmatch(r'[a-zA-Z0-9_-]+', reason) and cases
assert shutil.disk_usage('<DATA_ROOT>').free >= 120 * 1024**3


def quiet():
    while QUIET.exists():
        time.sleep(3)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


for ident in cases:
    assert re.fullmatch(r'c\d\d', ident)
    case = ROOT / ident
    assert not (case / 'cache').exists(), 'Finish E41/PIN/cache consumers first'
    # A new arm must never be launched before this staging command finishes.
    original = {p.name: sha(p) for p in (case / 'input').iterdir() if p.is_file()}
    cpu_receipt = sha(case / 'cpu_run.json')
    target = DATA / ident / 'attempts' / reason
    target.mkdir(parents=True, exist_ok=False)
    mapping, files, external_links = {}, [], []
    for name in ('gpu', 'gpu_proof', 'gpu_scratch', 'e41', 'tmp'):
        logical = case / name
        if not logical.exists():
            continue
        physical = logical.resolve()
        owned = physical.is_relative_to(ROOT / ident) or physical.is_relative_to(DATA / ident)
        if not owned:
            # Qualified reuse points at a read-only producer. Archive only our
            # link and receipt; never move or delete the producer's directory.
            assert name == 'gpu' and logical.is_symlink()
            reuse_path = case / 'GPU_REUSE.json'
            reuse = json.loads(reuse_path.read_text())
            assert physical == (Path(reuse['producer_case']) / 'gpu').resolve()
            producer_receipt = Path(reuse['producer_receipt'])
            digest = sha(producer_receipt)
            assert digest == reuse['producer_receipt_sha256']
            external_links.append(dict(link=str(logical), target=str(physical),
                reuse_receipt_sha256=sha(reuse_path), producer_receipt_sha256=digest,
                action='Removed only owned symlink; external producer untouched'))
            quiet()
            logical.unlink()
            assert sha(producer_receipt) == digest
            continue
        quiet()
        for p in physical.rglob('*'):
            if p.is_file() and not p.is_symlink():
                quiet()
                files.append(dict(old_path=str(p), archive=str(target / name / p.relative_to(physical)),
                                  bytes=p.stat().st_size, sha256=sha(p)))
        shutil.move(str(physical), str(target / name))
        mapping[str(logical)] = str(target / name)
        if logical.is_symlink():
            # Re-create the physical spill target for the next arm. The old
            # attempt is immutable in its archive; ROOT link stays stable.
            physical.mkdir()
    patterns = ('gpu*.json', 'gpu*.log', 'QUALIFICATION.json', 'REFERENCE_OF_RECORD.json', 'D6_*.json', 'compare_*.json',
                'GPU_REUSE.json', 'SCORING_IO_*.json', 'ADMITTED_SCOPE_GATE.json', 'FAILED_SETUP_CLEANUP.json', 'WRITER_RECHECK.json',
                'target_config*', 'MAIN_SEARCH_BEFORE_GPU.json', 'SPILL_PATHS.json',
                'e41*.log', 'score*.log', 'cleanup*.log', '*INVENTORY*.json')
    selected = {p for pattern in patterns for p in case.glob(pattern) if p.is_file() and not p.is_symlink()}
    for p in sorted(selected):
        quiet()
        files.append(dict(old_path=str(p), archive=str(target / p.name), bytes=p.stat().st_size, sha256=sha(p)))
        shutil.move(str(p), str(target / p.name))
    assert original == {p.name: sha(p) for p in (case / 'input').iterdir() if p.is_file()}
    assert cpu_receipt == sha(case / 'cpu_run.json')
    # Verify copied bytes too, including cross-filesystem ROOT -> DATA moves.
    for f in files:
        quiet()
        assert sha(Path(f['archive'])) == f['sha256']
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), case=ident, reason=reason,
        scope='Preserved historical GPU attempt before changed-source qualification; original CPU references and inputs untouched',
        path_relocation=mapping, files=files, read_only_external_links=external_links, original_input_sha256=original,
        original_cpu_receipt_sha256=cpu_receipt, cache='ABSENT; prior E41/PIN consumers completed')
    (target / 'ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    view = case / 'attempts' / reason
    view.parent.mkdir(exist_ok=True)
    if view.resolve() != target:
        view.symlink_to(target, target_is_directory=True)
    print(ident, len(files), 'SHA-verified files preserved at', target, flush=True)
