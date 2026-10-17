"""Private, alias-free stage for the original CPU-WRF c08 rerun with do_radar_ref=1 (rv-t-classd F4).

usage: python proofs/t_classd/stage_c08_radar_cpu.py <new_stage_dir>
* Inputs (namelist.input, wrfinput_d01, wrfbdy_d01) are COPIED as regular files from the sealed
  v-prep c08 reference and verified against its cpu_run.json input_sha256 before any edit.
* The radar setting is inserted into the private namelist copy only.
* Symlinks are created only for read-only tables inside the pristine WRF tree, never for
  namelist*/wrfout*/wrfrst*/rsl*/wrfinput*/wrfbdy*/*.exe names; every write refuses symlinks.
* wrf.exe is a private copy of the pristine executable with only the two diagnostic OPEN path
  literals redirected into the stage (FINDINGS E104).
Writes STAGE.json (original and used SHA-256 of every staged input) into the stage.
"""
import hashlib, json, os, re, shutil, sys
from pathlib import Path

PRISTINE = Path('<USER_HOME>/src/wrf_pristine/WRF')
REFERENCE = Path('<USER_HOME>/wrf_gpu2_lanes/v-prep/c08')
FORBIDDEN = re.compile(r'^(namelist|wrfout|wrfrst|rsl\.|wrfinput|wrfbdy|wrfout|.*\.exe$)')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_private(path: Path, data: bytes):
    if path.is_symlink() or path.exists():
        raise SystemExit(f'refusing to write existing/aliased {path}')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)


def main(stage: Path):
    stage.mkdir(parents=True, exist_ok=False)
    sealed = json.loads((REFERENCE / 'cpu_run.json').read_text())['input_sha256']
    record = dict(reference=str(REFERENCE), sealed_input_sha256=sealed, staged={}, links={})
    for name in ('namelist.input', 'wrfinput_d01', 'wrfbdy_d01'):
        data = (REFERENCE / 'input' / name).read_bytes()          # follows the owner's read-only alias
        if hashlib.sha256(data).hexdigest() != sealed[name]:
            raise SystemExit(f'{name} differs from the sealed original')
        if name == 'namelist.input':
            text = data.decode()
            assert 'do_radar_ref' not in text
            used = re.sub(r'(&physics\s*\n)', r'\1 do_radar_ref = 1,\n', text, count=1, flags=re.I)
            assert used != text and used.replace(' do_radar_ref = 1,\n', '', 1) == text
            data = used.encode()
        write_private(stage / name, data)
        record['staged'][name] = dict(original_sha256=sealed[name], used_sha256=hashlib.sha256(data).hexdigest(),
                                      regular_file=not (stage / name).is_symlink())
    for link in sorted((REFERENCE / 'cpu').iterdir()):
        if not link.is_symlink():
            continue
        target = Path(os.readlink(link))
        if FORBIDDEN.match(link.name) or not str(target).startswith(str(PRISTINE) + '/'):
            continue
        (stage / link.name).symlink_to(target)
        record['links'][link.name] = str(target)
    exe = bytearray((PRISTINE / 'main/wrf.exe').read_bytes())
    original = hashlib.sha256(exe).hexdigest()
    patches = []
    for old, suffix in ((b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_quarter_ss_center_dump.txt', 'd0'),
                        (b'<DATA_ROOT>/wrf_gpu2/wrf_truth/em_grav2d_x_touchdown_dump.txt', 'd1')):
        new = str(stage.parent / suffix).encode()
        assert exe.count(old) == 1 and len(new) < len(old)
        at = exe.index(old)
        exe[at:at + len(old)] = new + b'\0' * (len(old) - len(new))
        patches.append(dict(offset=at, old=old.decode(), new=new.decode()))
    write_private(stage / 'wrf.exe', bytes(exe))
    (stage / 'wrf.exe').chmod(0o755)
    record['executable'] = dict(original=str(PRISTINE / 'main/wrf.exe'), original_sha256=original,
                                private_sha256=sha(stage / 'wrf.exe'), patches=patches)
    write_private(stage / 'STAGE.json', (json.dumps(record, indent=1) + '\n').encode())
    print(json.dumps({k: v for k, v in record.items() if k != 'links'}, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
