"""Stage WN3 variants with the product namelist parser and no optional dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from gpuwrf.io.namelist_parser import parse_namelist


def prepare(root, base):
    root.mkdir(parents=True, exist_ok=True)
    if (root/"input_manifest.json").exists():
        raise ValueError("refuse existing GPU arm directory")
    manifest = {}
    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        real = path.resolve()
        if str(real).startswith(("<DATA_ROOT>/alisios/", "<DATA_ROOT>/server/work/src/alisios/")):
            raise ValueError("protected source")
        manifest[path.name] = dict(path=str(real), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    original = (base/"namelist.input").read_text()
    if re.search(r"&gpuwrf\b", original, re.I):
        raise ValueError("base already has a gpuwrf group; refuse ambiguous override")
    for name, diag, bl in (("off", 0, 1), ("on", 1, 1), ("bl0", 1, 0)):
        case = root/name/"input"
        case.mkdir(parents=True)
        for path in base.iterdir():
            if path.is_file() and path.name != "namelist.input":
                (case/path.name).symlink_to(path.resolve())
        text, count = re.subn(r"(?im)^\s*icloud_bl\s*=.*$", f" icloud_bl = {bl},", original)
        if not count:
            text, count = re.subn(r"(?im)^&physics\s*$", f"&physics\n icloud_bl = {bl},", text)
        if count != 1:
            raise ValueError("ambiguous physics group/icloud_bl")
        text += f"\n&gpuwrf\n rad_diag = {diag},\n/\n"
        path = case/"namelist.input"
        path.write_text(text)
        parsed = parse_namelist(path)
        assert parsed["physics"]["icloud_bl"] == bl
        assert parsed["gpuwrf"]["rad_diag"] == diag
        assert parsed["domains"]["max_dom"] == 3
        manifest[name] = dict(rad_diag=diag, icloud_bl=bl, namelist_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/"input_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        with tempfile.TemporaryDirectory(dir=args.root) as directory:
            prepare(Path(directory)/"stage", args.base)
        print("staging dry test PASS: all input hashes, 3 variants parsed, 0 protected reads")
    else:
        prepare(args.root, args.base)
