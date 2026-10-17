"""Every lane reference is present and authentic in a clean git archive."""
import hashlib,importlib.util,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def test_all_original_fixtures_load_from_clean_archive(tmp_path):
    data=subprocess.check_output(['git','-C',str(ROOT),'archive','HEAD','proofs/u2_land'],timeout=30)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:archive.extractall(tmp_path,filter='data')
    source=tmp_path/'proofs/u2_land/verify_fixtures.py'
    spec=importlib.util.spec_from_file_location('archived_land_fixtures',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.verify(source.parent)==20
    for directory in ('receipts','critic_fixes'):
        receipts=source.parent/directory
        manifest=json.loads((receipts/'MANIFEST.json').read_text())
        if directory=='receipts':assert len(manifest)==53
        for name,row in manifest.items():
            data=(receipts/name).read_bytes()
            assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
            # Nonfinite JSON numbers cannot be evidence for a finite physics gate.
            json.loads(data,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
