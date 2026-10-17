"""Verify all portable numeric references without JAX or external WRF inputs."""
import hashlib,json
from pathlib import Path


def verify(root):
    root=Path(root);index=json.loads((root/'FIXTURE_INDEX.json').read_text())
    listed={r['path'] for r in index['files']}
    auxiliary=index.get('auxiliary_files',[])
    assert listed=={p.name for p in root.glob('*oracle.json')}
    sums=dict(line.split('  ',1)[::-1] for line in (root/'SHA256SUMS').read_text().splitlines())
    assert set(sums)==listed|{r['path'] for r in auxiliary}
    for row in index['files']+auxiliary:
        path=root/row['path'];data=path.read_bytes()
        assert len(data)==row['bytes']
        assert hashlib.sha256(data).hexdigest()==row['sha256']==sums[row['path']]
        fixture=json.loads(data)
        if row in auxiliary:
            assert row['kind']=='independent_input_ulp_bounds'
            assert set(fixture['bounds'])=={'snow','irrigation','skin'}
            assert fixture['method'].startswith('independent pristine Fortran')
        else:
            assert fixture.get('source') and (fixture.get('flags') or fixture.get('compiler'))
        assert row['source_commit']==index['source_commit']
    return len(listed)


if __name__=='__main__':
    print(json.dumps({'ok':True,'fixtures':verify(Path(__file__).parent)}))
