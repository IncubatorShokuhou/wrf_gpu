"""Extract WRF Goddard constants without executing/importing model code.

Usage: python extract_tables.py /path/to/WRF src/gpuwrf/physics/goddard_tables.npz
Only host NumPy data is emitted. Default REAL DATA literals are rounded to
REAL4 before assignment to the explicit REAL8 solver tables, as in WRF.
"""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np


def values(text, dtype):
    result = []
    for word in text.replace('&', '').split(','):
        word = word.strip()
        if not word:
            continue
        count = 1
        if '*' in word:
            rep, word = word.split('*')
            count = int(rep)
        explicit_double = '_fp_kind' in word or 'd' in word.lower()
        word = word.lower().replace('_fp_kind', '').replace('d', 'e')
        value = float(word) if explicit_double else float(np.float32(word))
        result.extend([value] * count)
    return np.asarray(result, dtype=dtype)


def extract(wrf, out):
    src = wrf / 'phys/module_ra_goddard.F'
    raw = src.read_bytes()
    text = '\n'.join(line.split('!')[0].lower() for line in raw.decode().splitlines())
    result = {}
    pts = re.search(r'pts_re_visir\(mxpts_re_visir\).*?\(/(.*?)/\)', text, re.S)
    result['radius'] = values(pts[1], np.float32)
    climate = np.full((33, 10, 6), np.nan, np.float32)
    for match in re.finditer(r'data\s*\(\(mcdat\(ilev,(\d+),ifld\),ifld=1,6\),ilev=(\d+),(\d+)\)/(.*?)/', text, re.S):
        profile, lo, hi = map(int, match.group(1, 2, 3))
        climate[lo-1:hi, profile-1, :] = values(match[4], np.float32).reshape(hi-lo+1, 6)
    assert np.isfinite(climate).all()
    result['climate'] = climate
    lw = text.split('subroutine lwrad (',1)[1].split('end subroutine lwrad',1)[0]
    for name,shape in {'cb':(6,10),'xkw':(9,),'xke':(9,),'aw':(9,),
                       'bw':(9,),'pm':(9,),'mw':(9,),'fkw':(6,9),
                       'gkw':(6,3),'ne':(9,)}.items():
        match = re.search(r'\bdata\s+'+name+r'\s*/(.*?)/',lw,re.S)
        assert match, name
        dtype=np.int32 if name in ('mw','ne') else np.float64
        result['lw_'+name] = values(match[1], dtype).reshape(shape,order='F')
    for name in ('o1','o2','o3'):
        table = np.full((26,21),np.nan,np.float64)
        for match in re.finditer(r'data\s*\(\('+name+r'\(ip,iw\),iw=1,21\),\s*ip=\s*(\d+),\s*(\d+)\)/(.*?)/',lw,re.S):
            lo,hi = map(int,match.group(1,2))
            table[lo-1:hi,:] = values(match[3],np.float64).reshape(hi-lo+1,21)
        assert np.isfinite(table).all(), name
        result['lw_'+name] = table
    sw=text.split('subroutine swrad (',1)[1].split('end subroutine swrad',1)[0]
    for name,shape in {'cah':(43,37),'coa':(62,101)}.items():
        table=np.full(shape,np.nan,np.float64)
        for match in re.finditer(r'data\s*\(\('+name+r'\(i,j\),i=1,\s*'+str(shape[0])+r'\),j=\s*(\d+),\s*(\d+)\)/(.*?)/',sw,re.S):
            lo,hi=map(int,match.group(1,2))
            table[:,lo-1:hi]=values(match[3],np.float64).reshape(shape[0],hi-lo+1,order='F')
        assert np.isfinite(table).all(),name
        result['sw_'+name]=table
    ratio=re.search(r'data\s*\(\(ratio_lut\(i,j\),i=1,10\),j=1,10\)/(.*?)/',sw,re.S)
    result['sw_ratio']=values(ratio[1],np.float64).reshape((10,10),order='F')
    for routine,shapes in [('sw_uvpar',{'hk':(8,),'zk':(8,),'wk':(8,),'ry':(8,)}),
                           ('sw_ir',{'hk':(3,10),'xk':(10,),'ry':(3,)})]:
        body=text.split('subroutine '+routine+' (',1)[1].split('end subroutine '+routine,1)[0]
        for name,shape in shapes.items():
            match=re.search(r'\bdata\s+'+name+r'\s*/(.*?)/',body,re.S)
            assert match,name
            result[routine+'_'+name]=values(match[1],np.float64).reshape(shape,order='F')
    binary = wrf / 'run/BROADBAND_CLOUD_GODDARD.bin'
    # WRF configure.wrf BYTESWAPIO uses big-endian direct records. The shipped
    # file has 24 trailing bytes which read_lut_nc never reads (three records).
    data = np.fromfile(binary, dtype='>f4', count=3*109*6*11*2).astype(np.float32)
    assert data.size == 3*109*6*11*2
    # read_lut_nc assigns ns=2,3,1 for file species=1,2,3: cloud/rain/ice
    # in the file becomes ice/cloud/rain in the physics arrays.
    for index, name in enumerate(('extinction', 'ssa', 'asymmetry')):
        array = data[index*109*6*11*2:(index+1)*109*6*11*2].reshape((109, 6, 11, 2), order='F')
        result[name] = array[:, [2, 0, 1, 3, 4, 5], :, :]
    assert len(result['radius']) == 109
    assert all(np.isfinite(x).all() for x in result.values())
    np.savez_compressed(out, **result)
    digest=hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_name('goddard_table_digest.py').write_text(
        '"""Generated source identity for the WRF Goddard runtime asset (AOT key)."""\n'
        f'TABLES_SHA256 = {digest!r}\n')
    manifest = {
        'schema': 'wrf-goddard-tables-v1',
        'source_sha256': hashlib.sha256(raw).hexdigest(),
        'cloud_lut_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'asset_sha256': digest,
        'shapes': {k: list(v.shape) for k, v in result.items()},
        'dtypes': {k:str(v.dtype) for k,v in result.items()},
        'precision': 'WRF REAL4 wrapper/LUT/climatology; fp_kind REAL8 solver tables with default REAL4 literals promoted; mw/ne INTEGER4',
        'species': ['ice', 'cloud', 'rain', 'snow', 'graupel', 'hail'],
        'lut_bytes_consumed': int(data.nbytes),
        'lut_trailing_bytes_ignored_by_wrf': binary.stat().st_size-int(data.nbytes),
    }
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
