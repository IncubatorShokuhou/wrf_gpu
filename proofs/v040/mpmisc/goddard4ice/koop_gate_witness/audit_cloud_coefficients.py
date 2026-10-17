"""Read original initialized REAL cloud coefficients with GDB, no JAX."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import numpy as np

CACHE=Path('<USER_HOME>/wrf_gpu2_lanes/s2-mpmisc/thompson38/H38-03')


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--asset',type=Path,default=Path('data/fixtures/thompson-aero-tables-v1.npz'));a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);a.output=a.output.resolve()
    table=CACHE/'qr_acr_qg_mp38V1.dat';before=table.stat()
    source=CACHE/'module_mp_thompson.F';source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_sha=='fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'
    # Fortran first subscript varies fastest, exactly its in-memory layout.
    code='import gdb,struct,json; '
    code+='values={name:[float(gdb.parse_and_eval(name+"("+str(i)+","+str(j)+")")) for j in range(1,16) for i in range(1,6)] for name in ("cce","ccg")}; '
    code+='values.update({name:[float(gdb.parse_and_eval(name+"("+str(i)+")")) for i in range(1,16)] for name in ("ocg1","ocg2")}); '
    code+='types={name:{"type":str(gdb.parse_and_eval(name+suffix).type),"bytes":int(gdb.parse_and_eval(name+suffix).type.sizeof)} for name,suffix in (("cce","(1,1)"),("ccg","(1,1)"),("ocg1","(1)"),("ocg2","(1)"))}; '
    code+='open('+repr(str(a.output/'original_types.json'))+',"w").write(json.dumps(types,indent=2)); '
    code+='open('+repr(str(a.output/'original_values.json'))+',"w").write(json.dumps(values,indent=2))'
    commands=['set pagination off','set confirm off','break module_mp_thompson.F:1648','run 4','set language fortran','python '+code,'quit']
    script=a.output/'cloud_coefficients.gdb';script.write_text('\n'.join(commands)+'\n')
    process=subprocess.run(['gdb','-q','-batch','-x',str(script),str(CACHE/'oracle_debug')],cwd=CACHE,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False,timeout=180)
    (a.output/'gdb.log').write_text(process.stdout)
    assert process.returncode==0,process.stdout[-2000:]
    values=json.loads((a.output/'original_values.json').read_text())
    types=json.loads((a.output/'original_types.json').read_text())
    assert all(row['bytes']==4 for row in types.values()),types
    original={name:np.asarray(value,np.float32).reshape((5,15),order='F') if name in ('cce','ccg') else np.asarray(value,np.float32) for name,value in values.items()}
    payload={};receipt={'source_sha256':source_sha,'module_object_sha256':hashlib.sha256((CACHE/'thompson_debug.o').read_bytes()).hexdigest(),'asset_sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),'original_dtype':'REAL4 / float32','original_gdb_types':types,'original_build':'cached pristine thompson_debug.o O0 -g, compiler-defined zero init/no FMA; initialized via production both aerosol/hail flags','JAX_imported':False,'arrays':{}}
    with np.load(a.asset,allow_pickle=False) as asset:
        for name,want in original.items():
            raw=asset[name];got=raw.astype(np.float32)
            assert got.shape==want.shape,(name,got.shape,want.shape)
            different=got.view(np.uint32)!=want.view(np.uint32)
            rows=[]
            for index in np.argwhere(different):
                index=tuple(index);ref=want[index];value=got[index]
                rows.append({'fortran_indices_1based':[int(i)+1 for i in index],'original_REAL':float(ref),'asset_REAL':float(value),'original_u32_hex':format(int(np.asarray(ref).view(np.uint32)),'08x'),'asset_u32_hex':format(int(np.asarray(value).view(np.uint32)),'08x'),'integer_bit_distance':abs(int(np.asarray(ref).view(np.uint32))-int(np.asarray(value).view(np.uint32)))})
            receipt['arrays'][name]={'shape':list(want.shape),'asset_dtype':str(raw.dtype),'native_dtype':'float32','values':want.size,'BYTE_count':int(want.size-np.count_nonzero(different)),'different_count':int(np.count_nonzero(different)),'differences':rows}
            payload[name]=want
    after=table.stat();receipt['cached_table_unchanged']=(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);assert receipt['cached_table_unchanged']
    receipt['total_values']=sum(v['values'] for v in receipt['arrays'].values());receipt['total_BYTE']=sum(v['BYTE_count'] for v in receipt['arrays'].values())
    np.savez_compressed(a.output/'original_cloud_coefficients.npz',**payload)
    receipt['original_coefficients_sha256']=hashlib.sha256((a.output/'original_cloud_coefficients.npz').read_bytes()).hexdigest()
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
