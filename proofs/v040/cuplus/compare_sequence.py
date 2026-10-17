"""Qualify resident cadence/held columns against repeated original CPS calls."""
from __future__ import annotations
import argparse,json,os,re,sys,time
from pathlib import Path
import numpy as np
from compare_kfcup_parcel import parse_text
from literal_kfcup import intrinsic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--scheme',type=int,choices=(7,10,11),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('JAX_PLATFORMS')=='cpu'
    assert not Path('/tmp/wrf_gpu2_quiet').exists()
    sys.path.insert(0,str(args.source/'src'))
    import jax
    import jax.numpy as jnp
    from gpuwrf.kernels.phys_cuplus_driver import column_step,finish_column,initial_column
    assert all(d.platform=='cpu' for d in jax.devices())
    oracle=json.loads(args.oracle.read_text());records=[];calls={}
    for row in oracle['records']:
        if row.get('mode','plain')!='plain':continue
        assert row['returncode']==0,row
        split=re.split(r'^SEQUENCE_STEP=\s*(\d+)\s*$',Path(row['stdout']).read_text(),flags=re.M)
        assert len(split)==21,len(split)
        prefix,_=parse_text(split[0]);first,scalars=parse_text(split[2]);n=len(first.get('T',prefix.get('IN_T_PHY',[])))
        assert n>0
        scheme=args.scheme
        if scheme in (10,11):
            names=('T','QV','P','DZ','RHO','W0AVG','U','V')+ (('TH','Z','ZW','PI') if scheme==10 else ('PI',))
            arrays=tuple(np.asarray(first[key],np.float32) for key in names)
            w0=np.asarray(prefix['SEQ_INPUT_W0AVG'],np.float32)
            _,pre_scalar=parse_text(split[0])
            w=np.asarray(prefix['SEQ_INPUT_W']+[pre_scalar['SEQ_TOP_W']],np.float32)
            dt=scalars['DT'];dx=scalars['DX']
            if scheme==11:ctrl=np.asarray([dt,dx,1000.,150.,.3,-.1,0.,0.],np.float32)
            else:ctrl=np.asarray([dt,dx,1500.,6.,float(row['controls'][0]),1e-4,1.,0.],np.float32)
            pbl=9;cudt=0.
        else:
            names=('T_PHY','QV','P','P8W','PI_PHY','U_PHY','V_PHY','Z','Z_AT_W','TKE_PBL','TH','QC','QI','CLDFRA','DZ8W')
            arrays=tuple(np.asarray(prefix['IN_'+key],np.float32) for key in names)
            if len(arrays[3])==n:
                # Exact controlled WK driver equations, not a real-profile
                # interpolation: these two interface values were omitted from
                # the old dump. Full real bank dumps retain every interface.
                parts=list(arrays)
                top=np.float32(np.float32(parts[2][-1]*parts[2][-1])/parts[3][-1])
                parts[3]=np.concatenate((parts[3],[top])).astype(np.float32)
                top=np.float32(parts[8][-1]+parts[14][-1])
                parts[8]=np.concatenate((parts[8],[top])).astype(np.float32)
                arrays=tuple(parts)
            dt=54.;cudt=4.5;pbl=row['pbl'];w0=np.zeros(n,np.float32);w=np.zeros(n+1,np.float32)
            ctrl=np.asarray([dt,9000.,cudt,1.,100800. if row['case']==3 else 100000.,0.,1500.,6.,150.,1e-4,.3,1.,1.,arrays[0][0],.3],np.float32)
        key=(scheme,pbl)
        if key not in calls:
            def call(a,w,w0,held,c,step,_scheme=scheme,_pbl=pbl,_cudt=cudt):
                if _scheme==7:c=c.at[3].set(step.astype(jnp.float32))
                return column_step(a,w,w0,held,c,step,scheme=_scheme,stepcu=5,cudt=_cudt,pbl=_pbl,
                    num_bins=5,th_bin_size=.1,r_bin_size=1e-4,min_deep_freq=.1,min_shallow_freq=.001,interpret=True)
            calls[key]=jax.jit(call)
        held=initial_column(scheme,n)
        for index in range(1,11):
            assert int(split[2*index-1])==index
            target_fields,target_scalars=parse_text(split[2*index])
            assert all(np.isfinite(v).all() for v in target_fields.values())
            assert all(np.isfinite(v) for v in target_scalars.values())
            if scheme in (10,11) and index in (2,5):held=dict(held,NCA=jnp.float32(2*dt if index==2 else 0.))
            started=time.monotonic()
            w0,actual=jax.block_until_ready(calls[key](arrays,w,w0,held,ctrl,jnp.int32(index)))
            diff={};compared=0
            for name,value in actual.items():
                alias={'DUDT':'RUCUTEN','DVDT':'RVCUTEN','ISHALL':'SHALL'}.get(name,name)
                truth=target_fields.get(alias,target_scalars.get(alias))
                if truth is None:continue
                value=np.asarray(value);truth=np.asarray(truth,np.float32)
                if value.ndim:value=value[:len(truth)]
                assert value.shape==truth.shape,(name,value.shape,truth.shape)
                assert np.isfinite(value).all(),(row['case'],index,name)
                error=float(np.max(np.abs(value-truth)));compared+=value.size
                exact=name in ('NCA','CUTOP','CUBOT','CUPFLAG','ISHALL')
                limit=0. if exact else 128*np.finfo(np.float32).eps*float(np.max(np.abs(truth)))
                if error>limit:diff[name]=dict(max_abs=error,limit=float(limit))
            original_w0=target_fields.get('W0AVG')
            if original_w0 is not None:
                truth=np.asarray(original_w0,np.float32)
                error=float(np.max(np.abs(np.asarray(w0)-truth)))
                limit=128*np.finfo(np.float32).eps*float(np.max(np.abs(truth)))
                if error>limit:diff['W0AVG']=dict(max_abs=error,limit=float(limit))
            item=dict(case=row['case'],pbl=pbl,step=index,compared=compared,violations=diff,elapsed_s=time.monotonic()-started)
            records.append(item)
            args.output.write_text(json.dumps(dict(source=str(args.source),oracle=str(args.oracle),scheme=scheme,
                gate='finitefirst;128epsREAL32peak;cadence/timers/indices EXACT; compilerexcess disclosed',records=records),indent=2)+'\n')
            print(json.dumps(item),flush=True)
            held,_=finish_column(actual,dt,scheme=scheme)


if __name__=='__main__':main()
