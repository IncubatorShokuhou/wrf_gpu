"""Explicit phy_prep caller regime inputs for the changed FLG boundary.

Real structured retained profiles, explicit constructed sigma/height weights;
eight additional latitude/season stresses are labelled, not forecast replay.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from parse import operands


def main(original,directory,stage):
    directory.mkdir(exist_ok=True,parents=True)
    if stage=='prepare':
        data=operands(original);fields=data['fields'];nz=fields.shape[1]-1
        cloudy=np.flatnonzero(np.any(fields[:,:nz,6]+fields[:,:nz,8]>0.,axis=1))
        assert len(cloudy)>=4
        index=np.r_[np.arange(len(fields)),np.resize(cloudy,8)]
        nc=len(index)
        data={k:v[index].copy() if k in ('fields','sfc','julian','tau','declin','solcon','julday') else v for k,v in data.items()}
        data['sfc'][-8:,4]=[45.,-45.,75.,-75.,45.,-45.,75.,-75.]
        data['julday'][-8:]=[15]*4+[195]*4
        data['fields'][:,-1,:]=0.  # Upper mass padding, explicit native allocation value.
        fields=data['fields']
        # P8W is fully defined on interfaces; restore that full operand.
        fields[:,:,12]=operands(original)['fields'][index,:,12]
        heights=np.concatenate((np.zeros((nc,1),np.float32),np.cumsum(fields[:,:nz,13],axis=1,dtype=np.float32)),axis=1)
        with (directory/'phy_input.bin').open('wb') as stream:
            np.asarray([nc,nz],np.int32).tofile(stream)
            for value in (fields[:,:nz,0],fields[:,:nz,1],heights,np.full(nz+1,.5,np.float32),np.full(nz+1,.5,np.float32)):
                value.ravel(order='F').tofile(stream)
        np.savez_compressed(directory/'operands.npz',**data)
        (directory/'scope.json').write_text(json.dumps(dict(real_columns=40,latitude_season_stresses=8,
            constructed=['linear sigma midpoint weights .5/.5','explicit heights from retained DZ cumsum','8latitude/season stress copies'],
            physical_caller=['unchanged pristine phy_prep T8W interpolation/extrapolation','allocated upper mass QV0'],
            scope='routine caller qualification, not original forecast replay'),indent=2)+'\n')
    else:
        with np.load(directory/'operands.npz',allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        fields=data['fields'];nc,n1,_=fields.shape;nz=n1-1
        t8w=np.fromfile(directory/'phy_output.bin',np.float32).reshape(nc,n1,order='F')
        assert np.isfinite(t8w).all();fields[:,:,4]=t8w
        with (directory/'input.bin').open('wb') as stream:
            np.asarray([nc,nz],np.int32).tofile(stream);data['scalars'].tofile(stream);data['year'].tofile(stream)
            for key in ('znw','znu','julian','tau','declin','solcon','julday','sfc','fields'):
                data[key].ravel(order='F').tofile(stream)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('directory',type=Path)
    p.add_argument('stage',choices=('prepare','finish'));a=p.parse_args();main(a.original,a.directory,a.stage)
