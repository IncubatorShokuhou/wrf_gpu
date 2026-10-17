import netCDF4, numpy as np, json
V='<USER_HOME>/wrf_gpu2_lanes/v-prep/c40'
dirs={'cpu':V+'/cpu','f68':V+'/attempts/before_YSU_f68ca7296/gpu','04cb':V+'/gpu'}
fields=('T2','TSK','HFX','LH','SWDOWN','GLW','GRDFLX','PBLH','U10','Q2','UST')
out={}
for h in range(0,25):
    fn='wrfout_d01_2026-10-13_%02d:00:00'%h if h<24 else 'wrfout_d01_2026-10-14_00:00:00'
    try: ds={k:netCDF4.Dataset(f'{d}/{fn}') for k,d in dirs.items()}
    except Exception as e: continue
    land=np.asarray(ds['cpu']['XLAND'][0])<1.5
    row={}
    for f in fields:
        if f not in ds['cpu'].variables: continue
        c=np.asarray(ds['cpu'][f][0],np.float64)
        for k in ('f68','04cb'):
            if f not in ds[k].variables: continue
            g=np.asarray(ds[k][f][0],np.float64)
            row[f'{f}_{k}_land']=float(np.sqrt(np.mean((g-c)[land]**2)))
            row[f'{f}_{k}_water']=float(np.sqrt(np.mean((g-c)[~land]**2))) if (~land).any() else None
            row[f'{f}_{k}_landbias']=float(np.mean((g-c)[land]))
    out[h]=row
    for d in ds.values(): d.close()
json.dump(out,open('c40_localize.json','w'),indent=1)
print('land frac', float(land.mean()))
for h,row in out.items():
    print(h, ' '.join(f"{f}:{row.get(f+'_f68_land',float('nan')):.2f}/{row.get(f+'_04cb_land',float('nan')):.2f}(b{row.get(f+'_04cb_landbias',float('nan')):+.1f})" for f in ('T2','TSK','HFX','LH','SWDOWN','GLW','GRDFLX')))
