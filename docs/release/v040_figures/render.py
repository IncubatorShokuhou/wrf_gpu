"""Reproduce the v0.4.0 documentation figures from the sealed CSV scores.

CPU only. Run: JAX_PLATFORMS=cpu MPLCONFIGDIR=/your/scratch python render.py
No model execution, scientific-file write, tolerance change or archive read.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
def rows(name):
 with(OUT/name).open()as f:return list(csv.DictReader(f))
def save(fig,name):
 for ext in ('png','svg'):
  p=OUT/(name+'.'+ext);fig.savefig(p,dpi=140,bbox_inches='tight',facecolor='white')
  if ext=='svg':p.write_text('\n'.join(line.rstrip()for line in p.read_text().splitlines())+'\n')
  if p.stat().st_size>=1000000:raise ValueError(f'{p.name} exceeds1MiB')
 plt.close(fig)

r=rows('composites.csv');cases=list(dict.fromkeys(x['case']for x in r))
forecast=['T2','PSFC','U10','V10','RAINNC','T','U','V','W','QVAPOR'];static=sorted({x['field']for x in r if x['metric']=='max_abs'})
assert len(cases)==66 and len(static)==52
fig,(left,right)=plt.subplots(1,2,figsize=(18,15),sharey=True,gridspec_kw={'width_ratios':[10,34],'wspace':.07})
for ax,fields,kind in [(left,forecast,'forecast'),(right,static,'static')]:
 a=np.full((len(cases),len(fields)),np.nan)
 for x in r:
  if x['field']not in fields or x['status']!='MEASURED':continue
  if kind=='static':assert float(x['value'])<=float(x['limit'])
  a[cases.index(x['case']),fields.index(x['field'])]=float(x['ratio'])
 assert np.isfinite(a).all()
 cm=plt.get_cmap('YlGnBu_r').copy();cm.set_bad('#d8d8d8')
 im=ax.imshow(np.ma.masked_invalid(a),aspect='auto',cmap=cm,vmin=0,vmax=1,interpolation='nearest')
 if kind=='forecast':
  ax.set_xticks(range(len(fields)),fields,rotation=90,fontsize=9);ax.set_yticks(range(len(cases)),[x+(' (6h)'if x=='c45'else'')for x in cases],fontsize=8)
  cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.09,fraction=.02);cb.set_label('Worst hourly/domain RMSE / frozen RMSE limit (1 = limit)')
 else:
  ax.set_xticks(range(len(fields)),fields,rotation=90,fontsize=7)
  cb=fig.colorbar(im,ax=ax,orientation='horizontal',pad=.09,fraction=.02);cb.set_label('Max absolute difference / frozen limit;48 exact-zero rules shown as 0 only on exact pass')
 ax.add_patch(Rectangle((-.5,cases.index('c45')-.5),len(fields),1,fill=False,edgecolor='#e69f00',lw=1.5))
 ax.set_title('10 forecast RMSE checks'if kind=='forecast'else'52 grid/static max-absolute-error checks',fontsize=11)
fig.suptitle('v0.4.0: 66 configurations × 62 bounded fields against original CPU-WRF\n65 full 24 h scopes + c45 accepted 6 h; all 66 scientific scopes PASS',fontsize=15,y=.94)
fig.text(.02,.025,'Grey/hatching: no model result or no applicable bound. Static invariants are separate from forecast-error statistics. REPORT_ONLY fields have no invented bound. E41/F2 remain separate.',fontsize=9)
save(fig,'composite_matrix')

r=rows('wn3_bands.csv');fields=list(dict.fromkeys(x['field']for x in r));fig,axes=plt.subplots(5,2,figsize=(14,14));colors={'d01':'#0072b2','d02':'#d55e00','d03':'#009e73'}
for ax,field in zip(axes.flat,fields):
 limit=0;unit=''
 for dom,color in colors.items():
  v=[x for x in r if x['field']==field and x['domain']==dom];t=[float(x['lead_h'])for x in v];rm=[float(x['rmse'])for x in v];p95=[float(x['p95_abs'])for x in v];unit=v[0]['unit'];limit=max(limit,max(float(x['rmse_limit'])for x in v))
  ax.fill_between(t,0,p95,color=color,alpha=.12);ax.plot(t,rm,color=color,lw=1.5,label=dom)
 ax.axhline(limit,color='#555',ls=':',lw=1,label='RMSE limit');ax.set_title(field+(' (perturbation theta)'if field=='T'else''));ax.set_ylabel(unit);ax.set_xlim(0,24);ax.set_ylim(bottom=0);ax.set_xticks([0,6,12,18,24]);ax.grid(alpha=.18)
axes[0,0].legend(ncol=4,fontsize=8);axes[-1,0].set_xlabel('Forecast lead (hours)');axes[-1,1].set_xlabel('Forecast lead (hours)')
fig.suptitle('WN3-0227: full 24 h GPU-versus-original-CPU differences, all three domains\nSolid: full-field RMSE; shading: 0–95th percentile absolute error; dotted: frozen RMSE limit',fontsize=13)
fig.tight_layout(rect=(0,.025,1,.955));fig.text(.02,.006,'Spatial percentiles are not confidence intervals or pointwise gates; the remaining 5% of cells can have larger errors. Source 353a85f8a; 75 paired frames.',fontsize=9);save(fig,'wn3_deviation_bands')

r=rows('c06_rain.csv');fig,ax=plt.subplots(figsize=(10,5.3));peaks={}
for phase,color,label in [('before','#d55e00','Before ROOT1–3 (04cb09f816)'),('after','#0072b2','After ROOT1–3 (db78f9bd8c)')]:
 v=[x for x in r if x['phase']==phase];y=[float(x['rmse_mm'])for x in v];peaks[phase]=max(y);ax.plot([float(x['lead_h'])for x in v],y,color=color,lw=2,label=label)
ax.axhline(float(r[0]['limit_mm']),color='#555',ls='--',lw=1.2,label='Frozen D6 RMSE limit');ax.set_xlim(0,24);ax.set_ylim(bottom=0);ax.set_xlabel('Forecast lead (hours)');ax.set_ylabel('RAINNC RMSE against the same original CPU-WRF (mm)');ax.set_title('Validation catches a real coupled-forecast bug: c06 Tiedtke');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
ax.text(.98,.2,f"Worst-hour RMSE: {peaks['before']:.3f} → {peaks['after']:.4f} mm\n{peaks['before']/peaks['after']:.1f}× smaller error in this case",transform=ax.transAxes,ha='right',fontsize=12,bbox={'facecolor':'white','edgecolor':'#ccc','alpha':.95})
fig.tight_layout();fig.text(.02,.005,'Full 25-frame trajectories; identical input hashes and original CPU receipt. This is not GPU-vs-GPU fidelity or a universal skill claim.',fontsize=8);save(fig,'c06_tiedtke_rain')
p=OUT/'PROVENANCE.json';d=json.loads(p.read_text());d['matplotlib_version']=matplotlib.__version__;d['numpy_version']=np.__version__;d['plot_data_sha256']={q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in OUT.glob('*.csv')};d['renderer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();d['figures']={q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in OUT.iterdir()if q.suffix in('.png','.svg')};p.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d['figures'],indent=2))
