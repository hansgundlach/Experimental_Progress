import os,sys,json,hashlib,shutil,contextlib,io
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(ROOT/'cache');os.environ['MPLBACKEND']='Agg';sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'source_snapshot'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import constants as k, utils as u, sensitivity_analysis as s
from graphing_utils import TrainingCurveAnalyzer
from experiments_config import experiments_config
out=ROOT/'results';out.mkdir(exist_ok=True)
AL,aL=k.A_L,k.alpha_L; C=float(u.ytc(2023))
def stats():
 d=u.compute_ceg_statistics(2012,2023)
 return dict(total=d['total_cumulative_growth'],architecture=d['total_transformer_growth'],scale_dependent=d['scale_dependent_growth'],chinchilla=d['total_chinchilla_growth'],share=100*np.log(d['scale_dependent_growth'])/np.log(d['total_cumulative_growth']))
base=stats();rows=[]
for anchor in [None,1e16,1e17]:
 for dev in np.linspace(-.25,.25,101):
  a=s.set_exponent_deviation(float(dev));u.A_L=AL if anchor is None else AL*anchor**(aL-a)
  r=dict(method='original_fixed_coefficient' if anchor is None else f'anchored_{anchor:.0e}',deviation=dev,alpha_L=a,**stats()); rows.append(r)
  if anchor is None: assert np.isclose(r['share'],s.scale_dependent_share(2012,2023))
s.reset_alpha_L();u.A_L=AL
pd.DataFrame(rows).to_csv(out/'exponent_gap.csv',index=False)
rows2=[]
for c in np.logspace(18,24,61):
 sd=float(u.lstm_to_2017_transformer_multiplier(c));ch=float(k.chinchilla_rebalancing_multiplier(c))
 rows2.append(dict(compute=c,architecture=sd/ch,scale_dependent=sd,total=sd*base['total']/base['scale_dependent']))
pd.DataFrame(rows2).to_csv(out/'compute_target.csv',index=False)
# Recover exact configured relative paths from the nearby historical checkout; snapshot inputs.
rawroot=Path('/Users/hansgundlach/Documents/MITresearch/Experimental_Progress_Old/experimental_data_folder')
a=TrainingCurveAnalyzer(irreducible_loss=1.9,use_theoretical_flops=False)
manifest=[];missing=[]
for e in experiments_config:
 if e.get('class') not in ['lstm','transformer','sin transformer']:continue
 rel=Path(e['csv_path']).parts[2:];src=rawroot.joinpath(*rel)
 if not src.exists():missing.append(str(src));continue
 dst=ROOT/'inputs'/Path(*rel);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 manifest.append(dict(source=str(src),copy=str(dst.relative_to(ROOT)),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),family=e['class']))
 a.add_experiment(name=e['name'],csv_path=str(dst),compute_col=e.get('compute_col'),include_in_frontier=True,class_name=e['class'],hidden_dim=e.get('hidden_dim'))
(ROOT/'input_manifest.json').write_text(json.dumps(dict(files=manifest,missing=missing),indent=2))
classes=['lstm','transformer','sin transformer']
print('INCOMPLETE RECOVERED DATA: exploratory refits only. Missing:', missing)
a.identify_frontier_by_class(method='pareto',use_all_points=True,classes=classes,flop_range_by_class={c:(1e16,1e17) for c in classes})
raw={}
for c in classes:
 pts=a.frontier_points_all_by_class[c];raw[c]=pd.DataFrame(pts,columns=['run','compute','adjusted_loss']);raw[c]['raw_loss']=raw[c].adjusted_loss+1.9;raw[c].to_csv(out/(c.replace(' ','_')+'_frontier.csv'),index=False)
def fit(df,E):
 x=df.compute.values;y=df.raw_loss.values-E
 assert (y>0).all()
 par,cov=curve_fit(lambda x,A,b:A*x**b,x,y,p0=[y.mean()/x.mean()**(-.1),-.1],bounds=([1e-10,-1],[1e10,.5]),maxfev=10000)
 return par
saved=(u.A_L,u.alpha_L,u.A_T17,u.alpha_T17,u.A_TM,u.alpha_TM)
def evalfits(f):
 u.A_L,u.alpha_L=f['lstm'];u.A_T17,u.alpha_T17=f['sin transformer'];u.A_TM,u.alpha_TM=f['transformer'];return stats()
fits={c:fit(raw[c],1.9) for c in classes};refit=evalfits(fits)
Erows=[]
for E in sorted(set(np.linspace(1.68,2.01,34).tolist()+[1.9])):
 f={c:fit(raw[c],E) for c in classes};r=dict(E=E,**evalfits(f))
 for c in classes:r[c+'_A'],r[c+'_alpha']=f[c]
 Erows.append(r)
pd.DataFrame(Erows).to_csv(out/'irreducible_loss_refits.csv',index=False)
# Point deletions diagnostic only, not independent sample uncertainty. Run deletion rebuilds frontier.
checks=[]
for c in classes:
 df=raw[c]
 for mode in ['drop_smallest_point','drop_lowest_compute_10percent','leave_one_point_out']:
  selections=[df.iloc[1:]] if mode=='drop_smallest_point' else ([df[df.compute>df.compute.quantile(.1)]] if mode=='drop_lowest_compute_10percent' else [df.drop(i) for i in df.index])
  for j,sub in enumerate(selections):
   f=dict(fits);f[c]=fit(sub,1.9);checks.append(dict(family=c,check=mode,index=j,points=len(sub),**evalfits(f)))
 original=a.experiments.copy()
 for run in sorted(set(df.run)):
  a.experiments={n:e for n,e in original.items() if n!=run}
  with contextlib.redirect_stdout(io.StringIO()): a.identify_frontier_by_class(method='pareto',use_all_points=True,classes=[c],flop_range_by_class={c:(1e16,1e17)})
  sub=pd.DataFrame(a.frontier_points_all_by_class[c],columns=['run','compute','adjusted_loss']);sub['raw_loss']=sub.adjusted_loss+1.9
  if len(sub)<3:continue
  f=dict(fits);f[c]=fit(sub,1.9);checks.append(dict(family=c,check='leave_one_run_out_rebuilt_frontier',index=run,points=len(sub),**evalfits(f)))
 a.experiments=original
pd.DataFrame(checks).to_csv(out/'frontier_checks.csv',index=False)
u.A_L,u.alpha_L,u.A_T17,u.alpha_T17,u.A_TM,u.alpha_TM=saved
summary=dict(compute_2023=C,configured_baseline=base,recovered_data_refit_baseline=refit,fit_details={c:dict(A=float(fits[c][0]),alpha=float(fits[c][1]),points=len(raw[c]),runs=int(raw[c].run.nunique()),min_compute=float(raw[c].compute.min()),max_compute=float(raw[c].compute.max())) for c in classes})
(out/'summary.json').write_text(json.dumps(summary,indent=2))
for filename,x,label in [('exponent_gap.csv','deviation','Fractional change in exponent gap'),('irreducible_loss_refits.csv','E','Assumed irreducible loss'),('compute_target.csv','compute','Target compute (FLOPs)')]:
 d=pd.read_csv(out/filename);fig,axs=plt.subplots(1,2,figsize=(11,4))
 if 'method' in d:
  for name,g in d.groupby('method'):axs[0].plot(g[x]*100,g.total,label=name)
  axs[0].legend(fontsize=7);label='Change in exponent gap (%)';d=d[d.method=='original_fixed_coefficient'];xx=d[x]*100
 else:axs[0].plot(d[x],d.total);xx=d[x]
 axs[0].set_yscale('log');axs[0].set_ylabel('Total efficiency multiplier');axs[1].plot(xx,d.share if 'share' in d else d.architecture);axs[1].set_ylabel('Scale-dependent log-share (%)' if 'share' in d else 'Architecture multiplier')
 for ax in axs:ax.set_xlabel(label);ax.grid(alpha=.25)
 if x=='compute':
  for ax in axs:ax.set_xscale('log')
 fig.tight_layout();fig.savefig(out/(filename[:-4]+'.png'),dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))

# Apply publication styling after generating plots.
from restyle_figures import main as restyle
restyle()
