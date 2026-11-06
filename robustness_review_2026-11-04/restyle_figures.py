"""Restyle existing results only; never refit or overwrite numerical tables."""
from pathlib import Path
import os, hashlib
R=Path(__file__).resolve().parent; D=R/'results'
os.environ['MPLCONFIGDIR']=str(R/'cache');os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
STYLE={'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'mathtext.fontset':'stix','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8,'axes.linewidth':.6,'lines.linewidth':1.7,'pdf.fonttype':42,'ps.fonttype':42}
COLORS=['#0072B2','#D55E00','#009E73']
def finish(fig,axes,name,title,subtitle):
 fig.suptitle(title,fontsize=13,y=1.01)
 fig.text(.5,.91,subtitle,ha='center',fontsize=9,color='.4')
 for ax in axes:
  ax.grid(True,alpha=.18,linewidth=.5);ax.set_axisbelow(True)
  ax.tick_params(width=.6)
  for sp in ax.spines.values():sp.set_linewidth(.6)
 fig.tight_layout(rect=(0,0,1,.86),pad=.6,w_pad=2.2)
 for ext in ['png','pdf']:fig.savefig(D/(name+'.'+ext),dpi=400,bbox_inches='tight')
 plt.close(fig)
def main():
 files=list(D.glob('*.csv'))+list(D.glob('*.json'));before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 with plt.rc_context(STYLE):
  d=pd.read_csv(D/'exponent_gap.csv');fig,axs=plt.subplots(1,2,figsize=(7.1,3.2))
  methods=[('original_fixed_coefficient','fixed_coefficient',COLORS[0],'-'),('anchored_1e+16',r'Anchored at $10^{16}$ FLOPs',COLORS[1],'--'),('anchored_1e+17',r'Anchored at $10^{17}$ FLOPs',COLORS[2],'-.')]
  for method,label,color,ls in methods:
   g=d[d.method==method]
   for ax,key in zip(axs,['total','share']):ax.plot(g.deviation*100,g[key],color=color,ls=ls,label=label,marker='o',markevery=20,markersize=3)
  axs[0].set_yscale('log');axs[0].set_ylabel('Total efficiency gain (×)');axs[1].set_ylabel('Scale-dependent log-share (%)')
  for ax in axs:
   ax.axvline(0,color='.45',ls=':',lw=.8);ax.set_xlabel('Change in exponent gap (%)');ax.set_xticks([-25,-12.5,0,12.5,25]);ax.legend(framealpha=.95,edgecolor='.85',loc='upper left')
  axs[1].legend(framealpha=.95,edgecolor='.85',loc='lower right')
  finish(fig,axs,'exponent_gap','Sensitivity to the scaling-exponent gap','Configured fits · 2023 compute budget · scenario analysis')
  d=pd.read_csv(D/'irreducible_loss_refits.csv');fig,axs=plt.subplots(1,2,figsize=(7.1,3.2))
  axs[0].plot(d.E,d.total,color=COLORS[0],marker='o',markevery=5,markersize=3)
  axs[0].yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v/1000:g}k'))
  axs[0].set_ylabel('Total efficiency gain (×)')
  axs[1].plot(d.E,d.share,color=COLORS[2],marker='o',markevery=5,markersize=3);axs[1].set_ylabel('Scale-dependent log-share (%)')
  for ax,key in zip(axs,['total','share']):
   ax.axvline(1.9,ls=':',color='.45',lw=.9);base=d.iloc[(d.E-1.9).abs().argmin()];ax.scatter([1.9],[base[key]],s=32,color=COLORS[1],zorder=4)
   ax.set_xlabel(r'Assumed irreducible loss $E$');ax.text(.04,.94,r'Reference: $E=1.9$',transform=ax.transAxes,va='top',fontsize=8,color='.4')
  finish(fig,axs,'irreducible_loss_refits','Sensitivity to irreducible loss','Exploratory recovered-data refits · not bounds on the published estimate')
  # Companion exponent panel mirrors the original E-sensitivity figure.
  fig,ax=plt.subplots(figsize=(4.5,3.2))
  for col,label,color in [('transformer_alpha','Modern Transformer',COLORS[0]),('lstm_alpha','LSTM',COLORS[1]),('sin transformer_alpha','Historical Transformer',COLORS[2])]:ax.plot(d.E,d[col],label=label,color=color)
  ax.axvline(1.9,ls=':',color='.45',lw=.9);ax.set_xlabel(r'Irreducible loss $E$');ax.set_ylabel(r'Fitted scaling exponent $\alpha$');ax.legend(framealpha=.9,edgecolor='.8')
  finish(fig,[ax],'irreducible_loss_exponents','Scaling exponents under refitting','Exploratory recovered-data refits')
 assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())
 print('Saved PNG and PDF figures; numerical files unchanged.')
if __name__=='__main__':main()
