from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
cohorts=['development','external'];names=['Development','Independent panel']
frames=[pd.read_csv(OUT/f'{c}_molecule_metrics.csv') for c in cohorts]
summaries=[json.loads((OUT/f'evaluation_summary_{c}.json').read_text(encoding='utf-8'))['primary'] for c in cohorts]
fig,axes=plt.subplots(2,2,figsize=(7.4,5.6),layout='constrained')
ax=axes[0,0]
for i,r in enumerate(summaries):
    ax.errorbar(i,r['paired_contrast'],yerr=np.array([[r['paired_contrast']-r['ci_low']],[r['ci_high']-r['paired_contrast']]]),fmt='o',color='#245b89',capsize=5)
ax.axhline(0,color='#888888',lw=.7);ax.axhline(.1,color='#888888',lw=.7,ls='--')
ax.set(xticks=[0,1],xticklabels=names,xlim=(-.5,1.5),ylabel='Paired nAP difference',title='A   Heteroatom occupancy probe − actual')
ax=axes[0,1]
x=np.arange(2)
for offset,variant,color,label in [(-.18,'actual','#245b89','Actual net-count response'),(.18,'removal','#ca8a36','Label-aligned occupancy probe')]:
    values=[f[f.concept.eq('heteroatom_count')][variant+'_nap'].median() for f in frames]
    ax.bar(x+offset,values,width=.34,color=color,label=label)
ax.axhline(0,color='#888888',lw=.7);ax.set(xticks=x,xticklabels=names,ylim=(0,.50),ylabel='Molecule median nAP',title='B   Same regions and averaging')
ax.legend(frameon=False,fontsize=7,loc='upper left')
ax=axes[1,0]
values=[f[f.concept.eq('heteroatom_count')].creation_response_mass_fraction.dropna().to_numpy() for f in frames]
ax.boxplot(values,tick_labels=names,widths=.4,showfliers=False,medianprops={'color':'#245b89','lw':1.5})
ax.set(ylim=(-.03,1.03),ylabel='Fraction of absolute response mass',title='C   Response from concept-negative regions')
ax=axes[1,1]
concepts=['halogen_count','heteroatom_count','aromatic_atom_count'];labels=['Halogen','Heteroatom','Aromatic']
source=[]
for i,(cohort,f,color,name) in enumerate(zip(cohorts,frames,['#245b89','#ca8a36'],names)):
    values=[]
    for concept in concepts:
        g=f[f.concept.eq(concept)];n=int((g.forced_tie_auc_upper_bound<1-1e-12).sum())
        values.append(100*n/len(g));source.append({'cohort':cohort,'concept':concept,'eligible_molecules':len(g),'molecules_with_forced_cross_label_ties':n,'percent':100*n/len(g)})
    ax.bar(np.arange(3)+(i-.5)*.34,values,width=.34,color=color,label=name)
ax.set(xticks=np.arange(3),xticklabels=labels,ylim=(0,105),ylabel='Eligible molecules (%)',title='D   Atoms forced to share a score')
ax.legend(frameon=False,fontsize=8)
fig.savefig(OUT/'construct_diagnostics.png',dpi=300)
fig.savefig(OUT/'construct_diagnostics.pdf')
fig.savefig(OUT/'construct_diagnostics.svg')
pd.DataFrame(source).to_csv(OUT/'figure_forced_ties.csv',index=False)
pd.DataFrame(summaries).to_csv(OUT/'figure_primary_contrast.csv',index=False)
pd.concat([f.assign(cohort=c) for c,f in zip(cohorts,frames)]).to_csv(OUT/'figure_molecule_source.csv',index=False)
print('Figure and numeric sources saved')
