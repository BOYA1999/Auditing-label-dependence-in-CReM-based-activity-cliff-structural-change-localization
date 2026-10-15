import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
BASE=ROOT/'research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1'
CONCEPTS=['halogen_count','heteroatom_count','aromatic_atom_count']
VARIANTS=['actual','removal','insertion','turnover']
RDLogger.DisableLog('rdApp.*')

def labels(mol):
    atoms=list(mol.GetAtoms())
    return np.array([[a.GetAtomicNum() in {9,17,35,53} for a in atoms],[a.GetAtomicNum() in {7,8,15,16} for a in atoms],[a.GetIsAromatic() for a in atoms]],dtype=int)

def nap(y,scores):
    values=np.atleast_2d(scores)
    order=np.argsort(-values,axis=1,kind='stable')
    ranked=np.take_along_axis(values,order,axis=1)
    cumulative=np.cumsum(y[order],axis=1)
    ends=np.column_stack([ranked[:,1:]!=ranked[:,:-1],np.ones(len(values),dtype=bool)])
    previous=np.maximum.accumulate(np.where(ends,cumulative,0),axis=1)
    previous=np.column_stack([np.zeros(len(values)),previous[:,:-1]])
    ap=np.sum((cumulative/(np.arange(len(y))+1))*(cumulative-previous)*ends,axis=1)/y.sum()
    return (ap-y.mean())/(1-y.mean())

def interval(frame,column):
    blocks=[g.groupby('component_id')[column].median().to_numpy() for _,g in frame.groupby('dataset')]
    rng=np.random.default_rng(20261009)
    point=float(np.median([np.median(x) for x in blocks]))
    draws=np.median(np.column_stack([np.median(rng.choice(x,size=(5000,len(x))),axis=1) for x in blocks]),axis=1)
    return point,*np.quantile(draws,[.025,.975]).tolist()

def run(cohort):
    started=time.monotonic()
    if cohort=='development':
        molecules=pd.read_csv(BASE/'prepared/molecules.csv')
        mutants=pd.read_csv(BASE/'full/counterfactuals/scored_mutants.csv')
        pairs=pd.read_csv(BASE/'prepared/sentinel_pairs.csv')
        chosen=set(mutants.row_id)|{f'{r.dataset}:{int(i)}' for r in pairs.itertuples() for i in [r.left_index,r.right_index]}
        molecules=molecules[molecules.component_split.eq('test')&molecules.row_id.isin(chosen)]
    else:
        molecules=pd.read_csv(OUT/'selected_external_molecules.csv')
        mutants=pd.read_csv(OUT/'external_scored_mutants.csv')
    groups={k:g for k,g in mutants.groupby('row_id',sort=False)}
    mutant_counts={s:labels(Chem.MolFromSmiles(s)).sum(axis=1) for s in mutants.mutant_smiles.unique()}
    molecule_rows=[];mask_rows=[];operator_rows=[]
    for r in molecules.itertuples():
        mol=Chem.MolFromSmiles(r.canonical_smiles);truth=labels(mol);n=mol.GetNumAtoms()
        g=groups.get(r.row_id,pd.DataFrame())
        masks=list(g.groupby(['mask_id','mask_atoms','mask_size'],sort=True)) if len(g) else []
        incidence=np.zeros((n,len(masks)))
        for j,((_,atoms,_),_) in enumerate(masks):incidence[json.loads(atoms),j]=1
        degree=incidence.sum(axis=1)
        _,classes=np.unique(incidence,axis=0,return_inverse=True)
        operator_rows.append({'row_id':r.row_id,'dataset':r.dataset,'atoms':n,'supported_atoms':int((degree>0).sum()),'masks':len(masks),'distinct_atom_signatures':int(classes.max()+1),'incidence_rank':int(np.linalg.matrix_rank(incidence)) if incidence.size else 0})
        seed=int(hashlib.sha256(r.row_id.encode()).hexdigest()[:8],16)^20261009
        rng=np.random.default_rng(seed)
        permutations=np.tile(np.arange(len(masks)),(256,1))
        sizes=np.array([key[2] for key,_ in masks])
        for size in np.unique(sizes):
            ix=np.flatnonzero(sizes==size)
            permutations[:,ix]=np.array([rng.permutation(ix) for _ in range(256)])
        for k,concept in enumerate(CONCEPTS):
            y=truth[k];responses=np.zeros((len(masks),4))
            negative_insertions=0;creation_mass=0.;total_mass=0.
            for j,((mask_id,atoms,size),part) in enumerate(masks):
                removed=int(y[json.loads(atoms)].sum())
                delta=np.array([mutant_counts[s][k] for s in part.mutant_smiles])-y.sum()
                inserted=delta+removed
                negative_insertions+=int((inserted<0).sum())
                responses[j]=[np.abs(delta).mean(),removed,np.maximum(inserted,0).mean(),removed+np.maximum(inserted,0).mean()]
                creation=(removed==0)&(inserted>0)
                creation_mass+=float(np.abs(delta)[creation].sum());total_mass+=float(np.abs(delta).sum())
                mask_rows.append({'row_id':r.row_id,'dataset':r.dataset,'concept':concept,'mask_id':mask_id,'mask_size':size,'removed':removed,'mutants':len(part),'created_only':int(creation.sum()),'destroyed_only':int(((removed>0)&(inserted==0)).sum()),'cancelled':int(((removed>0)&(inserted==removed)).sum()),'negative_insertions':int((inserted<0).sum()),'actual_response':responses[j,0],'insertion_response':responses[j,2]})
            if y.sum()==0 or y.sum()==len(y):continue
            scores=np.divide(incidence@responses,degree[:,None],out=np.zeros((n,4)),where=degree[:,None]>0)
            actual=nap(y,scores.T)
            assert np.allclose(actual,[(average_precision_score(y,s)-y.mean())/(1-y.mean()) for s in scores.T],atol=1e-12)
            null=[]
            for v in range(4):
                shuffled=np.divide(responses[permutations,v]@incidence.T,degree,out=np.zeros((256,n)),where=degree>0)
                null.append(float(nap(y,shuffled).mean()))
            forced=sum(int(y[classes==c].sum())*int((1-y[classes==c]).sum()) for c in np.unique(classes))/(y.sum()*(n-y.sum()))
            record={'row_id':r.row_id,'dataset':r.dataset,'component_id':r.component_id,'concept':concept,'atoms':n,'positive_atoms':int(y.sum()),'masks':len(masks),'supported_fraction':float((degree>0).mean()),'positive_support':float((degree[y>0]>0).mean()),'negative_insertions':negative_insertions,'creation_response_mass_fraction':creation_mass/total_mass if total_mass else np.nan,'removal_leakage':float(scores[y==0,1].sum()/scores[:,1].sum()) if scores[:,1].sum() else np.nan,'forced_tie_auc_upper_bound':float(1-.5*forced),'permutable_masks':int(sum((sizes==size).sum() for size in np.unique(sizes) if (sizes==size).sum()>1))}
            for v,variant in enumerate(VARIANTS):
                record[variant+'_nap']=float(actual[v]);record[variant+'_response_gain']=float(actual[v]-null[v])
            record['removal_minus_actual']=float(actual[1]-actual[0])
            assert all(np.all(scores[classes==c]==scores[classes==c][0]) for c in np.unique(classes))
            molecule_rows.append(record)
    frame=pd.DataFrame(molecule_rows);maskframe=pd.DataFrame(mask_rows)
    frame.to_csv(OUT/f'{cohort}_molecule_metrics.csv',index=False)
    maskframe.to_csv(OUT/f'{cohort}_mask_accounting.csv',index=False)
    pd.DataFrame(operator_rows).to_csv(OUT/f'{cohort}_operator_metrics.csv',index=False)
    summary=[]
    for concept,g in frame.groupby('concept'):
        point,low,high=interval(g,'removal_minus_actual')
        row={'cohort':cohort,'concept':concept,'molecules':len(g),'supported_molecules':int(g.masks.gt(0).sum()),'paired_contrast':point,'ci_low':low,'ci_high':high,'negative_insertions':int(g.negative_insertions.sum())}
        for variant in VARIANTS:
            row[variant+'_molecule_median']=float(g[variant+'_nap'].median())
            row[variant+'_response_gain_median']=float(g[variant+'_response_gain'].median())
        row['creation_mass_fraction_median']=float(g.creation_response_mass_fraction.median())
        row['removal_leakage_median']=float(g.removal_leakage.median())
        summary.append(row)
    result=pd.DataFrame(summary)
    if cohort=='development':
        old=pd.read_csv(ROOT/'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008/count_oracle_summary.csv').set_index('control')
        for r in result.itertuples():
            assert r.molecules==int(old.loc[r.concept,'molecules'])
            assert abs(r.actual_molecule_median-old.loc[r.concept,'median_oracle_nap'])<1e-12
    result.to_csv(OUT/f'{cohort}_summary.csv',index=False)
    primary=result.set_index('concept').loc['heteroatom_count']
    gate=bool(primary.paired_contrast>=.10 and primary.ci_low>0 and primary.negative_insertions==0 and (cohort=='development' or primary.supported_molecules>=50))
    record={'status':'completed','cohort':cohort,'oracle_reproduction':cohort=='development','primary_gate_passed':gate,'primary':primary.to_dict(),'molecules_evaluated':len(molecules),'bootstrap_draws':5000,'mask_response_permutations':256,'new_model_training':False,'independent_bioactivity_validation':False,'elapsed_seconds':time.monotonic()-started,'claim_update':'Removal-accounting contrast is supported in this cohort' if gate else 'Mechanistic support gate not met; retain null or negative evidence','comparison_baselines':VARIANTS,'comparability':'Fixed masks, molecules, labels and averaging; only mask responses differ. Removal is a synthetic label-aligned response, not chemical deletion.'}
    (OUT/f'evaluation_summary_{cohort}.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(result.to_json(orient='records',indent=2),flush=True)

if __name__=='__main__':run(sys.argv[1])
