import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from analyze_deep_construct import labels,nap,interval

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
RDLogger.DisableLog('rdApp.*')
variation=[]
for cohort in ['development','external']:
    masks=pd.read_csv(OUT/f'{cohort}_mask_accounting.csv')
    for (row_id,concept),g in masks.groupby(['row_id','concept']):
        for variant,column in [('actual','actual_response'),('removal','removed'),('insertion','insertion_response'),('turnover',None)]:
            values=g[column] if column else g.removed+g.insertion_response
            groups=pd.DataFrame({'size':g.mask_size,'value':values}).groupby('size').value
            variation.append({'cohort':cohort,'row_id':row_id,'concept':concept,'variant':variant,'informative_masks':int(sum(len(s) for _,s in groups if s.max()-s.min()>1e-12)),'supported_masks':len(g)})
pd.DataFrame(variation).to_csv(OUT/'permutation_informativeness.csv',index=False)
operator_rows=[]
for cohort in ['development','external']:
    base=ROOT/'research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1'
    parents=pd.read_csv(base/'prepared/molecules.csv' if cohort=='development' else OUT/'selected_external_molecules.csv').set_index('row_id')
    cache=pd.read_csv(base/'full/counterfactuals/scored_mutants.csv' if cohort=='development' else OUT/'external_scored_mutants.csv')
    groups={k:g for k,g in cache.groupby('row_id')}
    metrics=pd.read_csv(OUT/f'{cohort}_molecule_metrics.csv')
    for row_id,rows in metrics.groupby('row_id'):
        mol=Chem.MolFromSmiles(parents.loc[row_id,'canonical_smiles']);truth=labels(mol)
        part=groups.get(row_id,pd.DataFrame(columns=['mask_id','mask_atoms']))
        masks=part[['mask_id','mask_atoms']].drop_duplicates();matrix=np.zeros((mol.GetNumAtoms(),len(masks)))
        for j,r in enumerate(masks.itertuples()):matrix[json.loads(r.mask_atoms),j]=1
        supported=matrix.sum(axis=1)>0;_,classes=np.unique(matrix,axis=0,return_inverse=True)
        for r in rows.itertuples():
            y=truth[['halogen_count','heteroatom_count','aromatic_atom_count'].index(r.concept)]
            forced_all=0;forced_supported=0
            for c in np.unique(classes):
                ix=classes==c;pairs=int(y[ix].sum()*(1-y[ix]).sum());forced_all+=pairs
                if supported[ix].all():forced_supported+=pairs
            assert abs((1-.5*forced_all/(y.sum()*(len(y)-y.sum())))-r.forced_tie_auc_upper_bound)<1e-12
            operator_rows.append({'cohort':cohort,'row_id':row_id,'concept':r.concept,'any_positive_unsupported':bool((~supported[y>0]).any()),'forced_cross_label_pairs_all':forced_all,'forced_cross_label_pairs_supported':forced_supported,'any_supported_cross_label_tie':forced_supported>0})
pd.DataFrame(operator_rows).to_csv(OUT/'operator_support_separation.csv',index=False)
mutants=pd.read_csv(OUT/'external_scored_mutants.csv')
sample=pd.read_csv(OUT/'selected_external_molecules.csv').set_index('row_id')
symmetry=[];native_rows=[];trace_summary=[]
for row_id,g in mutants.groupby('row_id'):
    mol=Chem.MolFromSmiles(sample.loc[row_id,'canonical_smiles']);y=labels(mol)
    outside=g[g.outside_removed_atoms.gt(0)]
    if len(outside):
        maps=mol.GetSubstructMatches(mol,uniquify=False,useChirality=True,maxMatches=10000)
        for (mask_atoms,true_atoms),part in outside.groupby(['mask_atoms','true_removed_atoms']):
            mask=json.loads(mask_atoms);actual=set(json.loads(true_atoms))
            witness=any({mapping[i] for i in mask}==actual for mapping in maps)
            symmetry.append({'row_id':row_id,'allowed_atoms':mask_atoms,'native_removed_atoms':true_atoms,'retained_mutants':len(part),'automorphism_witness':witness,'mapping_search_censored':len(maps)==10000 and not witness})
    for concept,index,name in [('halogen_count',0,'halogen'),('heteroatom_count',1,'heteroatom')]:
        truth=y[index]
        if truth.sum()==0 or truth.sum()==len(truth):continue
        incidence=[];responses=[];mismatch=0
        for (_,atoms,_),part in g.groupby(['mask_id','mask_atoms','mask_size'],sort=True):
            mask=json.loads(atoms);vector=np.zeros(len(truth));vector[mask]=1
            incidence.append(vector);responses.append(part['true_removed_'+name].mean())
            mismatch+=int((part['true_removed_'+name]!=truth[mask].sum()).sum())
        matrix=np.array(incidence).T;degree=matrix.sum(axis=1)
        scores=np.divide(matrix@np.array(responses),degree,out=np.zeros(len(truth)),where=degree>0)
        native_rows.append({'row_id':row_id,'dataset':sample.loc[row_id,'dataset'],'component_id':sample.loc[row_id,'component_id'],'concept':concept,'true_removal_nap':float(nap(truth,scores)[0]),'occupancy_count_mismatch_rows':mismatch})
native=pd.DataFrame(native_rows)
original=pd.read_csv(OUT/'external_molecule_metrics.csv')
native=native.merge(original[['row_id','concept','actual_nap','removal_nap']],on=['row_id','concept'])
native['true_removal_minus_actual']=native.true_removal_nap-native.actual_nap
native.to_csv(OUT/'external_native_removal_metrics.csv',index=False)
for concept,name in [('halogen_count','halogen'),('heteroatom_count','heteroatom')]:
    lost=mutants['true_removed_'+name];gained=mutants['true_inserted_'+name]
    absolute=(gained-lost).abs();creation=(lost==0)&(gained>0)
    group=native[native.concept.eq(concept)]
    point,low,high=interval(group,'true_removal_minus_actual')
    trace_summary.append({'concept':concept,'retained_mutants':len(mutants),'actual_creation_only':int(creation.sum()),'actual_destruction_only':int(((lost>0)&(gained==0)).sum()),'actual_cancellation':int(((lost>0)&(lost==gained)).sum()),'actual_both_zero':int(((lost==0)&(gained==0)).sum()),'creation_absolute_response_mass_fraction_pooled':float(absolute[creation].sum()/absolute.sum()) if absolute.sum() else None,'occupancy_count_mismatch_rows':int(group.occupancy_count_mismatch_rows.sum()),'true_removal_nap_median':float(group.true_removal_nap.median()),'paired_true_removal_contrast':point,'ci_low':low,'ci_high':high})
pd.DataFrame(symmetry).to_csv(OUT/'external_symmetry_witnesses.csv',index=False)
pd.DataFrame(trace_summary).to_csv(OUT/'external_native_trace_summary.csv',index=False)
record={'native_retained_mutants':len(mutants),'outside_allowed_region_rows':int(mutants.outside_removed_atoms.gt(0).sum()),'retained_elemental_change_rows':int(mutants.retained_element_changes.gt(0).sum()),'elemental_accounting_invalid_rows':int((~mutants.elemental_accounting_valid).sum()),'partial_region_replacement_rows':sum(len(json.loads(s))<n for s,n in zip(mutants.true_removed_atoms,mutants.mask_size)),'outside_region_symmetry_cases':len(symmetry),'outside_rows_with_automorphism_witness':sum(r['retained_mutants'] for r in symmetry if r['automorphism_witness']),'unresolved_outside_cases':sum(not r['automorphism_witness'] for r in symmetry),'concepts':trace_summary}
(OUT/'native_trace_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
scaffolds={}
for row_id,r in sample.iterrows():
    temporary=Chem.MolFromSmiles(r.canonical_smiles);Chem.RemoveStereochemistry(temporary)
    scaffolds[row_id]=MurckoScaffold.MurckoScaffoldSmiles(mol=temporary,includeChirality=False) or 'ACYCLIC'
original['scaffold']=original.row_id.map(scaffolds)
primary=original[original.concept.eq('heteroatom_count')]
blocks=[g.removal_minus_actual.to_numpy() for _,g in primary.groupby('scaffold')]
rng=np.random.default_rng(20261009)
draws=[np.median(np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))])) for _ in range(5000)]
bench_scaffolds=set()
for file in (ROOT/'external/MoleculeACE/MoleculeACE/Data/benchmark_data').glob('*.csv'):
    data=pd.read_csv(file);column=next(n for n in data.columns if n.lower() in {'smiles','canonical_smiles'})
    for value in data[column].dropna():
        mol=Chem.MolFromSmiles(value)
        if mol:
            Chem.RemoveStereochemistry(mol)
            bench_scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False) or 'ACYCLIC')
scaffold_record={'status':'posthoc uncertainty sensitivity; primary gate unchanged','selected_scaffolds':len(set(scaffolds.values())),'eligible_heteroatom_scaffolds':len(blocks),'largest_eligible_block':max(map(len,blocks)),'selected_with_benchmark_scaffold':sum(s in bench_scaffolds for s in scaffolds.values()),'point_molecule_median':float(primary.removal_minus_actual.median()),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),'bootstrap_draws':5000,'seed':20261009,'all_acyclic_one_block':True}
(OUT/'external_scaffold_sensitivity.json').write_text(json.dumps(scaffold_record,indent=2)+'\n',encoding='utf-8')
original[['row_id','concept','scaffold','removal_minus_actual']].to_csv(OUT/'external_scaffold_membership.csv',index=False)
print(json.dumps(record,indent=2),flush=True)
print(json.dumps(scaffold_record,indent=2),flush=True)
