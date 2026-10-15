import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem
from analyze_deep_construct import labels,nap

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research_v2/artifacts/analysis/deep_construct_20261009_v1'
OUT=ROOT/'research_v2/artifacts/analysis/minor_revision_20261015/extension'
OUT.mkdir(parents=True,exist_ok=True)
RDLogger.DisableLog('rdApp.*')
sample=pd.read_csv(BASE/'selected_external_molecules.csv').set_index('row_id')
mutants=pd.read_csv(BASE/'external_scored_mutants.csv')
mols={row_id:Chem.MolFromSmiles(r.canonical_smiles) for row_id,r in sample.iterrows()}
counts={s:labels(Chem.MolFromSmiles(s)).sum(axis=1) for s in mutants.mutant_smiles.unique()}
metrics=[];atom_rows=[];mask_rows=[]
for row_id,g in mutants.groupby('row_id'):
    mol=mols[row_id]; truth=labels(mol); n=mol.GetNumAtoms()
    groups=list(g.groupby(['mask_id','mask_atoms'],sort=True)); matrix=np.zeros((n,len(groups)))
    for j,((_,atoms),_) in enumerate(groups):matrix[json.loads(atoms),j]=1
    degree=matrix.sum(axis=1)
    for k,(concept,name) in enumerate([('halogen_count','halogen'),('heteroatom_count','heteroatom')]):
        y=truth[k]
        if not 0<y.sum()<n:continue
        responses=[]
        for (mask_id,atoms),part in groups:
            values=[np.mean([abs(counts[s][k]-y.sum()) for s in part.mutant_smiles]),y[json.loads(atoms)].sum(),part['true_removed_'+name].mean()]
            responses.append(values)
            mask_rows.append({'row_id':row_id,'concept':concept,'mask_id':mask_id,'mask_atoms':atoms,'retained_mutants':len(part),'net_response':values[0],'occupancy_response':values[1],'native_removal_response':values[2]})
        scores=np.divide(matrix@np.array(responses),degree[:,None],out=np.zeros((n,3)),where=degree[:,None]>0)
        values=nap(y,scores.T)
        metrics.append({'row_id':row_id,'concept':concept,'component_id':sample.loc[row_id,'component_id'],'net_nap':values[0],'occupancy_nap':values[1],'native_removal_nap':values[2]})
        for i in range(n):
            atom_rows.append({'row_id':row_id,'concept':concept,'atom_index':i,'element':mol.GetAtomWithIdx(i).GetSymbol(),'current_concept':int(y[i]),'supported_masks':int(degree[i]),'net_score':scores[i,0],'occupancy_score':scores[i,1],'native_removal_score':scores[i,2]})
frame=pd.DataFrame(metrics); atoms=pd.DataFrame(atom_rows); masks=pd.DataFrame(mask_rows)
for a,b in [('occupancy','net'),('native_removal','net'),('native_removal','occupancy')]:frame[a+'_minus_'+b]=frame[a+'_nap']-frame[b+'_nap']
frame.to_csv(OUT/'common_population_molecule_metrics.csv',index=False)
atoms.to_csv(OUT/'common_population_atom_scores.csv',index=False)
masks.to_csv(OUT/'common_population_mask_responses.csv',index=False)
summary=[];draw_rows=[]
for concept,g in frame.groupby('concept'):
    g=g.sort_values('component_id'); indices=np.random.default_rng(20261009).integers(len(g),size=(5000,len(g)))
    record={'concept':concept,'supported_molecules':len(g),'bootstrap_draws':5000,'seed':20261009}
    draws={'draw':np.arange(5000),'concept':concept}
    for col in ['net_nap','occupancy_nap','native_removal_nap','occupancy_minus_net','native_removal_minus_net','native_removal_minus_occupancy']:
        values=g[col].to_numpy(); dist=np.median(values[indices],axis=1); lo,hi=np.quantile(dist,[.025,.975])
        record[col]=float(np.median(values));record[col+'_ci_low']=lo;record[col+'_ci_high']=hi;draws[col]=dist
    summary.append(record);draw_rows.append(pd.DataFrame(draws))
pd.DataFrame(summary).to_csv(OUT/'common_population_summary.csv',index=False)
pd.concat(draw_rows).to_csv(OUT/'common_population_bootstrap.csv',index=False)
original=pd.read_csv(BASE/'external_molecule_metrics.csv').merge(frame,on=['row_id','concept'])
native=pd.read_csv(BASE/'external_native_removal_metrics.csv').merge(frame,on=['row_id','concept'])
assert np.allclose(original.actual_nap,original.net_nap,atol=1e-12)
assert np.allclose(original.removal_nap,original.occupancy_nap,atol=1e-12)
assert np.allclose(native.true_removal_nap,native.native_removal_nap,atol=1e-12)
trace_rows=[];examples=[]
for index,r in enumerate(mutants.itertuples(index=False)):
    mol=mols[r.row_id]; expected=set(json.loads(r.true_removed_atoms)); exact=[]; target_count=0
    reaction=AllChem.ReactionFromSmarts(r.reaction_smarts)
    for products in reaction.RunReactants((Chem.AddHs(mol),)):
        product=products[0]
        if Chem.SanitizeMol(product,catchErrors=True):continue
        heavy=Chem.RemoveHs(product)
        if Chem.MolToSmiles(heavy,isomericSmiles=True)!=r.mutant_smiles:continue
        target_count+=1
        retained={a.GetIntProp('react_atom_idx'):a for a in heavy.GetAtoms() if a.HasProp('react_atom_idx')}
        removed=set(range(mol.GetNumAtoms()))-set(retained)
        if removed!=expected:continue
        inserted=[a for a in heavy.GetAtoms() if not a.HasProp('react_atom_idx')]
        checks=[]
        for name,elements in [('halogen',{9,17,35,53}),('heteroatom',{7,8,15,16})]:
            checks.extend([sum(mol.GetAtomWithIdx(i).GetAtomicNum() in elements for i in removed)==getattr(r,'true_removed_'+name),sum(a.GetAtomicNum() in elements for a in inserted)==getattr(r,'true_inserted_'+name)])
        if all(checks):exact.append(heavy)
    trace_rows.append({'record_index':index,'row_id':r.row_id,'mask_id':r.mask_id,'mutant_smiles':r.mutant_smiles,'matching_product_paths':target_count,'matching_origin_paths':len(exact),'trace_verified':bool(exact)})
    chosen=(r.row_id=='WOMBAT:132' and r.mask_id==1 and r.mutant_smiles=='N=C(CN)C(O)CN') or (r.row_id=='WOMBAT:107' and r.mask_id==3 and r.mutant_smiles=='CC(=O)Cc1ccc(CC(C)N)cc1')
    if chosen:
        assert exact
        product=exact[0]; trace=[]
        for a in product.GetAtoms():trace.append({'product_atom_index':a.GetIdx(),'element':a.GetSymbol(),'parent_atom_index':a.GetIntProp('react_atom_idx') if a.HasProp('react_atom_idx') else None})
        copied=Chem.Mol(product)
        for a in copied.GetAtoms():a.SetAtomMapNum(a.GetIdx()+1)
        allowed=json.loads(r.mask_atoms); native_removed=json.loads(r.true_removed_atoms)
        rm=masks[(masks.row_id==r.row_id)&(masks.concept=='heteroatom_count')&(masks.mask_id==r.mask_id)].iloc[0]
        examples.append({**r._asdict(),'event':'creation' if r.true_removed_heteroatom==0 else 'cancellation','parent_smiles':sample.loc[r.row_id,'canonical_smiles'],'allowed_parent_atoms':allowed,'deleted_parent_atoms':native_removed,'inserted_product_atoms':[a['product_atom_index'] for a in trace if a['parent_atom_index'] is None],'product_atom_provenance':trace,'product_smiles_with_atom_map_index_plus_one':Chem.MolToSmiles(copied),'single_mutant_net_response':abs(r.true_inserted_heteroatom-r.true_removed_heteroatom),'region_mean_net_response':float(rm.net_response),'region_retained_mutants':int(rm.retained_mutants),'parent_region_current_heteroatoms':sum(mol.GetAtomWithIdx(i).GetAtomicNum() in {7,8,15,16} for i in allowed)})
        writer=Chem.SDWriter(str(OUT/('example_'+examples[-1]['event']+'_product.sdf')));writer.write(product);writer.close()
    if (index+1)%2000==0:print('Native saved-reaction replay',index+1,flush=True)
traces=pd.DataFrame(trace_rows); traces.to_csv(OUT/'all_native_reaction_replay.csv',index=False)
assert traces.trace_verified.all()
assert len(examples)==2
(OUT/'molecular_examples.json').write_text(json.dumps(examples,indent=2)+'\n',encoding='utf8')
partial=mutants[[len(json.loads(a))<n for a,n in zip(mutants.true_removed_atoms,mutants.mask_size)]]
outside=mutants[mutants.outside_removed_atoms.gt(0)]
partial.to_csv(OUT/'partial_region_records.csv',index=False);outside.to_csv(OUT/'outside_index_records.csv',index=False)
pre='[!#1][CH]([NH]C(=O)[!#1])C(Cl)(Cl)Cl';post='[CHD3]([NH][CD3](=O))C(Cl)(Cl)Cl'
source_audit=json.loads((OUT/'inputs/source_population_audit.json').read_text(encoding='utf8'))
membership=OUT/'inputs/source_origin_membership.csv'
assert hashlib.sha256(membership.read_bytes()).hexdigest()==source_audit['membership_sha256']
source=pd.read_csv(membership)
selected=sample.reset_index().merge(source[['row_id','source_smiles','ORIGIN','MoleculeCDetector']],on=['row_id','source_smiles'],validate='1:1')
assert len(selected)==len(sample)
selected['pre_smarts_match']=[mols[r].HasSubstructMatch(Chem.MolFromSmarts(pre)) for r in selected.row_id]
selected['post_smarts_match']=[mols[r].HasSubstructMatch(Chem.MolFromSmarts(post)) for r in selected.row_id]
selected[['row_id','source_smiles','ORIGIN','MoleculeCDetector','pre_smarts_match','post_smarts_match']].to_csv(OUT/'source_origin_membership.csv',index=False)
origin={**source_audit,'selected':len(selected),'selected_origin_counts':selected.ORIGIN.value_counts().to_dict(),'selected_pre_smarts_positive':int(selected.pre_smarts_match.sum()),'selected_post_smarts_positive':int(selected.post_smarts_match.sum()),'pre_smarts':pre,'post_smarts':post,'source_composition_evidence':'Frozen membership audit of the pinned full parquet; full source selection independently rebuilt in the revision replay.'}
(OUT/'wombat_pattern1_definition.json').write_text(json.dumps(origin,indent=2)+'\n',encoding='utf8')
table=[]
for r in summary:
    for key,name in [('net_nap','Net response: marginal median nAP'),('occupancy_nap','Region occupancy: marginal median nAP'),('native_removal_nap','Native removal: marginal median nAP'),('occupancy_minus_net','Paired occupancy minus net'),('native_removal_minus_net','Paired native removal minus net'),('native_removal_minus_occupancy','Paired native removal minus occupancy')]:
        table.append({'Concept':r['concept'].replace('_count','').capitalize(),'Supported structures':int(r['supported_molecules']),'Quantity':name,'Estimate':f'{r[key]:.5f}','95% lower':f'{r[key+"_ci_low"]:.5f}','95% upper':f'{r[key+"_ci_high"]:.5f}'})
pd.DataFrame(table).to_csv(OUT/'table_s45.csv',index=False)
table=[]
for category,name in [('POSITIVE','Pattern-positive'),('TVERSKY','Tversky-selected negative'),('NEGATIVE','Ordinary negative')]:
    table.append({'Panel':'Source composition','Item':name,'Source pool':origin['source_origin_counts'].get(category,0),'Selected panel':origin['selected_origin_counts'].get(category,0),'Detail':'Upstream ORIGIN category; source pool construction precedes this study.'})
table.append({'Panel':'Source composition','Item':'Total','Source pool':origin['source_rows'],'Selected panel':origin['selected'],'Detail':'Current pre- and post-SMARTS matches among selected structures: 0 / 256.'})
for r in sorted(examples,key=lambda x:x['event'],reverse=True):
    table.append({'Panel':'Illustrative saved event','Item':r['event'].capitalize(),'Source pool':'','Selected panel':r['row_id'],'Detail':f"Region {r['mask_id']}; parent atoms {r['allowed_parent_atoms']}; deleted heteroatoms {r['true_removed_heteroatom']}; inserted heteroatoms {r['true_inserted_heteroatom']}; absolute net change {r['single_mutant_net_response']}; region mean {r['region_mean_net_response']:.5f} across {r['region_retained_mutants']} retained mutant(s)."})
pd.DataFrame(table).to_csv(OUT/'table_s46.csv',index=False)
result={'status':'PASS','selected':len(sample),'supported':mutants.row_id.nunique(),'retained_mutants':len(mutants),'all_saved_reaction_traces_verified':int(traces.trace_verified.sum()),'partial_region_rows':len(partial),'partial_region_molecules':partial.row_id.nunique(),'outside_index_rows':len(outside),'common_population':summary,'source_composition':origin,'scope':'Frozen saved-reaction and statistical replay, not model retraining or complete CReM neighbourhood regeneration.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'extension_revision_qa.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
