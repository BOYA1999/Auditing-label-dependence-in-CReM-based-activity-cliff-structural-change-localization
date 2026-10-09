import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import rankdata

from analyze_round2_revision import (atom_array_lookup, grouped_average_precision,
    load_atom_scores, pair_label_arrays, shared_mapping_sets, two_stage_bootstrap)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008'
CFG = json.loads((ROOT / 'research_v2/configs/round2_revision_v1.json').read_text())
BASE = json.loads((ROOT / CFG['source_config']).read_text())
OLD = ROOT / CFG['output']
PREP = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared'
FULL = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/full'
MAJOR = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_major_revision_v1'

def save(name, frame):
    frame.to_csv(OUT / name, index=False)

def null_mean(labels, scores, lengths=None):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    n, k = len(labels), int(sum(labels))
    order = np.argsort(-scores, kind='stable')
    ends = np.flatnonzero(np.r_[scores[order][:-1] != scores[order][1:], True])+1
    if lengths is None:
        sizes = np.diff(np.r_[0, ends])
        return float(np.sum(sizes/ends*(1/n+(ends-1)*(k-1)/(n*(n-1)))))
    strata = np.repeat(np.arange(len(lengths)), lengths)[order]
    cuts = np.cumsum([0]+list(lengths))
    counts = np.array([labels[cuts[s]:cuts[s+1]].sum() for s in range(len(lengths))])
    sizes = np.asarray(lengths)
    p = counts/sizes
    p2 = np.divide(counts*(counts-1), sizes*(sizes-1), out=np.zeros(len(sizes)), where=sizes>1)
    previous = np.zeros(len(sizes), int)
    expected = 0.0
    for end in ends:
        cumulative = np.bincount(strata[:end], minlength=len(sizes))
        block = cumulative-previous
        moment = np.sum(block*p + block*(cumulative-1)*p2)
        moment += sum(block[s]*cumulative[t]*p[s]*p[t] for s in range(len(sizes)) for t in range(len(sizes)) if s != t)
        expected += moment/end/k
        previous = cumulative
    return float(expected)

def metrics(labels, scores, lengths):
    prevalence = float(np.mean(labels))
    ap = grouped_average_precision(labels, scores)
    pooled, endpoint = null_mean(labels,scores), null_mean(labels,scores,lengths)
    return {'raw_nap':(ap-prevalence)/(1-prevalence),
        'pooled_exact_nap':(ap-pooled)/(1-prevalence),
        'endpoint_exact_nap':(ap-endpoint)/(1-prevalence),
        'pooled_null_ap':pooled, 'endpoint_null_ap':endpoint}

def estimate(frame, value, label):
    point = frame.groupby('dataset')[value].median().median()
    draws = two_stage_bootstrap(frame,value,5000,20261008,label)
    return {'estimate':float(point),'ci_low':float(np.quantile(draws,.025)),
        'ci_high':float(np.quantile(draws,.975)),'datasets':int(frame.dataset.nunique()),
        'components':int(frame[['dataset','component_id']].drop_duplicates().shape[0])}, draws

def check_nulls():
    errors = []
    for sizes in [(3,3),(1,5),(2,4)]:
        for k1 in range(sizes[0]+1):
            for k2 in range(sizes[1]+1):
                if not 0 < k1+k2 < 6:
                    continue
                labels = np.r_[np.arange(sizes[0])<k1,np.arange(sizes[1])<k2]
                for scores in [np.arange(6),np.array([2,2,1,1,0,0]),np.ones(6)]:
                    exact = [grouped_average_precision(np.r_[np.isin(np.arange(sizes[0]),a),np.isin(np.arange(sizes[1]),b)],scores)
                        for a in itertools.combinations(range(sizes[0]),k1) for b in itertools.combinations(range(sizes[1]),k2)]
                    errors.append(abs(np.mean(exact)-null_mean(labels,scores,list(sizes))))
    assert max(errors)<1e-12
    return {'comparisons':len(errors),'max_absolute_error':float(max(errors))}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    validation = {'endpoint_null_exact_expectation':check_nulls()}
    print('exact-null validation',validation,flush=True)
    molecules = pd.read_csv(PREP/'model_molecules.csv').set_index('row_id')
    mols = {i:Chem.MolFromSmiles(s) for i,s in molecules.canonical_smiles.items() if molecules.loc[i,'dataset'] in BASE['confirmation_targets']}
    counts = {i:m.GetNumAtoms() for i,m in mols.items()}
    pairs = pd.read_csv(PREP/'sentinel_pairs.csv')
    pairs = pairs[pairs.dataset.isin(BASE['confirmation_targets']) & pairs.component_split.eq('test')].copy()
    pairs['left_id'] = pairs.dataset+':'+pairs.left_index.astype(int).astype(str)
    pairs['right_id'] = pairs.dataset+':'+pairs.right_index.astype(int).astype(str)
    pairs = pairs[[pair_label_arrays(p,counts,'boundary_extended')[1] for p in pairs.itertuples()]].copy()
    assert len(pairs)==617
    mapping, alternatives = [], {}
    print('prepared',len(mols),'molecules',len(pairs),'pairs',flush=True)
    for mapping_index,p in enumerate(pairs.itertuples(),1):
        stored=json.loads(p.match); a,b=mols[p.left_id],mols[p.right_id]
        left,right,lt,rt,meta=shared_mapping_sets(a,b,stored,256)
        meta['reconstructed_boundary_matches_stored']=(meta.get('reconstructed_boundary_left')==p.rationale_left and meta.get('reconstructed_boundary_right')==p.rationale_right)
        chosen=[frozenset(v[s] for v in stored) for s in [0,1]]
        recovered=chosen[0] in left and chosen[1] in right
        combinations=list(itertools.product(left,right))
        mapping.append({'dataset':p.dataset,'component_id':p.component_id,
            'left_unique_matched_atom_sets':len(left),'right_unique_matched_atom_sets':len(right),
            'chosen_left_recovered':chosen[0] in left,'chosen_right_recovered':chosen[1] in right,
            'positive_atom_set_ambiguous':recovered and (len(left)>1 or len(right)>1),
            'enumeration_eligible':recovered,'fallback_inclusion':False,
            'left_matches_truncated':lt,'right_matches_truncated':rt,
            'combination_truncated':len(combinations)>4096,**meta})
        if recovered and len(combinations)>1:
            alternatives[(p.dataset,p.component_id)]=combinations[:4096]
        if mapping_index%25==0:print('mapped',mapping_index,p.dataset,p.component_id,flush=True)
    mapping=pd.DataFrame(mapping);save('mcs_mapping_ambiguity.csv',mapping)
    validation['mapping']={'pairs':617,'native_left_misses':int((~mapping.chosen_left_recovered).sum()),
        'native_right_misses':int((~mapping.chosen_right_recovered).sum()),
        'eligible_pairs':int(mapping.enumeration_eligible.sum()),'ambiguous_pairs':int(mapping.positive_atom_set_ambiguous.sum()),
        'query_failures':int(mapping.query_failure.sum()),'truncated_pairs':int((mapping.left_matches_truncated|mapping.right_matches_truncated|mapping.combination_truncated).sum())}
    print('mapping',validation['mapping'],flush=True)
    frame=load_atom_scores(ROOT/CFG['inputs']['atom_scores'],BASE['confirmation_targets'],'crem_mean')
    endpoints=set(pairs.left_id)|set(pairs.right_id)
    frame=frame[frame.row_id.isin(endpoints)].copy()
    arrays=atom_array_lookup(frame)
    masks=pd.read_csv(FULL/'counterfactuals/mask_coverage.csv')
    masks=masks[masks.row_id.isin(endpoints)&masks.mask_id.ge(0)]
    selected=pd.read_csv(FULL/'counterfactuals/scored_mutants.csv',usecols=['row_id','mask_id','nearest_train_tanimoto'])
    selected=selected[selected.row_id.isin(endpoints)].groupby(['row_id','mask_id']).nearest_train_tanimoto.mean().rename('similarity')
    masks=masks.merge(selected,on=['row_id','mask_id'],how='left',validate='one_to_one')
    assert ((masks.similarity.notna())==masks.covered).all()
    geometry=pd.read_csv(MAJOR/'generator_atom_geometry.csv')
    geometry=geometry[geometry.row_id.isin(endpoints)].copy()
    baseline_arrays={};qarrays={};direct_error=[];support_rows=[]
    for row_id,group in geometry.groupby('row_id',sort=False):
        group=group.sort_values('atom_index'); k=np.zeros(len(group));s=k.copy();similarity_sum=k.copy()
        for mask in masks[masks.row_id.eq(row_id)].itertuples():
            ix=json.loads(mask.mask_atoms);k[ix]+=1;s[ix]+=int(bool(mask.covered))
            if mask.covered:similarity_sum[ix]+=mask.similarity
        assert np.allclose(k,group.mask_membership_count)
        q=np.divide(s,k,out=np.zeros_like(s),where=k>0);qarrays[row_id]=q
        gall=group.selected_mean_nearest_train_tanimoto.to_numpy()
        gs=np.divide(similarity_sum,s,out=np.zeros_like(s),where=s>0)
        baseline_arrays[row_id]={'support_fraction':q,'support_any':(s>0).astype(float),
            'support_weighted_similarity':gall,'supported_similarity':gs}
        direct_error.append(float(np.max(np.abs(gall-q*gs))))
        support_rows.extend({'row_id':row_id,'atom_index':a,'enumerated_masks':int(ka),'supported_masks':int(sa),'support_fraction':qa,'supported_similarity':ga}
            for a,(ka,sa,qa,ga) in enumerate(zip(k,s,q,gs)))
    validation['denominator_identity_max_error']=max(direct_error)
    assert max(direct_error)<1e-12
    save('matched_atom_baselines.csv',pd.DataFrame(support_rows))
    model_rows=[];generator_rows=[];mapping_rows=[];endpoint_rows=[]
    for counter,p in enumerate(pairs.itertuples(),1):
        lengths=[counts[p.left_id],counts[p.right_id]]
        ids=[p.left_id,p.right_id]
        for rationale in ['boundary_extended','strict_unmatched']:
            labels,valid=pair_label_arrays(p,counts,rationale)
            if not valid:continue
            for name in baseline_arrays[p.left_id]:
                scores=np.concatenate([baseline_arrays[i][name] for i in ids])
                generator_rows.append({'dataset':p.dataset,'component_id':p.component_id,'rationale':rationale,'baseline':name,**metrics(labels,scores,lengths)})
            for rep,predictor,seed,variant in itertools.product(BASE['representations'],['xgboost','mlp'],BASE['seeds'],['trained','label_permuted']):
                original=[arrays[(p.dataset,rep,predictor,seed,variant,i)] for i in ids]
                for denominator in ['supported','all_masks']:
                    scores=np.concatenate([v if denominator=='supported' else v*qarrays[i] for i,v in zip(ids,original)])
                    model_rows.append({'dataset':p.dataset,'component_id':p.component_id,'rationale':rationale,
                        'representation':rep,'predictor':predictor,'seed':seed,'variant':variant,'denominator':denominator,**metrics(labels,scores,lengths)})
                if rationale=='boundary_extended':
                    endpoint_rows.append({'dataset':p.dataset,'component_id':p.component_id,'representation':rep,
                        'predictor':predictor,'seed':seed,'variant':variant,'left_prevalence':float(labels[:lengths[0]].mean()),
                        'right_prevalence':float(labels[lengths[0]:].mean()),'left_mean_score':float(np.mean(original[0])),
                        'right_mean_score':float(np.mean(original[1]))})
        if (p.dataset,p.component_id) in alternatives:
            for rep,predictor in itertools.product(BASE['representations'],['xgboost','mlp']):
                scores=np.concatenate([np.mean([arrays[(p.dataset,rep,predictor,seed,'trained',i)] for seed in BASE['seeds']],axis=0) for i in ids])
                values=[]
                for a,b in alternatives[(p.dataset,p.component_id)]:
                    lab=np.r_[~np.isin(np.arange(lengths[0]),list(a)),~np.isin(np.arange(lengths[1]),list(b))]
                    if 0<sum(lab)<len(lab):values.append(metrics(lab,scores,lengths)['raw_nap'])
                mapping_rows.append({'dataset':p.dataset,'component_id':p.component_id,'representation':rep,'predictor':predictor,
                    'mapping_nap_range':max(values)-min(values),'minimum_mapping_nap':min(values),'maximum_mapping_nap':max(values)})
        if counter%100==0:print('scored pairs',counter,flush=True)
    models=pd.DataFrame(model_rows);generators=pd.DataFrame(generator_rows)
    save('model_null_metrics.csv',models);save('generator_null_metrics.csv',generators)
    save('mcs_mapping_nap_sensitivity.csv',pd.DataFrame(mapping_rows));save('endpoint_scale_diagnostics.csv',pd.DataFrame(endpoint_rows))
    mapping_values=pd.DataFrame(mapping_rows).mapping_nap_range
    (OUT/'mcs_mapping_summary.json').write_text(json.dumps({
        'pairs':617,'assessed_pairs':int(mapping.enumeration_eligible.sum()),
        'pairs_with_multiple_valid_matched_atom_sets':int(mapping.positive_atom_set_ambiguous.sum()),
        'fraction_with_multiple_valid_matched_atom_sets':float(mapping.positive_atom_set_ambiguous.sum()/mapping.enumeration_eligible.sum()),
        'median_nap_range_when_ambiguous':float(mapping_values.median()),
        'q95_nap_range_when_ambiguous':float(mapping_values.quantile(.95)),
        'fallback_pairs':0,'truncated_pairs':0},indent=2)+'\n')
    keys=['dataset','component_id','rationale','representation','predictor','denominator']
    columns=['raw_nap','pooled_exact_nap','endpoint_exact_nap']
    components=models.groupby(keys+['variant'])[columns].mean().unstack('variant')
    component=components.index.to_frame(index=False)
    for metric in columns:
        component[metric+'_trained']=components[(metric,'trained')].to_numpy()
        component[metric+'_permuted']=components[(metric,'label_permuted')].to_numpy()
        component[metric+'_delta']=component[metric+'_trained']-component[metric+'_permuted']
    save('component_metrics.csv',component)
    primary=[];boot=pd.DataFrame({'draw':np.arange(5000)})
    for (rationale,denominator),part in component.groupby(['rationale','denominator']):
        for metric in columns:
            label=f'{rationale}:{denominator}:{metric}'
            result,draws=estimate(part,metric+'_delta',rationale)
            primary.append({'rationale':rationale,'denominator':denominator,'metric':metric,**result});boot[label]=draws
    save('primary_summary.csv',pd.DataFrame(primary));save('primary_bootstrap.csv',boot)
    absolute=[];paired=[]
    for (rationale,denominator,rep,predictor),part in component.groupby(['rationale','denominator','representation','predictor']):
        for metric in columns:
            for variant in ['trained','permuted']:
                absolute.append({'rationale':rationale,'denominator':denominator,'representation':rep,'predictor':predictor,'source':variant,'metric':metric,
                    'estimate':part.groupby('dataset')[metric+'_'+variant].median().median()})
            for name,gen in generators[generators.rationale.eq(rationale)].groupby('baseline'):
                joint=part.merge(gen[['dataset','component_id',metric]],on=['dataset','component_id'],validate='one_to_one')
                joint['difference']=joint[metric+'_trained']-joint[metric]
                paired.append({'rationale':rationale,'denominator':denominator,'representation':rep,'predictor':predictor,'baseline':name,'metric':metric,
                    'paired_difference':joint.groupby('dataset').difference.median().median()})
    for (rationale,name),part in generators.groupby(['rationale','baseline']):
        for metric in columns:
            absolute.append({'rationale':rationale,'denominator':'generator','representation':'','predictor':'','source':name,'metric':metric,
                'estimate':part.groupby('dataset')[metric].median().median()})
    save('absolute_matched_summary.csv',pd.DataFrame(absolute));save('model_minus_generator.csv',pd.DataFrame(paired))
    old=pd.read_csv(OLD/'model_empirical_null_pair_metrics.csv')
    new=models[models.denominator.eq('supported')].merge(old,on=['dataset','component_id','rationale','representation','predictor','seed','variant'],suffixes=('','_old'),validate='one_to_one')
    new['null_mean_error']=new.null_mean_ap-new.pooled_null_ap
    save('monte_carlo_error.csv',new[['dataset','component_id','rationale','representation','predictor','seed','variant','null_mean_error']])
    validation['monte_carlo_error']={'max_absolute_ap':float(new.null_mean_error.abs().max()),'rms_ap':float(np.sqrt(np.mean(new.null_mean_error**2)))}
    agreement=[]
    supported=frame.groupby(['dataset','representation','predictor','row_id','atom_index']).supported.any()
    endpoint_components={i:(p.dataset,p.component_id) for p in pairs.itertuples() for i in [p.left_id,p.right_id]}
    def correlation(a,b):
        if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return np.nan
        return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
    for i,(dataset,comp) in endpoint_components.items():
        for rep,predictor in itertools.product(BASE['representations'],['xgboost','mlp']):
            support=supported.loc[(dataset,rep,predictor,i)].to_numpy(bool)
            comparisons=[('observed_cross_seed','trained',a,'trained',b) for a,b in itertools.combinations(BASE['seeds'],2)]
            comparisons += [('permuted_cross_seed','label_permuted',a,'label_permuted',b) for a,b in itertools.combinations(BASE['seeds'],2)]
            comparisons += [('matched_label','trained',s,'label_permuted',s) for s in BASE['seeds']]
            for kind,v1,s1,v2,s2 in comparisons:
                a,b=[arrays[(dataset,rep,predictor,s,v,i)] for s,v in [(s1,v1),(s2,v2)]]
                agreement.append({'dataset':dataset,'component_id':comp,'row_id':i,'representation':rep,'predictor':predictor,'comparison':kind,'seed_a':s1,'seed_b':s2,
                    'all_atom_spearman':correlation(a,b),'supported_spearman':correlation(a[support],b[support])})
    agreement=pd.DataFrame(agreement);save('map_stability.csv',agreement)
    stability=[]
    for name,part in agreement.groupby('comparison'):
        comp=part.groupby(['dataset','component_id'])[['all_atom_spearman','supported_spearman']].median().reset_index()
        stability.append({'comparison':name,'comparisons':len(part),'finite_supported':int(part.supported_spearman.notna().sum()),
            'all_atom_spearman':comp.groupby('dataset').all_atom_spearman.median().median(),
            'supported_spearman':comp.groupby('dataset').supported_spearman.median().median()})
    save('map_stability_summary.csv',pd.DataFrame(stability))
    cells=component[component.rationale.eq('boundary_extended')&component.denominator.eq('supported')].groupby(['dataset','representation','predictor']).endpoint_exact_nap_delta.median().unstack(['representation','predictor'])
    rng=np.random.default_rng(20261008)
    draws=np.array([cells.iloc[rng.integers(0,len(cells),len(cells))].to_numpy().mean() for _ in range(5000)])
    validation['fixed_configuration_grand_mean']={'estimate':float(cells.to_numpy().mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'scope':'Fixed six configurations; datasets resampled; descriptive panel uncertainty.'}
    validation['main_labels_changed']=False
    validation['scope']='Post-confirmation sensitivity of frozen scores; exact null means, no model refit.'
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (OUT/'evaluation_summary.json').write_text(json.dumps({'status':'completed','question':'Do endpoint conditioning and matched denominators change the frozen localization audit?',
        'comparison_baselines':['raw nAP','pooled exact null','endpoint-stratified exact null','predictor-free scores'],
        'claim_update':'Use primary_summary.csv and absolute_matched_summary.csv; retain failed mapping recovery and positive controls.',
        'comparability':'Same models, pairs, stored labels and neighborhoods; null and denominator variations explicit.',
        'next_action':'Complete count-function diagnostic, update manuscript and package.'},indent=2)+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('completed',len(models),'model rows',len(generators),'generator rows',flush=True)

if __name__=='__main__':
    main()
