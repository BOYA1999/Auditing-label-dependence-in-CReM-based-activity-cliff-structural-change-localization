import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import beta

from analyze_minor_nested_mapping import vector_ap
from analyze_review_revision import null_mean
from analyze_round2_revision import pair_label_arrays, rng_for, two_stage_bootstrap
from verify_endpoint_20261019 import independent_expectation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research_v2/artifacts/analysis/extension_20261019/endpoint'
ATOM = ROOT/'research_v2/artifacts/experiment/crossed_20261019/atom_scores'
PREP = ROOT/'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared'
BASE = json.loads((ROOT/'research_v2/configs/sanity_full_v1.json').read_text())
CELLS = list(itertools.product(BASE['representations'], ['xgboost','mlp']))
METRICS = ['raw_nap','endpoint_exact_nap']
SEED, DRAWS, BOOTS = 20261019, 2000, 5000


def aggregate(values, datasets):
    return np.median(np.stack([np.median(values[...,datasets == name,:],axis=(-2,-1))
        for name in sorted(set(datasets))],axis=-1),axis=-1)


def bootstrap(delta, datasets):
    component = delta.mean(axis=(-2,-1))
    names = sorted(set(datasets))
    groups = [np.flatnonzero(datasets == name) for name in names]
    population_rng = rng_for(SEED,'cross-endpoint-population')
    stochastic_rng = rng_for(SEED,'cross-endpoint-stochastic')
    outputs = np.empty((3, 2, BOOTS))
    assignments = []
    for draw in range(BOOTS):
        initializations = stochastic_rng.integers(0,3,3)
        permutations = stochastic_rng.integers(0,5,5)
        weights = np.outer(np.bincount(initializations,minlength=3)/3,
            np.bincount(permutations,minlength=5)/5)
        stochastic = np.einsum('mpcij,ij->mpc',delta,weights)
        outputs[1,:,draw] = aggregate(stochastic,datasets)
        conditional, joint = [], []
        for t in population_rng.integers(0,len(names),len(names)):
            indices = groups[t][population_rng.integers(0,len(groups[t]),len(groups[t]))]
            conditional.append(np.median(component[:,indices,:],axis=(1,2)))
            joint.append(np.median(stochastic[:,indices,:],axis=(1,2)))
        outputs[0,:,draw] = np.median(conditional,axis=0)
        outputs[2,:,draw] = np.median(joint,axis=0)
        assignments.append({'draw':draw,'initialization_indices':json.dumps(initializations.tolist()),
            'permutation_indices':json.dumps(permutations.tolist())})
    return outputs, assignments


def main():
    started = time.time()
    OUT.mkdir(parents=True,exist_ok=True)
    export_manifest = ATOM.parent/'atom_export_manifest.json'
    export = json.loads(export_manifest.read_text())
    assert export['status'] == 'pass' and export['files'] == 162 and export['permuted_models'] == 2430
    expected_files = [ATOM/f'{target}--{rep}--{pred}.npz' for target in BASE['confirmation_targets'] for rep,pred in CELLS]
    assert all(path.exists() for path in expected_files), 'Crossed atom export is incomplete; no inference started.'
    export_checks = pd.read_csv(ATOM.parent/'atom_export_validation.csv')
    assert len(export_checks) == 162 and not export_checks.duplicated(['dataset','representation','predictor']).any()
    for row in export_checks.itertuples():
        path = ATOM/f'{row.dataset}--{row.representation}--{row.predictor}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256
    molecules = pd.read_csv(PREP/'model_molecules.csv').set_index('row_id')
    counts = {row_id:Chem.MolFromSmiles(row.canonical_smiles).GetNumAtoms()
        for row_id,row in molecules.iterrows() if row.dataset in BASE['confirmation_targets']}
    pairs = pd.read_csv(PREP/'sentinel_pairs.csv')
    pairs = pairs[pairs.dataset.isin(BASE['confirmation_targets']) & pairs.component_split.eq('test')].copy()
    pairs['left_id'] = pairs.dataset+':'+pairs.left_index.astype(int).astype(str)
    pairs['right_id'] = pairs.dataset+':'+pairs.right_index.astype(int).astype(str)
    pairs = pairs[[pair_label_arrays(row,counts,'boundary_extended')[1] for row in pairs.itertuples()]]
    pairs = pairs.sort_values(['dataset','component_id']).reset_index(drop=True)
    assert len(pairs) == 617 and not pairs.duplicated(['dataset','component_id']).any()
    datasets = pairs.dataset.to_numpy()
    delta = np.empty((2,617,6,3,5))
    source_delta = np.empty_like(delta)
    reference = np.empty((2,DRAWS,617,6))
    realization_rows, checks = [], []
    for target, target_pairs in pairs.groupby('dataset',sort=True):
        loaded, lookup = [], []
        for rep,pred in CELLS:
            with np.load(ATOM/f'{target}--{rep}--{pred}.npz',allow_pickle=False) as archive:
                data = {name:archive[name] for name in archive.files}
            assert data['initializations'].tolist() == [11,23,42]
            assert data['permutations'].tolist() == [42,43,44,45,46]
            assert data['scores'].shape[:2] == (3,5) and data['observed_scores'].shape[0] == 3
            assert np.isfinite(data['scores']).all() and np.isfinite(data['observed_scores']).all()
            assert data['scores_source'].shape == data['scores'].shape
            assert data['observed_scores_source'].shape == data['observed_scores'].shape
            assert np.isfinite(data['scores_source']).all() and np.isfinite(data['observed_scores_source']).all()
            ids = data['row_ids'].astype(str)
            offsets = data['offsets']
            assert len(offsets) == len(ids)+1 and len(set(ids)) == len(ids)
            assert all(offsets[i+1]-offsets[i] == counts[row_id] for i,row_id in enumerate(ids))
            lookup.append({row_id:slice(int(offsets[i]),int(offsets[i+1])) for i,row_id in enumerate(ids)})
            loaded.append(data)
        for pair in target_pairs.itertuples():
            index = pair.Index
            lengths = [counts[pair.left_id],counts[pair.right_id]]
            labels, _ = pair_label_arrays(pair,counts,'boundary_extended')
            prevalence = labels.mean()
            random_labels = np.zeros((DRAWS,len(labels)),dtype=np.int8)
            rng = rng_for(SEED,f'crossed-endpoint:{pair.dataset}:{pair.component_id}')
            offset = 0
            for n in lengths:
                k = int(labels[offset:offset+n].sum())
                if k:
                    selected = np.argpartition(rng.random((DRAWS,n)),k-1,axis=1)[:,:k]
                    random_labels[np.arange(DRAWS)[:,None],offset+selected] = 1
                assert np.all(random_labels[:,offset:offset+n].sum(axis=1) == k)
                offset += n
            for c,((rep,pred),data,regions) in enumerate(zip(CELLS,loaded,lookup)):
                ix = np.r_[np.arange(regions[pair.left_id].start,regions[pair.left_id].stop),
                    np.arange(regions[pair.right_id].start,regions[pair.right_id].stop)]
                observed_scores = data['observed_scores'][:,ix]
                permuted_scores = data['scores'][:,:,ix]
                source_observed_scores = data['observed_scores_source'][:,ix]
                source_permuted_scores = data['scores_source'][:,:,ix]
                observed_metrics = np.empty((2,3))
                permuted_metrics = np.empty((2,3,5))
                source_observed_metrics = np.empty_like(observed_metrics)
                source_permuted_metrics = np.empty_like(permuted_metrics)
                random_observed, random_permuted = np.zeros(DRAWS), np.zeros(DRAWS)
                for variant,scores_array,metrics_array,source_scores_array,source_metrics_array in [
                    ('trained',observed_scores,observed_metrics,source_observed_scores,source_observed_metrics),
                    ('label_permuted',permuted_scores,permuted_metrics,source_permuted_scores,source_permuted_metrics)]:
                    for position in np.ndindex(scores_array.shape[:-1]):
                        scores = scores_array[position]
                        ap = vector_ap(labels,scores)[0]
                        expected = null_mean(labels,scores,lengths)
                        metrics_array[(0,)+position] = (ap-prevalence)/(1-prevalence)
                        metrics_array[(1,)+position] = (ap-expected)/(1-prevalence)
                        source_scores = source_scores_array[position]
                        source_ap = vector_ap(labels,source_scores)[0]
                        source_expected = null_mean(labels,source_scores,lengths)
                        source_metrics_array[(0,)+position] = (source_ap-prevalence)/(1-prevalence)
                        source_metrics_array[(1,)+position] = (source_ap-source_expected)/(1-prevalence)
                        random_ap = vector_ap(random_labels,scores)/(1-prevalence)
                        if variant == 'trained':
                            random_observed += random_ap/3
                        else:
                            random_permuted += random_ap/15
                        realization_rows.append({'dataset':target,'component_id':pair.component_id,
                            'representation':rep,'predictor':pred,'variant':variant,'initialization':[11,23,42][position[0]],
                            'permutation':-1 if variant == 'trained' else [42,43,44,45,46][position[1]],
                            'AP':ap,'endpoint_null_AP':expected,'prevalence':prevalence,
                            'source_AP':source_ap,'source_endpoint_null_AP':source_expected,
                            **{metric:metrics_array[(m,)+position] for m,metric in enumerate(METRICS)},
                            **{metric+'_source':source_metrics_array[(m,)+position] for m,metric in enumerate(METRICS)}})
                        if pair.Index % 50 == 0:
                            checks.append(abs(expected-independent_expectation(labels,scores,lengths)))
                delta[:,index,c] = observed_metrics[:,:,None]-permuted_metrics
                source_delta[:,index,c] = source_observed_metrics[:,:,None]-source_permuted_metrics
                reference[0,:,index,c] = random_observed-random_permuted
                reference[1,:,index,c] = reference[0,:,index,c]+delta[1,index,c].mean()-delta[0,index,c].mean()
        print('endpoint completed',target,'seconds',round(time.time()-started,1),flush=True)
    assert np.isfinite(delta).all() and np.isfinite(source_delta).all() and np.isfinite(reference).all() and max(checks) < 1e-12
    pd.DataFrame(realization_rows).to_csv(OUT/'crossed_realization_metrics.csv',index=False)
    component = delta.mean(axis=(-2,-1))
    source_component = source_delta.mean(axis=(-2,-1))
    component_rows = [{**pairs.loc[i,['dataset','component_id']].to_dict(),'representation':rep,'predictor':pred,
        **{metric+'_delta':component[m,i,c] for m,metric in enumerate(METRICS)},
        **{metric+'_source_delta':source_component[m,i,c] for m,metric in enumerate(METRICS)}}
        for i in range(len(pairs)) for c,(rep,pred) in enumerate(CELLS)]
    pd.DataFrame(component_rows).to_csv(OUT/'crossed_component_metrics.csv',index=False)
    np.savez_compressed(OUT/'crossed_paired_metric_arrays.npz',delta=delta,source_delta=source_delta,datasets=datasets.astype(str),
        component_ids=pairs.component_id.to_numpy(),metrics=np.array(METRICS),cells=np.array(CELLS),
        initializations=np.array([11,23,42]),permutations=np.array([42,43,44,45,46]))
    primary = aggregate(component,datasets)
    source_primary = aggregate(source_component,datasets)
    implementation = pd.DataFrame([{'metric':metric,'uniform_backend_estimate':primary[m],'source_compatible_estimate':source_primary[m],
        'primary_change':primary[m]-source_primary[m],
        'component_configuration_max_absolute_change':np.max(np.abs(component[m]-source_component[m])),
        'component_configuration_count_change_gt_1e_minus12':int(np.sum(np.abs(component[m]-source_component[m])>1e-12))}
        for m,metric in enumerate(METRICS)])
    implementation.to_csv(OUT/'crossed_backend_sensitivity.csv',index=False)
    implementation.to_csv(OUT/'compact_implementation_sensitivity.csv',index=False)
    boot, assignments = bootstrap(delta,datasets)
    pd.DataFrame(assignments).to_csv(OUT/'crossed_stochastic_draw_assignments.csv',index=False)
    scopes = ['conditional_population','stochastic_only','population_and_stochastic']
    pd.DataFrame({'draw':np.arange(BOOTS),**{scope+'_'+metric:boot[s,m]
        for s,scope in enumerate(scopes) for m,metric in enumerate(METRICS)}}).to_csv(OUT/'crossed_bootstrap_draws.csv',index=False)
    summaries = []
    for m,metric in enumerate(METRICS):
        frame = pd.DataFrame(component_rows)
        replay = two_stage_bootstrap(frame,metric+'_delta',100,SEED,'cross-endpoint-population')
        assert np.max(np.abs(replay-boot[0,m,:100])) < 1e-12
        for s,scope in enumerate(scopes):
            summaries.append({'metric':metric,'estimate':primary[m],'uncertainty_scope':scope,
                'ci_low':np.quantile(boot[s,m],.025),'ci_high':np.quantile(boot[s,m],.975),
                'draws':BOOTS,'datasets':27,'pairs':617,'configurations':6,'initializations':3,'permutations':5})
    pd.DataFrame(summaries).to_csv(OUT/'crossed_primary_summary.csv',index=False)
    pd.DataFrame([{'metric':metric,'estimate':primary[m],**{scope+suffix:float(np.quantile(boot[s,m],q))
        for s,scope in enumerate(scopes) for suffix,q in [('_low',.025),('_high',.975)]},
        'datasets':27,'pairs':617,'initializations':3,'permutations':5}
        for m,metric in enumerate(METRICS)]).to_csv(OUT/'compact_crossed_results.csv',index=False)
    references = aggregate(reference,datasets)
    pd.DataFrame({'draw':np.arange(DRAWS),**{metric:references[m] for m,metric in enumerate(METRICS)}}).to_csv(OUT/'crossed_structural_reference_draws.csv',index=False)
    reference_summary = []
    median_rng = rng_for(SEED,'cross-endpoint-reference-median')
    resamples = median_rng.integers(0,DRAWS,(2000,DRAWS))
    for m,metric in enumerate(METRICS):
        values = references[m]
        count = int(np.sum(values <= primary[m]))
        medians = np.median(values[resamples],axis=1)
        reference_summary.append({'metric':metric,'observed':primary[m],'reference_mean':values.mean(),
            'reference_median':np.median(values),'reference_q025':np.quantile(values,.025),'reference_q975':np.quantile(values,.975),
            'lower_tail_count':count,'draws':DRAWS,'corrected_lower_tail_fraction':(count+1)/(DRAWS+1),
            'tail_binomial_ci_low':beta.ppf(.025,count,DRAWS-count+1) if count else 0.,
            'tail_binomial_ci_high':beta.ppf(.975,count+1,DRAWS-count) if count<DRAWS else 1.,
            'mc_se_reference_median':medians.std(ddof=1),'mc_se_reference_mean':values.std(ddof=1)/np.sqrt(DRAWS),
            'first_half_median':np.median(values[:DRAWS//2]),'second_half_median':np.median(values[DRAWS//2:])})
    pd.DataFrame(reference_summary).to_csv(OUT/'crossed_structural_reference_summary.csv',index=False)
    pd.DataFrame(reference_summary)[['metric','observed','reference_median','reference_q025','reference_q975',
        'mc_se_reference_median','lower_tail_count','draws','corrected_lower_tail_fraction',
        'tail_binomial_ci_low','tail_binomial_ci_high']].to_csv(OUT/'compact_crossed_reference.csv',index=False)
    marginal = []
    for axis,ids in [('initialization',[11,23,42]),('permutation',[42,43,44,45,46])]:
        for position,identity in enumerate(ids):
            selected = delta[:,:,:,position,:].mean(axis=-1) if axis == 'initialization' else delta[:,:,:,:,position].mean(axis=-1)
            values = aggregate(selected,datasets)
            marginal.extend({'factor':axis,'value':identity,'metric':metric,'estimate':values[m]} for m,metric in enumerate(METRICS))
    pd.DataFrame(marginal).to_csv(OUT/'crossed_factor_sensitivity.csv',index=False)
    validation = {'status':'PASS','seconds':time.time()-started,'realization_rows':len(realization_rows),
        'independent_endpoint_expectation_checks':len(checks),'independent_endpoint_expectation_max_error':max(checks),
        'conditional_bootstrap_first100_matches_established_component_safe_function':True,
        'seed':SEED,'structural_reference_draws':DRAWS,'bootstrap_draws':BOOTS,
        'scope':'Original frozen 617 pairs, fixed six configurations; cross initialization x permutation sensitivity is post-confirmation.',
        'backend':'Primary: all XGBoost CPU inference and existing MLP GPU contract, including freshly reconstructed observed scores. Source-compatible historical inference and original observed CSV scores are a separate point sensitivity. No score rounding or tie merging.',
        'aggregation':'AP per realization; paired matched-initialization difference; mean initializations/permutations per component/configuration; median over components and configurations jointly within dataset; median across datasets.',
        'resampling':'Dataset and component resampling keeps six configuration rows together. Stochastic draws resample 3 initialization and 5 permutation IDs separately, shared globally across datasets/configurations and metrics; observed and permuted initializations remain paired. Joint draws reuse conditional population indices.',
        'reference':'Frozen atom scores, endpoint positive counts, shared structural labels across all models. Reference center need not be zero; reference quantiles are not population confidence intervals and tails are not model-label exchangeability p-values.',
        'limitation':'Three initializations and five permutations provide bounded stochastic sensitivity, not convergence or independent validation.'}
    (OUT/'crossed_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    paths = expected_files+[export_manifest,ATOM.parent/'atom_export_validation.csv',PREP/'model_molecules.csv',PREP/'sentinel_pairs.csv',Path(__file__),
        Path(__file__).parent/'analyze_review_revision.py',Path(__file__).parent/'analyze_minor_nested_mapping.py',
        Path(__file__).parent/'analyze_round2_revision.py',Path(__file__).parent/'verify_endpoint_20261019.py',
        ROOT/'research_v2/configs/sanity_full_v1.json',ROOT/'research_v2/artifacts/analysis/extension_20261019/PLAN.md']
    paths.append(OUT/'BACKEND_AMENDMENT.md')
    (OUT/'crossed_input_hashes.json').write_text(json.dumps({str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths},indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    print(pd.DataFrame(reference_summary).to_string(index=False),flush=True)
    print(json.dumps(validation,indent=2),flush=True)


if __name__ == '__main__':
    main()
