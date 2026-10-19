import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import rankdata

from analyze_round2_revision import rng_for
from verify_endpoint_20261019 import independent_ap, independent_expectation

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research_v2/artifacts/analysis/extension_20261019/brics'
PREP = ROOT/'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared'
METRICS = ['raw_nap','endpoint_exact_nap']


def main():
    started = time.time()
    population = pd.read_csv(OUT/'frozen_pair_population.csv')
    pairs = pd.read_csv(PREP/'sentinel_pairs.csv').merge(population,
        on=['dataset','component_id','left_index','right_index'],validate='one_to_one')
    pairs['left_id'] = pairs.dataset+':'+pairs.left_index.astype(int).astype(str)
    pairs['right_id'] = pairs.dataset+':'+pairs.right_index.astype(int).astype(str)
    molecules = pd.read_csv(PREP/'model_molecules.csv').set_index('row_id')
    budgets = pd.read_csv(OUT/'matched_budgets.csv')
    products = pd.read_csv(OUT/'matched_scored_products.csv')
    assert not budgets.duplicated(['row_id','mask_id']).any()
    actual = products.groupby(['row_id','mask_id','generator']).size().unstack('generator')
    expected = budgets.set_index(['row_id','mask_id']).matched_n
    assert all(actual[generator].eq(expected).all() for generator in ['crem','brics'])
    assert expected.between(1,4).all()
    for _,part in products.groupby(['row_id','mask_id']):
        assert part.mask_atoms.nunique() == 1 and part.mask_size.nunique() == 1
    mask_counts = budgets.groupby('row_id').size()
    endpoints = []
    for pair in pairs.itertuples():
        counts = [int(mask_counts.get(row_id,0)) for row_id in [pair.left_id,pair.right_id]]
        assert pair.common_both_endpoints == all(n > 0 for n in counts)
        for side,row_id,n in zip(['left','right'],[pair.left_id,pair.right_id],counts):
            n_atoms = Chem.MolFromSmiles(molecules.loc[row_id,'canonical_smiles']).GetNumAtoms()
            selected = products[(products.row_id == row_id) & products.generator.eq('crem')].drop_duplicates('mask_id')
            incidence = np.array([np.isin(np.arange(n_atoms),json.loads(mask)) for mask in selected.mask_atoms],dtype=int).T
            classes = len(np.unique(incidence,axis=0)) if n else 1
            endpoints.append({'dataset':pair.dataset,'component_id':pair.component_id,'side':side,'row_id':row_id,
                'common_both_endpoints':pair.common_both_endpoints,'atoms':n_atoms,'common_masks':n,
                'atom_incidence_classes':classes,'supported_atoms':int(incidence.any(axis=1).sum()) if n else 0})
    endpoints = pd.DataFrame(endpoints)
    assert len(endpoints) == 1234 and endpoints.row_id.nunique() == 1234
    endpoints.to_csv(OUT/'independent_endpoint_information.csv',index=False)
    saved = pd.read_csv(OUT/'pair_metrics.csv')
    keys = ['dataset','component_id','representation','predictor','generator','variant','seed']
    saved_lookup = saved.set_index(keys)
    errors, expectation_errors, all_atom_positive, all_atom_count = [], [], 0, 0
    realized, paired = [], []
    paths = sorted((OUT/'atom_scores').glob('*.npz'))
    assert len(paths) == 162
    for path in paths:
        dataset,rep,pred = path.stem.split('--')
        with np.load(path,allow_pickle=False) as archive:
            data = {name:archive[name] for name in archive.files}
        assert data['generators'].tolist() == ['crem','brics']
        assert data['variants'].tolist() == ['trained','label_permuted'] and data['seeds'].tolist() == [11,23,42]
        scores = data['scores']
        assert scores.shape[:3] == (2,2,3) and np.isfinite(scores).all() and np.min(scores) >= 0
        all_atom_positive += int(np.count_nonzero(scores))
        all_atom_count += int(scores.size)
        regions = dict(zip(data['row_ids'],zip(data['offsets'][:-1],data['offsets'][1:])))
        for pair in pairs[pairs.dataset.eq(dataset)].itertuples():
            left,right = regions[pair.left_id],regions[pair.right_id]
            lengths = [left[1]-left[0],right[1]-right[0]]
            ix = np.r_[np.arange(*left),np.arange(*right)]
            labels = np.r_[np.isin(np.arange(lengths[0]),json.loads(pair.rationale_left)),
                np.isin(np.arange(lengths[1]),json.loads(pair.rationale_right))]
            values = np.empty((2,2,3,2))
            for gi,generator in enumerate(['crem','brics']):
                for si,seed in enumerate([11,23,42]):
                    for vi,variant in enumerate(['trained','label_permuted']):
                        score = scores[gi,vi,si,ix]
                        ap = independent_ap(labels,score)
                        null = independent_expectation(labels,score,lengths)
                        values[gi,vi,si] = [(ap-labels.mean())/(1-labels.mean()),(ap-null)/(1-labels.mean())]
                        record = saved_lookup.loc[(dataset,pair.component_id,rep,pred,generator,variant,seed)]
                        errors.extend(abs(values[gi,vi,si,m]-record[metric]) for m,metric in enumerate(METRICS))
                        expectation_errors.append(abs(null-record.endpoint_null_ap))
                        realized.append({'dataset':dataset,'component_id':pair.component_id,'representation':rep,'predictor':pred,
                            'generator':generator,'variant':variant,'seed':seed,'common_both_endpoints':pair.common_both_endpoints,
                            **{metric:values[gi,vi,si,m] for m,metric in enumerate(METRICS)}})
                    observed,permuted = scores[gi,0,si,ix],scores[gi,1,si,ix]
                    paired.append({'dataset':dataset,'component_id':pair.component_id,'representation':rep,'predictor':pred,
                        'generator':generator,'seed':seed,'common_both_endpoints':pair.common_both_endpoints,
                        'left_common_masks':int(mask_counts.get(pair.left_id,0)), 'right_common_masks':int(mask_counts.get(pair.right_id,0)),
                        'observed_all_zero':bool(np.all(observed == 0)), 'permuted_all_zero':bool(np.all(permuted == 0)),
                        'pair_rank_changed':not np.array_equal(rankdata(observed),rankdata(permuted)),
                        'left_rank_changed':not np.array_equal(rankdata(observed[:lengths[0]]),rankdata(permuted[:lengths[0]])),
                        'right_rank_changed':not np.array_equal(rankdata(observed[lengths[0]:]),rankdata(permuted[lengths[0]:])),
                        'amplitudes_changed':not np.array_equal(observed,permuted),
                        **{metric+'_delta':values[gi,0,si,m]-values[gi,1,si,m] for m,metric in enumerate(METRICS)}})
    assert len(realized) == len(saved) == 44424 and max(errors) < 1e-12 and max(expectation_errors) < 1e-12
    frame = pd.DataFrame(realized)
    paired = pd.DataFrame(paired)
    paired.to_csv(OUT/'independent_pair_seed_information.csv',index=False)
    groupkeys = keys[:5]+['common_both_endpoints']
    mean = frame.groupby(groupkeys+['variant'])[METRICS].mean().unstack('variant')
    effects = pd.DataFrame({metric:mean[(metric,'trained')]-mean[(metric,'label_permuted')] for metric in METRICS}).reset_index()
    previous = pd.read_csv(OUT/'paired_label_effects.csv').set_index(groupkeys)
    effect_error = float(np.max(np.abs((effects.set_index(groupkeys)[METRICS]-previous[METRICS]).to_numpy())))
    effects.to_csv(OUT/'independent_paired_label_effects.csv',index=False)
    wide = effects.pivot(index=groupkeys[:4]+['common_both_endpoints'],columns='generator',values=METRICS)
    contrast = wide.index.to_frame(index=False)
    columns = []
    for metric in METRICS:
        for effect in ['crem','brics','brics_minus_crem']:
            name = metric+'__'+effect
            contrast[name] = (wide[(metric,'brics')]-wide[(metric,'crem')]).to_numpy() if effect == 'brics_minus_crem' else wide[(metric,effect)].to_numpy()
            columns.append(name)
    saved_boot = pd.read_csv(OUT/'paired_bootstrap_draws.csv.gz')
    summary = pd.read_csv(OUT/'comparison_summary.csv')
    diagnostic, rank_summary, bootstrap_errors = [], [], []
    for population,sub in [('full617',contrast),('common272',contrast[contrast.common_both_endpoints])]:
        groups = [[part[columns].to_numpy() for _,part in target.groupby('component_id',sort=True)]
            for _,target in sub.groupby('dataset',sort=True)]
        rng = rng_for(20261019,'brics-'+population)
        draws = np.empty((5000,6))
        for draw in range(5000):
            target_values = []
            for t in rng.integers(0,len(groups),len(groups)):
                chosen = np.concatenate([groups[t][i] for i in rng.integers(0,len(groups[t]),len(groups[t]))])
                target_values.append(np.median(chosen,axis=0))
            draws[draw] = np.median(target_values,axis=0)
        for ci,name in enumerate(columns):
            metric,effect = name.split('__')
            actual = saved_boot[(saved_boot.population == population) & (saved_boot.metric == metric) & (saved_boot.effect == effect)].sort_values('draw').value.to_numpy()
            bootstrap_errors.append(float(np.max(np.abs(draws[:,ci]-actual))))
            record = summary[(summary.population == population) & (summary.metric == metric) & (summary.effect == effect)].iloc[0]
            estimate = sub.groupby('dataset')[name].median().median()
            assert abs(estimate-record.point) < 1e-12 and sub.dataset.nunique() == record.datasets
            assert sub[['dataset','component_id']].drop_duplicates().shape[0] == record.pairs
            nonzero = sub[name].abs() > 1e-12
            diagnostic.append({'population':population,'metric':metric,'effect':effect,'component_configurations':len(sub),
                'nonzero_component_configurations':int(nonzero.sum()),'nonzero_fraction':float(nonzero.mean()),
                'pairs_any_nonzero':sub.loc[nonzero,['dataset','component_id']].drop_duplicates().shape[0],
                'datasets_any_nonzero':sub.loc[nonzero,'dataset'].nunique(),
                'datasets_nonzero_median':int((sub.groupby('dataset')[name].median().abs()>1e-12).sum()),
                'maximum_absolute_effect':sub[name].abs().max(),'arithmetic_mean_component_configuration_effect':sub[name].mean(),
                'target_balanced_median':estimate,'bootstrap_nonzero_draws':int(np.sum(np.abs(draws[:,ci])>1e-12))})
        scope = paired if population == 'full617' else paired[paired.common_both_endpoints]
        for generator,part in scope.groupby('generator'):
            rank_summary.append({'population':population,'generator':generator,'pair_seed_configuration_rows':len(part),
                **{column:int(part[column].sum()) for column in ['observed_all_zero','permuted_all_zero','pair_rank_changed','left_rank_changed','right_rank_changed','amplitudes_changed']},
                **{metric+'_nonzero_seed_delta':int((part[metric+'_delta'].abs()>1e-12).sum()) for metric in METRICS}})
    assert effect_error < 1e-12 and max(bootstrap_errors) < 1e-12
    pd.DataFrame(diagnostic).to_csv(OUT/'independent_metric_information.csv',index=False)
    pd.DataFrame(rank_summary).to_csv(OUT/'independent_rank_information.csv',index=False)
    pair_masks = endpoints.pivot(index=['dataset','component_id','common_both_endpoints'],columns='side',values='common_masks').reset_index()
    output = {'status':'PASS_WITH_INFORMATION_LIMITATION','seconds':time.time()-started,
        'pair_realization_rows':len(frame),'NPZ_files':len(paths),'model_fitting':'none',
        'independent_AP_metric_max_error':max(errors),'independent_endpoint_expectation_max_error':max(expectation_errors),
        'seed_mean_paired_effect_max_error':effect_error,'all_5000_shared_component_bootstrap_draws_max_error':max(bootstrap_errors),
        'finite_nonnegative_atom_values':all_atom_count,'strictly_positive_atom_values':all_atom_positive,
        'endpoint_common_mask_count_distribution':{str(k):int(v) for k,v in endpoints.common_masks.value_counts().sort_index().items()},
        'endpoint_incidence_class_distribution':{str(k):int(v) for k,v in endpoints.atom_incidence_classes.value_counts().sort_index().items()},
        'pair_support_categories':{'both':int(((pair_masks.left>0)&(pair_masks.right>0)).sum()),
            'one':int(((pair_masks.left>0)^(pair_masks.right>0)).sum()),'neither':int(((pair_masks.left==0)&(pair_masks.right==0)).sum())},
        'common_pair_mask_categories':{'one_each':int(((pair_masks.left==1)&(pair_masks.right==1)).sum()),
            'multiple_at_least_one_endpoint':int(((pair_masks.left>0)&(pair_masks.right>0)&((pair_masks.left>1)|(pair_masks.right>1))).sum())},
        'common_pairs':272,'common_datasets':int(pairs.loc[pairs.common_both_endpoints,'dataset'].nunique()),
        'scope':'All NPZ scores independently rescored using threshold AP and a second-moment endpoint expectation; seed means precede paired and generator differences, joint component/configuration median within dataset then median across datasets.',
        'interpretation':'Zero nested medians and degenerate bootstrap summaries do not establish equivalence, universal label invariance or absence of local effects. Sparse common masks, shared support/ties, and nonlinear median aggregation limit discrimination.',
        'nonzero_reporting_threshold':1e-12,'threshold_scope':'Floating-point reporting only; scores and ties are never modified.',
        'input_hashes':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths+[Path(__file__),
            Path(__file__).parent/'verify_endpoint_20261019.py',OUT/'pair_metrics.csv',OUT/'paired_label_effects.csv',
            OUT/'comparison_summary.csv',OUT/'matched_budgets.csv',OUT/'matched_scored_products.csv',OUT/'frozen_pair_population.csv',
            OUT/'paired_bootstrap_draws.csv.gz']}}
    (OUT/'independent_verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k != 'input_hashes'},indent=2),flush=True)
    print(pd.DataFrame(diagnostic).to_string(index=False),flush=True)
    print(pd.DataFrame(rank_summary).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
