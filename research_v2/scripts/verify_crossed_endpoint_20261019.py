import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from analyze_round2_revision import pair_label_arrays, rng_for

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research_v2/artifacts/analysis/extension_20261019/endpoint'
ATOM = ROOT/'research_v2/artifacts/experiment/crossed_20261019/atom_scores'
PREP = ROOT/'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared'
METRICS = ['raw_nap','endpoint_exact_nap']


def direct_ap(labels, scores):
    thresholds = np.unique(scores)
    above = scores[None,:] >= thresholds[:,None]
    tied = scores[None,:] == thresholds[:,None]
    cumulative = labels @ above.T
    groups = labels @ tied.T
    return np.sum(groups*cumulative/above.sum(axis=1),axis=1)/labels.sum(axis=1)


def main():
    started = time.time()
    rows = pd.read_csv(OUT/'crossed_realization_metrics.csv')
    keys = ['dataset','component_id','representation','predictor']
    assert len(rows) == 66636 and not rows.duplicated(keys+['variant','initialization','permutation']).any()
    assert rows[rows.variant.eq('trained')].groupby(keys).size().eq(3).all()
    assert rows[rows.variant.eq('label_permuted')].groupby(keys).size().eq(15).all()
    component = pd.read_csv(OUT/'crossed_component_metrics.csv').set_index(keys)
    errors = []
    for metric in METRICS:
        for suffix,result in [('',metric+'_delta'),('_source',metric+'_source_delta')]:
            trained = rows[rows.variant.eq('trained')].groupby(keys)[metric+suffix].mean()
            permuted = rows[rows.variant.eq('label_permuted')].groupby(keys)[metric+suffix].mean()
            errors.append(float(((trained-permuted)-component[result]).abs().max()))
    boot = pd.read_csv(OUT/'crossed_bootstrap_draws.csv')
    for row in pd.read_csv(OUT/'crossed_primary_summary.csv').itertuples():
        value = component.groupby('dataset')[row.metric+'_delta'].median().median()
        errors.append(abs(value-row.estimate))
        draws = boot[row.uncertainty_scope+'_'+row.metric]
        errors += [abs(draws.quantile(.025)-row.ci_low),abs(draws.quantile(.975)-row.ci_high)]
    molecules = pd.read_csv(PREP/'model_molecules.csv').set_index('row_id')
    datasets = set(rows.dataset)
    counts = {row_id:Chem.MolFromSmiles(row.canonical_smiles).GetNumAtoms()
        for row_id,row in molecules.iterrows() if row.dataset in datasets}
    pairs = pd.read_csv(PREP/'sentinel_pairs.csv')
    pairs = pairs.merge(rows[['dataset','component_id']].drop_duplicates(),on=['dataset','component_id'],validate='one_to_one')
    pairs['left_id'] = pairs.dataset+':'+pairs.left_index.astype(int).astype(str)
    pairs['right_id'] = pairs.dataset+':'+pairs.right_index.astype(int).astype(str)
    selected_draws = [0,499,999,1499,1999]
    reference_rows = []
    for (dataset,rep,pred), group in rows.groupby(['dataset','representation','predictor'],sort=True):
        with np.load(ATOM/f'{dataset}--{rep}--{pred}.npz',allow_pickle=False) as archive:
            data = {name:archive[name] for name in ['row_ids','offsets','scores','observed_scores']}
        offsets = dict(zip(data['row_ids'],zip(data['offsets'][:-1],data['offsets'][1:])))
        for pair in pairs[pairs.dataset.eq(dataset)].itertuples():
            lengths = [counts[pair.left_id],counts[pair.right_id]]
            labels,_ = pair_label_arrays(pair,counts,'boundary_extended')
            rng = rng_for(20261019,f'crossed-endpoint:{dataset}:{pair.component_id}')
            random_labels = np.zeros((2000,len(labels)),dtype=np.int16)
            start = 0
            for n in lengths:
                k = int(labels[start:start+n].sum())
                if k:
                    selected = np.argpartition(rng.random((2000,n)),k-1,axis=1)[:,:k]
                    random_labels[np.arange(2000)[:,None],start+selected] = 1
                start += n
            selected_labels = random_labels[selected_draws]
            ix = np.r_[np.arange(*offsets[pair.left_id]),np.arange(*offsets[pair.right_id])]
            observed = data['observed_scores'][:,ix]
            permuted = data['scores'][:,:,ix].reshape(15,len(labels))
            raw = (np.mean([direct_ap(selected_labels,score) for score in observed],axis=0)
                -np.mean([direct_ap(selected_labels,score) for score in permuted],axis=0))/(1-labels.mean())
            values = component.loc[(dataset,pair.component_id,rep,pred)]
            exact = raw+values.endpoint_exact_nap_delta-values.raw_nap_delta
            reference_rows.extend({'dataset':dataset,'component_id':pair.component_id,'representation':rep,'predictor':pred,
                'draw':draw,'raw_nap':raw[j],'endpoint_exact_nap':exact[j]} for j,draw in enumerate(selected_draws))
    independent = pd.DataFrame(reference_rows).groupby(['draw','dataset'])[METRICS].median().groupby('draw').median()
    saved = pd.read_csv(OUT/'crossed_structural_reference_draws.csv').set_index('draw').loc[selected_draws]
    reference_error = float(np.max(np.abs(independent.to_numpy()-saved[METRICS].to_numpy())))
    assert max(errors) < 1e-12 and reference_error < 1e-12
    for name,expected in json.loads((OUT/'crossed_input_hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    output = {'status':'PASS','seconds':time.time()-started,'realization_rows':len(rows),
        'saved_seed_permutation_pairing_aggregation_and_bootstrap_quantile_max_error':max(errors),
        'independent_threshold_AP_full_statistic_draws':selected_draws,
        'independent_threshold_AP_full_statistic_max_error':reference_error,
        'all_recorded_input_hashes_match':True,
        'scope':'Independent threshold-based AP on five complete structural-reference draws, plus saved-row reaggregation; no new training or validation cohort.'}
    (OUT/'crossed_independent_verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
