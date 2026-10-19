import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_v2/artifacts/experiment/crossed_20261019'
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/noise'


def distance(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    if np.array_equal(rankdata(x), rankdata(y)):
        return 0.
    return 1-spearmanr(x, y).statistic


def main():
    frame = pd.read_csv(OUT / 'anchor_paired_distances.csv.gz')
    full = frame[frame.scenario.eq('full') & frame.domain.eq('supported')]
    cache, errors, relative_errors = {}, [], []
    for row in full.sample(100, random_state=20261019).itertuples():
        key = f'{row.dataset}--{row.representation}--{row.predictor}'
        if key not in cache:
            cache[key] = dict(np.load(RUN / 'atom_scores' / f'{key}.npz'))
        data = cache[key]
        i = list(data['row_ids']).index(row.row_id)
        a,b = data['offsets'][i:i+2]
        mask = data['supported_flat'][a:b]
        si,pi = [11,23,42].index(row.anchor_seed), row.permutation_id-42
        observed = data['observed_scores'][:,a:b][:,mask]
        permuted = data['scores'][:,:,a:b][:,:,mask]
        matched = distance(observed[si],permuted[si,pi])
        for name,array in [('observed',observed),('permuted',permuted[:,pi])]:
            ds = [distance(array[si],array[s]) for s in range(3) if s != si]
            noise = np.median(ds) if np.isfinite(ds).all() else np.nan
            common = np.isfinite(matched) and np.isfinite(noise)
            expected = [matched-noise if common else np.nan,matched/noise if common and noise>0 else np.nan]
            saved = [getattr(row,name+'_excess'),getattr(row,name+'_ratio')]
            assert np.allclose(expected,saved,atol=1e-10,rtol=1e-12,equal_nan=True)
            errors.extend(abs(x-y) for x,y in zip(expected,saved) if np.isfinite(x) and np.isfinite(y))
            relative_errors.extend(abs(x-y)/abs(x) for x,y in zip(expected,saved) if np.isfinite(x) and np.isfinite(y) and x!=0)
    metrics = ['matched_distance','mixed_distance','observed_noise','permuted_noise','observed_excess','permuted_excess','observed_ratio','permuted_ratio']
    component = full.groupby(['dataset','component_id','row_id','representation','predictor'])[metrics].median().groupby(['dataset','component_id']).median()
    estimate = component.groupby('dataset').median().median()
    summary = pd.read_csv(OUT / 'summary.csv')
    saved = summary[summary.scenario.eq('full') & summary.domain.eq('supported')].set_index('metric').estimate
    assert np.allclose(estimate.loc[metrics],saved.loc[metrics],rtol=0,atol=1e-14)
    backend = pd.read_csv(RUN / 'backend_difference_summary.csv')
    assert backend.pair_realizations.sum()==66636 and backend.ap_changed.sum()==105
    report = {'status':'pass','independent_scipy_anchor_rows':100,'max_independent_derived_metric_error':max(errors),'max_independent_relative_error':max(relative_errors),'reaggregated_full_supported_points_equal':True,'backend_pair_realizations':int(backend.pair_realizations.sum()),'backend_ap_changes':int(backend.ap_changed.sum()),'comparison_scope':'Frozen complete crossed grid; conditional noise analysis; no manuscript validation claimed'}
    (OUT / 'independent_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
