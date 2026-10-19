import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_review_revision import null_mean
from analyze_minor_nested_mapping import vector_ap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/endpoint'
OLD = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008'
NESTED = ROOT / 'research_v2/artifacts/analysis/minor_revision_20261015/nested_mapping'


def independent_ap(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    return sum(labels[scores == score].sum() * labels[scores >= score].sum()
        / np.count_nonzero(scores >= score) for score in np.unique(scores)) / labels.sum()


def independent_expectation(labels, scores, lengths):
    labels, scores = np.asarray(labels), np.asarray(scores)
    strata = np.repeat(np.arange(len(lengths)), lengths)
    cuts = np.cumsum([0] + list(lengths))
    counts = np.array([labels[cuts[i]:cuts[i+1]].sum() for i in range(len(lengths))])
    probability = counts / np.asarray(lengths)
    moments = probability[strata, None] * probability[strata[None, :]]
    for s, n in enumerate(lengths):
        ix = np.flatnonzero(strata == s)
        moments[np.ix_(ix, ix)] = counts[s] * (counts[s]-1) / (n * (n-1)) if n > 1 else 0
    np.fill_diagonal(moments, probability[strata])
    return sum(moments[np.ix_(scores == score, scores >= score)].sum()
        / np.count_nonzero(scores >= score) for score in np.unique(scores)) / labels.sum()


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    comparisons, errors, production_errors, ap_errors = 0, [], [], []
    for lengths in [(1, 1), (1, 4), (2, 3), (3, 3), (2, 4), (4, 4)]:
        n = sum(lengths)
        patterns = [np.arange(n), np.arange(n)[::-1], np.ones(n), np.arange(n)//2,
            np.tile([2, 0, 1], (n+2)//3)[:n], np.repeat([0, 5], lengths)]
        for counts in itertools.product(*[range(n+1) for n in lengths]):
            if not 0 < sum(counts) < n:
                continue
            labels = np.concatenate([np.arange(n) < k for n, k in zip(lengths, counts)])
            assignments = [np.concatenate([np.isin(np.arange(n), ix) for n, ix in zip(lengths, choices)])
                for choices in itertools.product(*[itertools.combinations(range(n), k) for n, k in zip(lengths, counts)])]
            for scores in patterns:
                brute = np.mean([independent_ap(lab, scores) for lab in assignments])
                expected = independent_expectation(labels, scores, lengths)
                errors.append(abs(expected-brute))
                production_errors.append(abs(expected-null_mean(labels, scores, lengths)))
                ap_errors.append(float(np.max(np.abs(vector_ap(np.array(assignments),scores)
                    - np.array([independent_ap(lab,scores) for lab in assignments])))))
                comparisons += 1
    assert max(errors) < 1e-12 and max(production_errors) < 1e-12 and max(ap_errors) < 1e-12
    hashes = {}
    for folder, filename in [(NESTED, 'input_provenance.json'), (NESTED, 'manifest.json')]:
        for name, expected in json.loads((folder/filename).read_text()).items():
            path = ROOT / name if filename == 'input_provenance.json' else folder/name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == expected, str(path)
            hashes[str(path.relative_to(ROOT))] = actual
    original = pd.read_csv(OLD/'component_metrics.csv')
    original = original[original.denominator.eq('supported')]
    mapping = pd.read_csv(OLD/'mcs_mapping_ambiguity.csv')
    failed = set(map(tuple, mapping.loc[~mapping.enumeration_eligible, ['dataset','component_id']].to_numpy()))
    historical = []
    expected = pd.read_csv(ROOT/'research_v2/artifacts/analysis/xai_sanity_cliffs_rereview_20261008/mapping_exclusion_primary_summary.csv')
    for row in expected.itertuples():
        frame = original[original.rationale.eq(row.rationale)]
        if row.population == 'reproduced_only':
            frame = frame[[tuple(k) not in failed for k in frame[['dataset','component_id']].to_numpy()]]
        value = frame.groupby('dataset')[row.metric+'_delta'].median().median()
        assert abs(value-row.estimate) < 1e-12
        assert frame[['dataset','component_id']].drop_duplicates().shape[0] == row.components
        historical.append({'population':row.population,'rationale':row.rationale,'metric':row.metric,
            'pairs':row.components,'estimate':value,'saved_estimate':row.estimate})
    candidates = pd.read_csv(NESTED/'mapping_candidate_paired_metrics.csv')
    seeds = pd.read_csv(NESTED/'mapping_candidate_seed_metrics.csv')
    scenarios = pd.read_csv(NESTED/'mapping_scenario_assignments.csv')
    keys = ['dataset','component_id','rationale','representation','predictor']
    candidate_errors = []
    for row in pd.read_csv(NESTED/'mapping_primary_sensitivity.csv').itertuples():
        source = original[original.rationale.eq(row.rationale)].copy()
        value = row.metric+'_delta'
        part = candidates[candidates.rationale.eq(row.rationale) & candidates.metric.eq(row.metric)]
        means = seeds[seeds.rationale.eq(row.rationale)].groupby(keys+['candidate_id'])
        delta = means[row.metric+'_trained'].mean()-means[row.metric+'_label_permuted'].mean()
        candidate_errors.append(float((delta-part.set_index(keys+['candidate_id']).paired_delta).abs().max()))
        for scenario, operation in [('conservative_lower_envelope','min'),('conservative_upper_envelope','max')]:
            replacement = part.groupby(keys).paired_delta.agg(operation).rename('replacement')
            merged = source.join(replacement,on=keys)
            merged[value] = merged.replacement.fillna(merged[value])
            candidate_errors.append(abs(merged.groupby('dataset')[value].median().median()-getattr(row,scenario)))
        for scenario in ['lexicographic_first','pair_median_minimum','pair_median_maximum']:
            selected = scenarios[scenarios.rationale.eq(row.rationale) & scenarios.metric.eq(row.metric) & scenarios.scenario.eq(scenario)]
            replacement = part.merge(selected[['dataset','component_id','candidate_id']],on=['dataset','component_id','candidate_id'])
            merged = source.merge(replacement[keys+['paired_delta']],on=keys,how='left',validate='one_to_one')
            merged[value] = merged.paired_delta.fillna(merged[value])
            candidate_errors.append(abs(merged.groupby('dataset')[value].median().median()-getattr(row,scenario)))
    assert max(candidate_errors) < 1e-12
    reference = pd.read_csv(NESTED/'nested_reference_draws.csv')
    reference_checks = []
    for row in pd.read_csv(NESTED/'nested_reference_summary.csv').itertuples():
        values = reference[row.metric].to_numpy()
        calculated = {'reference_mean':values.mean(),'reference_median':np.median(values),
            'reference_q025':np.quantile(values,.025),'reference_q975':np.quantile(values,.975),
            'lower_tail_count':int(np.sum(values <= row.observed))}
        assert all(abs(value-getattr(row,key)) < 1e-12 for key,value in calculated.items())
        reference_checks.append({'metric':row.metric,**calculated})
    output = {'status':'PASS','seconds':time.time()-started,
        'scope':'Independent analytical small-list verification and frozen CSV reaggregation, not new independent biological/model evidence.',
        'exhaustive_score_count_cases':comparisons,'independent_expectation_vs_enumeration_max_error':max(errors),
        'independent_expectation_vs_production_max_error':max(production_errors),'verified_historical_fingerprints':len(hashes),
        'independent_AP_vs_vectorized_production_max_error':max(ap_errors),
        'historical_populations':historical,'S43_saved_draws_reaggregated':reference_checks,
        'S44_seed_pairing_scenarios_and_envelopes_max_error':max(candidate_errors),
        'bootstrap_audit':'Existing two_stage_bootstrap resamples datasets and components within dataset, retaining six configuration rows together; metrics share draws within each rationale. Full/excluded populations use different draw streams.',
        'remaining_scope':'S43 applies to original 617 boundary pairs only. New crossed predictions require fresh per-realization endpoint expectations and a full-statistic reference. Existing 611/575 exclusion, 581 strict, and matched-generator point results are complete.',
        'historical_pooled_null_uses':['Adjusted columns of S1-S4 target/configuration/row/primary summaries','Adjusted S6 historical heterogeneity','S8 competence-qualified localization and competence associations','S10 structural-label sensitivity','S12 generator-only summaries','Adjusted historical chronology rows in S16'],
        'historical_raw_not_endpoint_exact':['S17-S18 ordered permutation sensitivity is raw nAP; pooled there means pooling configurations, not pooled random-label adjustment','S19 rebuilt-split sensitivity is raw nAP','S7 contains predictor-competence metrics, not a localization null adjustment'],
        'reference_caveat':'Exact row expectation is zero under structural-label randomization; nested median reference need not be zero. Not a model-label exchangeability test or population confidence interval.'}
    (OUT/'historical_verification.json').write_text(json.dumps(output,indent=2)+'\n')
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ['analyze_review_revision.py','analyze_round2_revision.py']:
        path=Path(__file__).parent/name
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'historical_input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
