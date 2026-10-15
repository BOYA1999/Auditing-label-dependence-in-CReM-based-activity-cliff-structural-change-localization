import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/analysis/minor_revision_20261015/nested_mapping'
OLD = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008'


def main():
    checks = {}
    original = pd.read_csv(OLD / 'component_metrics.csv')
    original = original[original.denominator.eq('supported')]
    summary = pd.read_csv(OUT / 'mapping_primary_sensitivity.csv')
    candidates = pd.read_csv(OUT / 'mapping_candidate_paired_metrics.csv')
    seeds = pd.read_csv(OUT / 'mapping_candidate_seed_metrics.csv')
    keys = ['dataset', 'component_id', 'rationale', 'representation', 'predictor']
    assignments = pd.read_csv(OUT / 'mapping_scenario_assignments.csv')
    errors = []
    for row in summary.itertuples():
        source = original[original.rationale.eq(row.rationale)].copy()
        value = row.metric + '_delta'
        estimate = source.groupby('dataset')[value].median().median()
        errors.append(abs(estimate-row.stored_mapping_estimate))
        part = candidates[candidates.rationale.eq(row.rationale) & candidates.metric.eq(row.metric)]
        for name, operation in [('conservative_lower_envelope','min'),('conservative_upper_envelope','max')]:
            replacement = part.groupby(keys).paired_delta.agg(operation).rename('replacement').reset_index()
            merged = source.merge(replacement,on=keys,how='left',validate='one_to_one')
            merged[value] = merged.replacement.fillna(merged[value])
            estimate = merged.groupby('dataset')[value].median().median()
            errors.append(abs(estimate-getattr(row,name)))
        for name in ['lexicographic_first','pair_median_minimum','pair_median_maximum']:
            selection = assignments[assignments.rationale.eq(row.rationale) & assignments.metric.eq(row.metric) & assignments.scenario.eq(name)]
            replacement = part.merge(selection[['dataset','component_id','candidate_id']],on=['dataset','component_id','candidate_id'],validate='many_to_one')
            merged = source.merge(replacement[keys+['paired_delta']],on=keys,how='left',validate='one_to_one')
            merged[value] = merged.paired_delta.fillna(merged[value])
            estimate = merged.groupby('dataset')[value].median().median()
            errors.append(abs(estimate-getattr(row,name)))
        grouped = seeds[seeds.rationale.eq(row.rationale)].groupby(keys+['candidate_id'])
        actual = grouped[row.metric+'_trained'].mean()-grouped[row.metric+'_label_permuted'].mean()
        saved = part.set_index(keys+['candidate_id']).paired_delta
        errors.append(float((actual-saved).abs().max()))
    checks['saved_mapping_rows_and_scenario_recalculation_max_error'] = max(errors)
    assert max(errors)<1e-12
    reference = pd.read_csv(OUT/'nested_reference_draws.csv')
    reference_summary = pd.read_csv(OUT/'nested_reference_summary.csv')
    for row in reference_summary.itertuples():
        values = reference[row.metric].to_numpy()
        assert len(values)==row.draws
        assert int(np.sum(values<=row.observed))==row.lower_tail_count
        assert abs(np.median(values)-row.reference_median)<1e-12
        assert abs(values.mean()-row.reference_mean)<1e-12
        assert abs(np.quantile(values,.025)-row.reference_q025)<1e-12
        assert abs(np.quantile(values,.975)-row.reference_q975)<1e-12
    checks['saved_reference_summary_recomputed'] = True
    coverage = pd.read_csv(OUT/'mapping_candidate_coverage.csv')
    checks['distinct_pair_label_vectors_by_rationale'] = coverage.groupby('rationale').size().to_dict()
    checks['all_candidates_evaluable'] = bool(coverage.valid.all())
    checks['all_77_pairs_have_stored_candidate'] = bool(candidates[candidates.stored_candidate].groupby('rationale').apply(
        lambda x:x[['dataset','component_id']].drop_duplicates().shape[0],include_groups=False).eq(77).all())
    assert checks['all_77_pairs_have_stored_candidate']
    checks['status'] = 'PASS'
    (OUT/'independent_csv_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    inputs = json.loads((OUT/'input_provenance.json').read_text())
    inputs.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),OLD/'component_metrics.csv']})
    (OUT/'input_provenance.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
        if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
