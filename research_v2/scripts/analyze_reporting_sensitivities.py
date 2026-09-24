import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def two_stage_bootstrap(frame, value, draws, seed, label):
    grouped = {dataset: [part[value].to_numpy(float) for _, part in group.groupby('component_id', sort=True)] for dataset, group in frame.groupby('dataset', sort=True)}
    names = sorted(grouped)
    offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], 'little')
    rng = np.random.default_rng((seed + offset) % (2**32))
    result = np.empty(draws)
    for draw in range(draws):
        values = []
        for index in rng.integers(0, len(names), len(names)):
            components = grouped[names[index]]
            selected = [components[j] for j in rng.integers(0, len(components), len(components))]
            values.append(np.median(np.concatenate(selected)))
        result[draw] = np.median(values)
    return result

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_round2_v1'
FULL = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1'
OUT = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_reporting_v1'
OUT.mkdir(parents=True, exist_ok=True)
inputs = {
    'unadjusted': BASE / 'crem_mean_unadjusted_component_deltas.csv',
    'adjusted': BASE / 'crem_mean_empirical_null_component_deltas.csv',
    'rank': FULL / 'full/sanity_explanations/atom_rank_similarity.csv',
    'evaluation': FULL / 'prepared/evaluation_molecules.csv',
    'sentinel': ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_round2_sentinel_v1/paired_selection_comparison.csv',
}
raw = pd.read_csv(inputs['unadjusted'])
adjusted = pd.read_csv(inputs['adjusted']).query("rationale == 'boundary_extended'")
evaluation = pd.read_csv(inputs['evaluation'])
eligible = evaluation.merge(raw[['dataset', 'component_id']].drop_duplicates(), on=['dataset', 'component_id']).query("component_split == 'test'")
rank = pd.read_csv(inputs['rank'])
rank = rank[(rank.explainer == 'crem_mean') & rank.matched_seed & rank.row_id.isin(eligible.row_id)]
assert len(rank) == 22212 and len(eligible) == 1234
rank_rows = []
for label, frame in [('All configurations', rank), *[(f'{a}/{p}', f) for (a, p), f in rank.groupby(['representation', 'predictor'])]]:
    medians = frame.groupby('dataset')[['spearman_supported', 'spearman_all_atoms']].median().median()
    rank_rows.append({'configuration': label, 'datasets': frame.dataset.nunique(), 'comparisons': len(frame), 'finite_supported': frame.spearman_supported.notna().sum(), 'finite_all_atoms': frame.spearman_all_atoms.notna().sum(), 'supported_spearman': medians.spearman_supported, 'all_atom_spearman': medians.spearman_all_atoms})
pd.DataFrame(rank_rows).to_csv(OUT / 'map_similarity_summary.csv', index=False)

sentinel = pd.read_csv(inputs['sentinel']).query("explainer == 'crem_mean'")
assert len(sentinel) == 3702
sentinel_rows = []
for scope, subset in [('All paired components', sentinel), ('Changed pair only', sentinel[~sentinel.same_as_original]), ('Unchanged pair only', sentinel[sentinel.same_as_original])]:
    for label, frame in [('All configurations', subset), *[(f'{a}/{p}', f) for (a, p), f in subset.groupby(['representation', 'predictor'])]]:
        delta = frame.selection_change_in_delta_nap
        row = {'scope': scope, 'configuration': label, 'datasets': frame.dataset.nunique(), 'components': len(frame[['dataset', 'component_id']].drop_duplicates()), 'rows': len(frame), 'negative_rows': int((delta < -1e-12).sum()), 'zero_rows': int(np.isclose(delta, 0, atol=1e-12, rtol=0).sum()), 'positive_rows': int((delta > 1e-12).sum()), 'target_balanced_median_change': frame.groupby('dataset').selection_change_in_delta_nap.median().median()}
        row.update({f'row_abs_q{int(q*100)}': delta.abs().quantile(q) for q in [.5, .75, .9, .95, 1]})
        sentinel_rows.append(row)
pd.DataFrame(sentinel_rows).to_csv(OUT / 'sentinel_selection_distribution.csv', index=False)

assay_rows, draws_rows = [], []
for omitted in ['CHEMBL237_EC50', 'CHEMBL237_Ki']:
    for label, frame, value in [('Unadjusted', raw, 'delta_nap'), ('Null-adjusted', adjusted, 'delta_null_adjusted_nap')]:
        frame = frame[frame.dataset != omitted]
        draws = two_stage_bootstrap(frame, value, 5000, 20260924, omitted + label)
        low, high = np.quantile(draws, [.025, .975])
        assay_rows.append({'omitted_dataset': omitted, 'estimand': label, 'datasets': frame.dataset.nunique(), 'components': len(frame[['dataset', 'component_id']].drop_duplicates()), 'estimate': frame.groupby('dataset')[value].median().median(), 'ci95_low': low, 'ci95_high': high})
        draws_rows.append(pd.DataFrame({'omitted_dataset': omitted, 'estimand': label, 'draw': np.arange(5000), 'estimate': draws}))
pd.DataFrame(assay_rows).to_csv(OUT / 'assay_leave_one_out.csv', index=False)
pd.concat(draws_rows).to_csv(OUT / 'assay_leave_one_out_bootstrap.csv', index=False)

sets = {name: set(part.canonical_smiles) for name, part in eligible.groupby('dataset')}
overlaps = [{'dataset_a': a, 'dataset_b': b, 'shared_structures': len(sets[a] & sets[b])} for i, a in enumerate(sorted(sets)) for b in sorted(sets)[i+1:] if sets[a] & sets[b]]
summary = {'scope': 'retrospective reporting sensitivities on frozen records; no retraining', 'map_aggregation': 'median of row correlations within each dataset, then median across datasets; matched seeds; undefined values excluded, not zero-filled', 'map_limit': 'descriptive cross-label comparison, without within-observed-label seed-stability reference; no causal map-response claim', 'sentinel_aggregation': 'signed target-balanced median plus unweighted component-configuration row absolute quantiles; changed-pair subset explicit', 'assay_sensitivity': 'drop either CHEMBL237 assay; two-stage dataset/component bootstrap, 5000 draws, seed 20260924; a changed analysis population, not a clustering correction', 'datasets': len(sets), 'distinct_target_ids': len({x.split('_')[0] for x in sets}), 'eligible_endpoints': len(eligible), 'distinct_structures': eligible.canonical_smiles.nunique(), 'cross_dataset_test_overlap': overlaps, 'primary_unadjusted_unchanged': float(raw.groupby('dataset').delta_nap.median().median()), 'primary_adjusted_unchanged': float(adjusted.groupby('dataset').delta_null_adjusted_nap.median().median())}
assert np.isclose(summary['primary_unadjusted_unchanged'], -0.00523271980315315)
assert np.isclose(summary['primary_adjusted_unchanged'], -0.008317338515946699)
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'input_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs.values()}, 'script_sha256': digest(Path(__file__)), 'output_sha256': {p.name: digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}, 'bootstrap_draws': 5000, 'seed': 20260924}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'pass', 'assay_sensitivities': assay_rows, 'sentinel_all_configurations': [r for r in sentinel_rows if r['configuration'] == 'All configurations']}, indent=2))
