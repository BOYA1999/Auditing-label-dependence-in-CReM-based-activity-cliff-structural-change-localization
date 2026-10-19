import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_v2/artifacts/experiment/crossed_20261019'
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/noise'
SEEDS = [11, 23, 42]
METRICS = ['matched_distance', 'mixed_distance', 'observed_noise', 'permuted_noise', 'observed_excess', 'permuted_excess', 'observed_ratio', 'permuted_ratio']


def distance_matrix(x):
    if x.shape[1] < 2:
        return np.full((len(x), len(x)), np.nan)
    ranks = rankdata(x, axis=1)
    ranks -= ranks.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.square(ranks).sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        correlations = ranks @ ranks.T / np.outer(norms, norms)
    distances = 1 - np.clip(correlations, -1, 1)
    identical_ranks = (ranks[:, None, :] == ranks[None, :, :]).all(axis=2)
    distances[identical_ranks & np.outer(norms > 0, norms > 0)] = 0.
    return distances


def strict_median(values):
    return float(np.median(values)) if np.isfinite(values).all() else np.nan


def bootstrap(component, draws=5000):
    targets = sorted(component.dataset.unique())
    rng = np.random.default_rng(20261019)
    sampled_targets = rng.integers(0, len(targets), size=(draws, len(targets)))
    target_values = np.full((draws, len(targets), len(METRICS)), np.nan)
    for index, target in enumerate(targets):
        rows, slots = np.where(sampled_targets == index)
        data = component[component.dataset.eq(target)].sort_values('component_id')[METRICS].to_numpy()
        selected = rng.integers(0, len(data), size=(len(rows), len(data)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            target_values[rows, slots] = np.nanmedian(data[selected], axis=1)
    return np.nanmedian(target_values, axis=1)


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True, parents=True)
    export = json.loads((RUN / 'atom_export_manifest.json').read_text())
    assert export['status'] == 'pass' and 'XGBoost CPU' in export['backend_primary']
    verified = pd.read_csv(RUN / 'atom_export_validation.csv').set_index(['dataset','representation','predictor']).sha256
    rng = np.random.default_rng(31)
    test = rng.integers(0, 5, size=(12, 17)).astype(float)
    test[0], test[1], test[2] = 1., test[3], 3*test[3]+2
    test_distances = distance_matrix(test)
    assert np.isnan(test_distances[0]).all() and test_distances[1, 2] == 0.
    errors = [abs(test_distances[a,b]-(1-spearmanr(test[a],test[b]).statistic)) for a in range(1,12) for b in range(1,12)]
    assert max(errors) < 1e-14 and np.isnan(distance_matrix(np.ones((3,1)))).all()
    pairs = pd.read_csv(RUN / 'consolidated_pair_explanations.csv')
    pairs = pairs[pairs.component_split.eq('test')][['dataset', 'component_id', 'left_index', 'right_index']].drop_duplicates()
    components = {f'{p.dataset}:{int(i)}': int(p.component_id) for p in pairs.itertuples() for i in [p.left_index, p.right_index]}
    records = []
    source_hashes = {}
    for path in sorted((RUN / 'atom_scores').glob('*.npz')):
        dataset, representation, predictor = path.stem.split('--')
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert source_hashes[path.name] == verified.loc[(dataset,representation,predictor)]
        data = np.load(path)
        assert data['initializations'].tolist() == SEEDS
        for backend in ['', '_source']:
            for mol, ident in enumerate(data['row_ids']):
                a, b = data['offsets'][mol:mol+2]
                observed, permuted = data['observed_scores'+backend][:, a:b], data['scores'+backend][:, :, a:b]
                combined = np.vstack([observed, permuted.reshape(15, b-a)])
                for domain, mask in [('supported', data['supported_flat'][a:b]), ('all_atoms', np.ones(b-a, bool))]:
                    distances = distance_matrix(combined[:, mask])
                    for omitted in ([None] if backend else [None, 0, 1, 2]):
                        retained = [s for s in range(3) if s != omitted]
                        scenario = ('source_compatible' if backend else 'full') if omitted is None else f'leave_seed_{SEEDS[omitted]}'
                        for anchor in retained:
                            others = [s for s in retained if s != anchor]
                            observed_noise = strict_median(distances[anchor, others])
                            for perm_index, permutation in enumerate(data['permutations']):
                                perm_anchor = 3 + anchor*5 + perm_index
                                perm_others = [3 + s*5 + perm_index for s in others]
                                permuted_noise = strict_median(distances[perm_anchor, perm_others])
                                matched = float(distances[anchor, perm_anchor])
                                row = {'dataset': dataset, 'component_id': components[str(ident)], 'row_id': str(ident), 'representation': representation, 'predictor': predictor, 'domain': domain, 'scenario': scenario, 'anchor_seed': SEEDS[anchor], 'permutation_id': int(permutation), 'n_atoms': int(mask.sum()), 'n_reference_comparisons': len(others), 'matched_distance': matched, 'mixed_distance': strict_median(distances[anchor, perm_others])}
                                for label, noise in [('observed', observed_noise), ('permuted', permuted_noise)]:
                                    common = np.isfinite(matched) and np.isfinite(noise)
                                    row[label+'_noise'] = noise if common else np.nan
                                    row[label+'_excess'] = matched-noise if common else np.nan
                                    row[label+'_ratio'] = matched/noise if common and noise > 0 else np.nan
                                    row[label+'_zero_denominator'] = bool(common and noise == 0)
                                    row[label+'_undefined_reference'] = bool(not np.isfinite(noise))
                                records.append(row)
    frame = pd.DataFrame(records)
    assert len(source_hashes) == 162 and len(pairs) == 617
    assert frame['matched_distance'].dropna().between(0,2).all()
    frame.to_csv(OUT / 'anchor_paired_distances.csv.gz', index=False)
    missing = frame.groupby(['scenario', 'domain']).agg(rows=('row_id', 'size'), matched_defined=('matched_distance', 'count'), observed_ratio_defined=('observed_ratio', 'count'), permuted_ratio_defined=('permuted_ratio', 'count'), observed_zero_denominator=('observed_zero_denominator', 'sum'), permuted_zero_denominator=('permuted_zero_denominator', 'sum'), observed_reference_undefined=('observed_undefined_reference', 'sum'), permuted_reference_undefined=('permuted_undefined_reference', 'sum'))
    missing.to_csv(OUT / 'missingness.csv')
    summaries, molecule_rows, draw_tables = [], [], []
    selections = [(name, part) for name, part in frame.groupby('scenario')]
    selections += [(f'anchor_seed_{seed}', frame[frame.scenario.eq('full') & frame.anchor_seed.eq(seed)]) for seed in SEEDS]
    for scenario, selected in selections:
        for domain, part in selected.groupby('domain'):
            molecule = part.groupby(['dataset', 'component_id', 'row_id', 'representation', 'predictor'], as_index=False)[METRICS].median()
            molecule['scenario'], molecule['domain'] = scenario, domain
            molecule_rows.append(molecule)
            component = molecule.groupby(['dataset', 'component_id'], as_index=False)[METRICS].median()
            target = component.groupby('dataset')[METRICS].median()
            distribution = bootstrap(component)
            draws = pd.DataFrame(distribution, columns=METRICS)
            draws['scenario'], draws['domain'], draws['draw'] = scenario, domain, np.arange(len(draws))
            draw_tables.append(draws)
            for col, metric in enumerate(METRICS):
                summaries.append({'scenario': scenario, 'domain': domain, 'metric': metric, 'estimate': float(target[metric].median()), 'ci_low': float(np.nanquantile(distribution[:, col], .025)), 'ci_high': float(np.nanquantile(distribution[:, col], .975)), 'datasets_defined': int(target[metric].notna().sum()), 'components_defined': int(component[metric].notna().sum()), 'molecule_configuration_rows_defined': int(molecule[metric].notna().sum())})
            print(scenario, domain, 'complete', flush=True)
    pd.DataFrame(summaries).to_csv(OUT / 'summary.csv', index=False)
    pd.concat(molecule_rows).to_csv(OUT / 'molecule_configuration_summary.csv.gz', index=False)
    pd.concat(draw_tables).to_csv(OUT / 'bootstrap_draws.csv.gz', index=False)
    summary = {'status': 'pass', 'elapsed_seconds': time.time()-start, 'source_hashes': source_hashes, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'n_pair_components': len(pairs), 'n_anchor_rows': len(frame), 'bootstrap_draws': 5000, 'bootstrap_seed': 20261019, 'aggregation': 'Within molecule/configuration median over permutation and anchor; median over endpoint/configuration within component; median components within dataset; median datasets.', 'reference_definition': 'Require every other retained initialization correlation defined. Matched common-evaluable rows for each noise comparison; ratio additionally requires strictly positive denominator. No epsilon or imputation.', 'uncertainty': 'Conditional shared target/component resampling; all six configurations and dependent seed/permutation comparisons retained as a block. Three initialization seeds do not establish convergence.', 'claim_update': 'Descriptive initialization-normalized map distances; no biochemical attribution accuracy or significance interpretation of ratios.'}
    (OUT / 'manifest.json').write_text(json.dumps(summary, indent=2))
    (OUT / 'validation.json').write_text(json.dumps({'status':'pass','independent_scipy_rank_distance_max_error':max(errors),'constant_vectors_undefined':True,'identical_rank_distance_exact_zero':True,'insufficient_atoms_undefined':True,'distance_bounds':[0,2],'npz_files':len(source_hashes),'components':len(pairs)},indent=2))


if __name__ == '__main__':
    main()
