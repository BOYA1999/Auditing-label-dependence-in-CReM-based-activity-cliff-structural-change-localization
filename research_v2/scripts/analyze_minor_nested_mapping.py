import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from scipy.stats import beta

from analyze_review_revision import null_mean
from analyze_round2_revision import atom_array_lookup, grouped_average_precision, load_atom_scores, pair_label_arrays, rng_for

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/analysis/minor_revision_20261015/nested_mapping'
OLD = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008'
PREP = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared'
CFG = json.loads((ROOT / 'research_v2/configs/round2_revision_v1.json').read_text())
BASE = json.loads((ROOT / CFG['source_config']).read_text())
CELLS = list(itertools.product(BASE['representations'], ['xgboost', 'mlp']))
SEEDS = BASE['seeds']
VARIANTS = ['trained', 'label_permuted']
METRICS = ['raw_nap', 'endpoint_exact_nap']
DRAWS = 2000
SEED = 20261015


def save(name, frame):
    pd.DataFrame(frame).to_csv(OUT / name, index=False)


def vector_ap(labels, scores):
    labels = np.atleast_2d(labels).astype(np.int16)
    order = np.argsort(-scores, kind='stable')
    ends = np.flatnonzero(np.r_[scores[order][:-1] != scores[order][1:], True])
    cumulative = np.cumsum(labels[:, order], axis=1)[:, ends]
    positives = np.diff(np.c_[np.zeros(len(labels), dtype=int), cumulative], axis=1)
    return np.sum(positives * cumulative / (ends + 1), axis=1) / labels.sum(axis=1)


def score_labels(labels, scores, lengths):
    prevalence = float(np.mean(labels))
    raw = np.empty((6, 3, 2))
    exact = np.empty_like(raw)
    for c, s, v in itertools.product(range(6), range(3), range(2)):
        ap = vector_ap(labels, scores[c, s, v])[0]
        raw[c, s, v] = (ap - prevalence) / (1 - prevalence)
        exact[c, s, v] = (ap - null_mean(labels, scores[c, s, v], lengths)) / (1 - prevalence)
    return np.array([raw, exact])


def paired(values):
    means = values.mean(axis=-2)
    return means[..., 0] - means[..., 1]


def aggregate(values, datasets):
    return np.median(np.stack([np.median(values[..., datasets == name, :], axis=(-2, -1))
        for name in sorted(set(datasets))], axis=-1), axis=-1)


def endpoint_options(molecule, query):
    options = set()
    matches = molecule.GetSubstructMatches(query, uniquify=False, maxMatches=4097)
    assert len(matches) <= 4096
    for match in matches:
        atoms = frozenset(match)
        bonds = {molecule.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]).GetIdx()
            for b in query.GetBonds()}
        changed = set(range(molecule.GetNumAtoms())) - atoms
        for bond in molecule.GetBonds():
            if bond.GetIdx() not in bonds:
                changed.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        changed |= {n.GetIdx() for i in list(changed) for n in molecule.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in atoms}
        boundary = frozenset(changed)
        options.add((atoms, boundary))
    return sorted(options, key=lambda a: (tuple(sorted(a[0])), tuple(sorted(a[1]))))


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    molecules = pd.read_csv(PREP / 'model_molecules.csv').set_index('row_id')
    mols = {i: Chem.MolFromSmiles(row.canonical_smiles) for i, row in molecules.iterrows()
        if row.dataset in BASE['confirmation_targets']}
    counts = {i: mol.GetNumAtoms() for i, mol in mols.items()}
    pairs = pd.read_csv(PREP / 'sentinel_pairs.csv')
    pairs = pairs[pairs.dataset.isin(BASE['confirmation_targets']) & pairs.component_split.eq('test')].copy()
    pairs['left_id'] = pairs.dataset + ':' + pairs.left_index.astype(int).astype(str)
    pairs['right_id'] = pairs.dataset + ':' + pairs.right_index.astype(int).astype(str)
    pairs = pairs[[pair_label_arrays(p, counts, 'boundary_extended')[1] for p in pairs.itertuples()]].sort_values(['dataset', 'component_id']).reset_index(drop=True)
    assert len(pairs) == 617 and not pairs.duplicated(['dataset', 'component_id']).any()
    mapping = pd.read_csv(OLD / 'mcs_mapping_ambiguity.csv').set_index(['dataset', 'component_id'])
    assert mapping.positive_atom_set_ambiguous.sum() == 77 and mapping.enumeration_eligible.sum() == 611
    atoms = load_atom_scores(ROOT / CFG['inputs']['atom_scores'], BASE['confirmation_targets'], 'crem_mean')
    endpoints = set(pairs.left_id) | set(pairs.right_id)
    arrays = atom_array_lookup(atoms[atoms.row_id.isin(endpoints)])
    del atoms
    old_rows = pd.read_csv(OLD / 'model_null_metrics.csv')
    old_rows = old_rows[old_rows.denominator.eq('supported')].set_index(
        ['dataset', 'component_id', 'rationale', 'representation', 'predictor', 'seed', 'variant'])
    datasets = pairs.dataset.to_numpy()
    reference = np.empty((2, DRAWS, len(pairs), 6))
    observed = {r: np.full((2, len(pairs), 6), np.nan) for r in ['boundary_extended', 'strict_unmatched']}
    candidates = {r: {} for r in observed}
    candidate_rows, candidate_seed_rows, enumeration, coverage = [], [], [], []
    errors = []
    for index, pair in enumerate(pairs.itertuples()):
        lengths = [counts[pair.left_id], counts[pair.right_id]]
        scores = np.array([[[(np.r_[arrays[(pair.dataset, rep, pred, seed, variant, pair.left_id)],
            arrays[(pair.dataset, rep, pred, seed, variant, pair.right_id)]]) for variant in VARIANTS]
            for seed in SEEDS] for rep, pred in CELLS])
        labels, _ = pair_label_arrays(pair, counts, 'boundary_extended')
        values = score_labels(labels, scores, lengths)
        observed['boundary_extended'][:, index] = paired(values)
        for c, (rep, pred) in enumerate(CELLS):
            for s, seed in enumerate(SEEDS):
                for v, variant in enumerate(VARIANTS):
                    previous = old_rows.loc[(pair.dataset, pair.component_id, 'boundary_extended', rep, pred, seed, variant)]
                    errors.extend(abs(values[m, c, s, v] - previous[metric]) for m, metric in enumerate(METRICS))
                    errors.append(abs(vector_ap(labels, scores[c, s, v])[0] - grouped_average_precision(labels, scores[c, s, v])))
        strict, valid = pair_label_arrays(pair, counts, 'strict_unmatched')
        if valid:
            observed['strict_unmatched'][:, index] = paired(score_labels(strict, scores, lengths))
        rng = rng_for(SEED, f'nested-reference:{pair.dataset}:{pair.component_id}')
        random_labels = np.zeros((DRAWS, sum(lengths)), dtype=np.int8)
        offset = 0
        for n in lengths:
            k = int(labels[offset:offset+n].sum())
            if k:
                selected = np.argpartition(rng.random((DRAWS, n)), k-1, axis=1)[:, :k]
                random_labels[np.arange(DRAWS)[:, None], offset+selected] = 1
            assert np.all(random_labels[:, offset:offset+n].sum(axis=1) == k)
            offset += n
        prevalence = labels.mean()
        for c in range(6):
            raw = np.zeros(DRAWS)
            for s in range(3):
                raw += (vector_ap(random_labels, scores[c, s, 0]) - vector_ap(random_labels, scores[c, s, 1])) / (3 * (1-prevalence))
            reference[0, :, index, c] = raw
            reference[1, :, index, c] = raw + observed['boundary_extended'][1, index, c] - observed['boundary_extended'][0, index, c]
        status = mapping.loc[(pair.dataset, pair.component_id)]
        if status.positive_atom_set_ambiguous:
            query = Chem.MolFromSmarts(status.query_smarts)
            options = [endpoint_options(mols[i], query) for i in [pair.left_id, pair.right_id]]
            assert len({a for a,b in options[0]}) == status.left_unique_matched_atom_sets
            assert len({a for a,b in options[1]}) == status.right_unique_matched_atom_sets
            stored_sets = [frozenset(m[side] for m in json.loads(pair.match)) for side in [0, 1]]
            stored_found = {r:0 for r in candidates}
            seen = {r:set() for r in candidates}
            local = {r: [] for r in candidates}
            for candidate_id, (left, right) in enumerate(itertools.product(*options)):
                chosen = left[0] == stored_sets[0] and right[0] == stored_sets[1]
                enumeration.append({'dataset': pair.dataset, 'component_id': pair.component_id, 'candidate_id': candidate_id,
                    'stored_atom_sets': chosen, 'left_matched_atoms': json.dumps(sorted(left[0])), 'right_matched_atoms': json.dumps(sorted(right[0])),
                    'left_boundary_atoms': json.dumps(sorted(left[1])), 'right_boundary_atoms': json.dumps(sorted(right[1]))})
                for rationale in candidates:
                    sets = [left[1], right[1]] if rationale == 'boundary_extended' else [set(range(n))-a[0] for n, a in zip(lengths, [left, right])]
                    lab = np.r_[np.isin(np.arange(lengths[0]), list(sets[0])), np.isin(np.arange(lengths[1]), list(sets[1]))]
                    key = tuple(np.flatnonzero(lab))
                    if key in seen[rationale]:
                        continue
                    seen[rationale].add(key)
                    original, original_valid = pair_label_arrays(pair, counts, rationale)
                    chosen_labels = np.array_equal(original, lab)
                    stored_found[rationale] += chosen_labels
                    valid = 0 < lab.sum() < len(lab)
                    coverage.append({'dataset': pair.dataset, 'component_id': pair.component_id, 'candidate_id': candidate_id,
                        'rationale': rationale, 'valid': valid, 'positive_atoms': int(lab.sum()), 'atoms': len(lab)})
                    if not valid:
                        continue
                    candidate_values = score_labels(lab, scores, lengths)
                    delta = paired(candidate_values)
                    local[rationale].append((candidate_id, delta))
                    if chosen_labels:
                        assert original_valid and np.array_equal(original, lab), (pair.dataset, pair.component_id, rationale,
                            np.flatnonzero(original).tolist(), np.flatnonzero(lab).tolist())
                        errors.append(float(np.max(np.abs(delta-observed[rationale][:, index]))))
                    for c, (rep, pred) in enumerate(CELLS):
                        for m, metric in enumerate(METRICS):
                            candidate_rows.append({'dataset': pair.dataset, 'component_id': pair.component_id, 'candidate_id': candidate_id,
                                'rationale': rationale, 'stored_candidate': chosen_labels, 'representation': rep, 'predictor': pred, 'metric': metric,
                                'trained_seed_mean': candidate_values[m, c, :, 0].mean(), 'permuted_seed_mean': candidate_values[m, c, :, 1].mean(),
                                'paired_delta': delta[m, c]})
                        for s, seed in enumerate(SEEDS):
                            candidate_seed_rows.append({'dataset': pair.dataset, 'component_id': pair.component_id, 'candidate_id': candidate_id,
                                'rationale': rationale, 'representation': rep, 'predictor': pred, 'seed': seed,
                                **{metric+'_'+variant: candidate_values[m,c,s,v] for m,metric in enumerate(METRICS) for v,variant in enumerate(VARIANTS)}})
            assert all(value == 1 for value in stored_found.values()), (pair.dataset, pair.component_id, stored_found)
            for rationale, entries in local.items():
                if entries:
                    candidates[rationale][index] = entries
        if (index+1) % 50 == 0:
            print('pairs', index+1, 'elapsed_seconds', round(time.time()-started, 1), flush=True)
    assert max(errors) < 1e-12
    primary_reference = aggregate(reference, datasets)
    observed_primary = aggregate(observed['boundary_extended'], datasets)
    reference_summary = []
    median_rng = rng_for(SEED, 'reference-median-mc-error')
    median_resamples = median_rng.integers(0, DRAWS, (2000, DRAWS))
    for m, metric in enumerate(METRICS):
        values = primary_reference[m]
        median_draws = np.median(values[median_resamples], axis=1)
        lower_count = int(np.sum(values <= observed_primary[m]))
        lower = (1 + lower_count) / (DRAWS + 1)
        upper = (1 + int(np.sum(values >= observed_primary[m]))) / (DRAWS + 1)
        reference_summary.append({'metric': metric, 'observed': observed_primary[m], 'reference_mean': values.mean(),
            'reference_median': np.median(values), 'reference_q025': np.quantile(values,.025), 'reference_q975': np.quantile(values,.975),
            'mc_se_mean': values.std(ddof=1)/np.sqrt(DRAWS), 'mc_se_median': median_draws.std(ddof=1),
            'mc_median_ci_low': np.quantile(median_draws,.025), 'mc_median_ci_high': np.quantile(median_draws,.975),
            'lower_tail_fraction': lower, 'upper_tail_fraction': upper, 'two_sided_tail_fraction': min(1, 2*min(lower,upper)),
            'lower_tail_count':lower_count,
            'lower_tail_binomial_ci_low':float(beta.ppf(.025,lower_count,DRAWS-lower_count+1)) if lower_count else 0.,
            'lower_tail_binomial_ci_high':float(beta.ppf(.975,lower_count+1,DRAWS-lower_count)) if lower_count<DRAWS else 1.,
            'tail_mc_se': np.sqrt(min(lower,upper)*(1-min(lower,upper))/DRAWS),
            'first_half_median': np.median(values[:DRAWS//2]), 'second_half_median': np.median(values[DRAWS//2:]),
            'draws': DRAWS, 'datasets': len(set(datasets)), 'pairs': len(pairs)})
    save('nested_reference_summary.csv', reference_summary)
    save('table_S43.csv', pd.DataFrame(reference_summary)[['metric','observed','reference_median','reference_q025','reference_q975',
        'mc_se_median','lower_tail_count','draws','lower_tail_fraction','lower_tail_binomial_ci_low','lower_tail_binomial_ci_high']])
    save('nested_reference_draws.csv', {'draw': np.arange(DRAWS), **{metric:primary_reference[m] for m,metric in enumerate(METRICS)}})
    save('mapping_candidates.csv', enumeration)
    save('mapping_candidate_coverage.csv', coverage)
    save('mapping_candidate_seed_metrics.csv', candidate_seed_rows)
    save('mapping_candidate_paired_metrics.csv', candidate_rows)
    mapping_summary, sensitivity_draws, sensitivity_assignments, ranges = [], [], [], []
    for rationale, stored in observed.items():
        valid = np.isfinite(stored).all(axis=(0,2))
        pair_indices = np.flatnonzero(valid)
        positions = {original: position for position, original in enumerate(pair_indices)}
        dsets = datasets[valid]
        random_deltas = np.broadcast_to(stored[:, None, valid], (2,DRAWS,int(valid.sum()),6)).copy()
        choices = {name:stored[:, valid].copy() for name in ['lexicographic_first','pair_median_minimum','pair_median_maximum','conservative_lower_envelope','conservative_upper_envelope']}
        for original, entries in candidates[rationale].items():
            assert original in positions
            position = positions[original]
            ids = np.array([c for c,_ in entries])
            values = np.array([v for _,v in entries])
            rng = rng_for(SEED, f'mapping-assignment:{pairs.iloc[original].dataset}:{pairs.iloc[original].component_id}:{rationale}')
            selected = rng.integers(0,len(entries),DRAWS)
            random_deltas[:,:,position] = values[selected].transpose(1,0,2)
            for m,metric in enumerate(METRICS):
                summary = np.median(values[:,m],axis=1)
                selections = {'lexicographic_first':0, 'pair_median_minimum':int(np.argmin(summary)), 'pair_median_maximum':int(np.argmax(summary))}
                for name, selection in selections.items():
                    choices[name][m,position] = values[selection,m]
                    sensitivity_assignments.append({'dataset':pairs.iloc[original].dataset,'component_id':pairs.iloc[original].component_id,
                        'rationale':rationale,'metric':metric,'scenario':name,'candidate_id':int(ids[selection])})
                choices['conservative_lower_envelope'][m,position] = values[:,m].min(axis=0)
                choices['conservative_upper_envelope'][m,position] = values[:,m].max(axis=0)
                for c,(rep,pred) in enumerate(CELLS):
                    ranges.append({'dataset':pairs.iloc[original].dataset,'component_id':pairs.iloc[original].component_id,'rationale':rationale,
                        'metric':metric,'representation':rep,'predictor':pred,'candidates':len(entries),'paired_delta_min':values[:,m,c].min(),
                        'paired_delta_max':values[:,m,c].max(),'paired_delta_range':np.ptp(values[:,m,c])})
        random_primary = aggregate(random_deltas,dsets)
        for m,metric in enumerate(METRICS):
            row = {'rationale':rationale,'metric':metric,'stored_mapping_estimate':float(aggregate(stored[:,valid],dsets)[m]),
                'pairs':int(valid.sum()),'ambiguous_pairs':len(candidates[rationale]),'random_assignment_draws':DRAWS,
                'pairs_with_multiple_distinct_label_vectors':sum(len(entries)>1 for entries in candidates[rationale].values()),
                'distinct_candidate_label_vectors':sum(len(entries) for entries in candidates[rationale].values()),
                'random_assignment_median':float(np.median(random_primary[m])), 'random_assignment_q025':float(np.quantile(random_primary[m],.025)),
                'random_assignment_q975':float(np.quantile(random_primary[m],.975)), 'random_assignment_min':float(random_primary[m].min()),
                'random_assignment_max':float(random_primary[m].max()), 'random_assignment_positive_fraction':float(np.mean(random_primary[m]>0))}
            row.update({name:float(aggregate(values,dsets)[m]) for name,values in choices.items()})
            mapping_summary.append(row)
            sensitivity_draws.extend({'rationale':rationale,'metric':metric,'draw':draw,'estimate':value} for draw,value in enumerate(random_primary[m]))
    save('mapping_primary_sensitivity.csv', mapping_summary)
    save('table_S44.csv', pd.DataFrame(mapping_summary)[['rationale','metric','stored_mapping_estimate','pairs','ambiguous_pairs',
        'random_assignment_q025','random_assignment_q975','conservative_lower_envelope','conservative_upper_envelope']])
    save('mapping_assignment_draws.csv', sensitivity_draws)
    save('mapping_scenario_assignments.csv', sensitivity_assignments)
    save('mapping_pair_config_ranges.csv', ranges)
    validation = {'status':'completed','row_metric_max_absolute_error':max(errors),'row_checks':len(errors),'boundary_pairs':617,
        'strict_pairs':int(np.isfinite(observed['strict_unmatched']).all(axis=(0,2)).sum()),'mapping_assessed_pairs':611,'mapping_ambiguous_pairs':77,
        'mapping_candidates':len(enumeration),'invalid_candidates':sum(not row['valid'] for row in coverage),
        'distinct_candidate_label_vectors':{r:sum(len(entries) for entries in candidates[r].values()) for r in candidates},
        'same_candidate_shared_across_conditions_seeds_configurations':True,'boundary_invariance_by_matched_atom_set':False,
        'mapping_enumeration':'All query embeddings in the existing 77 atom-set-ambiguous pairs; unique localization labels within each rationale.',
        'unreproduced_mappings_retained':6,'monte_carlo_draws':DRAWS,'seed':SEED,'no_refit':True,
        'scope':'Post-confirmation frozen-score conditional reference and mapping sensitivity; not population intervals or a label-exchangeability test.',
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'rdkit':rdBase.rdkitVersion,'seconds':time.time()-started}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    input_paths = [ROOT/CFG['inputs']['atom_scores'],PREP/'sentinel_pairs.csv',PREP/'model_molecules.csv',
        OLD/'mcs_mapping_ambiguity.csv',OLD/'model_null_metrics.csv',Path(__file__)]
    (OUT/'input_provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
        if p.is_file() and p.name!='manifest.json'},indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)
    print(pd.DataFrame(reference_summary).to_string(index=False),flush=True)
    print(pd.DataFrame(mapping_summary).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
