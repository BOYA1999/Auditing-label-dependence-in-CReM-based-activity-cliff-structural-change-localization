import hashlib
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from analyze_round2_revision import load_atom_scores, atom_array_lookup
from score_factorial_explanations import load_npz, load_predictor, predict, mean_replacement_arrays, normalized_ap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/experiment/crossed_20261019'
FULL = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/full'
PREP = FULL.parent / 'prepared'
SEEDS, PERMS = [11, 23, 42], [42, 43, 44, 45, 46]


def main():
    start = time.time()
    output = OUT / 'atom_scores'
    output.mkdir(exist_ok=True)
    models = pd.read_csv(OUT / 'consolidated_model_manifest.csv')
    expected = pd.read_csv(OUT / 'consolidated_pair_explanations.csv')
    expected = expected[expected.component_split.eq('test')]
    pairs = pd.read_csv(PREP / 'sentinel_pairs.csv')
    keys = expected[['dataset', 'component_id']].drop_duplicates()
    pairs = pairs.merge(keys, on=['dataset', 'component_id'], validate='one_to_one')
    molecules = pd.read_csv(PREP / 'model_molecules.csv')
    evaluation = pd.read_csv(PREP / 'evaluation_molecules.csv')
    wanted = {f'{p.dataset}:{int(i)}' for p in pairs.itertuples() for i in [p.left_index, p.right_index]}
    unique = pd.read_csv(FULL / 'counterfactuals/unique_scored_mutants_model.csv')
    scored = pd.read_csv(FULL / 'counterfactuals/scored_mutants_model.csv')
    scored['mutant_position'] = scored.mutant_smiles.map(dict(zip(unique.canonical_smiles, range(len(unique)))))
    original_positions = dict(zip(molecules.row_id, range(len(molecules))))
    original = load_atom_scores(FULL / 'sanity_explanations/atom_scores.csv.gz', models.dataset.unique(), 'crem_mean')
    original = original[original.row_id.isin(wanted)]
    lookup = atom_array_lookup(original)
    support_lookup = {key: part.sort_values('atom_index').supported.to_numpy(bool) for key, part in original[original.seed.eq(42) & original.variant.eq('trained')].groupby(['dataset', 'representation', 'predictor', 'row_id'])}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validation = []
    for representation in sorted(models.representation.unique()):
        values = load_npz(FULL / 'representations' / f'{representation}.npz', molecules.row_id.astype(str).to_numpy())
        mutants = load_npz(FULL / 'counterfactual_representations' / f'{representation}.npz', unique.row_id.astype(str).to_numpy())
        for (dataset, predictor), group in models[models.representation.eq(representation)].groupby(['dataset', 'predictor']):
            ev = evaluation[evaluation.dataset.eq(dataset)]
            ev_export = ev[ev.row_id.isin(wanted)].sort_values('row_id')
            geom = scored[scored.row_id.isin(ev.row_id)]
            mutant_indices = np.sort(geom.mutant_position.unique()).astype(int)
            parent_indices = [original_positions[i] for i in ev.row_id]
            all_scores, common_support, max_raw_error, max_original_error = [], None, 0., 0.
            discrepancies = []
            for init in SEEDS:
                seed_scores = []
                for perm in PERMS:
                    row = next(group[group.model_initialization_seed.eq(init) & group.permutation_id.eq(perm)].itertuples())
                    loaded = load_predictor(predictor, row, device)
                    if predictor == 'xgboost' and row.source == 'new_fit':
                        loaded.set_params(device=str(device))
                    pred_mutant = np.full(len(unique), np.nan)
                    pred_mutant[mutant_indices] = predict(predictor, loaded, mutants[mutant_indices], device)
                    parent_pred = dict(zip(ev.row_id, predict(predictor, loaded, values[parent_indices], device)))
                    arrays, supports = mean_replacement_arrays(ev, geom, pred_mutant, parent_pred, 0.)
                    support = np.concatenate([supports[i] for i in ev_export.row_id])
                    assert common_support is None or np.array_equal(common_support, support)
                    common_support = support
                    seed_scores.append(np.concatenate([arrays[i] for i in ev_export.row_id]))
                    ref = expected[expected.dataset.eq(dataset) & expected.representation.eq(representation) & expected.predictor.eq(predictor) & expected.model_initialization_seed.eq(init) & expected.permutation_id.eq(perm)].set_index('component_id')
                    for pair in pairs[pairs.dataset.eq(dataset)].itertuples():
                        left, right = f'{dataset}:{int(pair.left_index)}', f'{dataset}:{int(pair.right_index)}'
                        labels = []
                        for ident, label in [(left, pair.rationale_left), (right, pair.rationale_right)]:
                            y = np.zeros(len(arrays[ident]), int)
                            y[json.loads(label)] = 1
                            labels.extend(y)
                        _, _, nap = normalized_ap(np.asarray(labels), np.concatenate([arrays[left], arrays[right]]))
                        error = abs(nap - ref.loc[pair.component_id, 'normalized_ap_lift'])
                        max_raw_error = max(max_raw_error, error)
                        if error > 1e-10:
                            detail = {'initialization':init,'permutation':perm,'component_id':int(pair.component_id),'recomputed_nap':nap,'saved_nap':float(ref.loc[pair.component_id,'normalized_ap_lift']),'labels':labels,'scores':np.concatenate([arrays[left],arrays[right]]).tolist()}
                            if init == 42 and perm == 42:
                                source = np.concatenate([lookup[(dataset,representation,predictor,42,'label_permuted',i)] for i in [left,right]])
                                detail['source_atom_scores'] = source.tolist()
                                detail['source_atom_nap'] = normalized_ap(np.asarray(labels),source)[2]
                            discrepancies.append(detail)
                    if init == 42 and perm == 42:
                        for ident in ev_export.row_id:
                            max_original_error = max(max_original_error, float(np.max(np.abs(arrays[ident] - lookup[(dataset, representation, predictor, 42, 'label_permuted', ident)]))))
                all_scores.append(seed_scores)
            observed = np.stack([np.concatenate([lookup[(dataset, representation, predictor, s, 'trained', ident)] for ident in ev_export.row_id]) for s in SEEDS])
            offsets = np.cumsum([0] + [len(lookup[(dataset, representation, predictor, 42, 'trained', ident)]) for ident in ev_export.row_id])
            if discrepancies:
                (OUT / f'atom_discrepancy_{dataset}--{representation}--{predictor}.json').write_text(json.dumps(discrepancies,indent=2,default=int))
            assert max_raw_error < 1e-10 and max_original_error < 1e-7, (dataset, representation, predictor, max_raw_error, max_original_error)
            assert np.isfinite(observed).all() and np.isfinite(all_scores).all()
            assert np.array_equal(common_support, np.concatenate([support_lookup[(dataset, representation, predictor, ident)] for ident in ev_export.row_id]))
            path = output / f'{dataset}--{representation}--{predictor}.npz'
            np.savez_compressed(path, row_ids=ev_export.row_id.astype(str).to_numpy(dtype=str), offsets=offsets, supported_flat=common_support, scores=np.asarray(all_scores), observed_scores=observed, initializations=SEEDS, permutations=PERMS)
            validation.append({'dataset': dataset, 'representation': representation, 'predictor': predictor, 'molecules': len(ev_export), 'atoms': int(offsets[-1]), 'max_pair_nap_error': max_raw_error, 'max_original_atom_score_error': max_original_error, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f'{len(validation)}/162 {path.name} raw_error={max_raw_error}', flush=True)
    pd.DataFrame(validation).to_csv(OUT / 'atom_export_validation.csv', index=False)
    (OUT / 'atom_export_manifest.json').write_text(json.dumps({'status': 'pass', 'files': len(validation), 'permuted_models': len(models), 'elapsed_seconds': time.time()-start, 'method': 'saved model inference; no refit', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
