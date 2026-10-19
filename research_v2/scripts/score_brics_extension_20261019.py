import hashlib
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem

from analyze_review_revision import metrics
from analyze_round2_revision import two_stage_bootstrap
from score_factorial_explanations import load_predictor, mean_replacement_arrays, predict

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1'
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/brics'


def main():
    started = time.time()
    cfg = json.loads((ROOT / 'research_v2/configs/sanity_full_v1.json').read_text())
    population = pd.read_csv(OUT / 'frozen_pair_population.csv')
    pairs = pd.read_csv(BASE / 'prepared/sentinel_pairs.csv').merge(population, on=['dataset', 'component_id', 'left_index', 'right_index'], validate='one_to_one')
    molecules = pd.read_csv(BASE / 'prepared/model_molecules.csv')
    ids = {f'{r.dataset}:{int(i)}' for r in pairs.itertuples() for i in [r.left_index, r.right_index]}
    evaluation = molecules[molecules.row_id.isin(ids)].sort_values('row_id')
    sizes = {r.row_id: Chem.MolFromSmiles(r.canonical_smiles).GetNumAtoms() for r in evaluation.itertuples()}
    selected = pd.read_csv(OUT / 'matched_scored_products.csv')
    old_unique = pd.read_csv(BASE / 'full/counterfactuals/unique_scored_mutants_model.csv')
    new_unique = pd.read_csv(OUT / 'unique_brics_mutants.csv')
    models = pd.read_csv(BASE / 'full/sanity_heads/model_manifest.csv')
    device = torch.device('cuda')
    atomdir = OUT / 'atom_scores'
    atomdir.mkdir(exist_ok=True)
    rows, manifests = [], []
    for representation in cfg['representations']:
        original = np.load(BASE / f'full/representations/{representation}.npz', allow_pickle=False)
        assert np.array_equal(original['row_id'].astype(str), molecules.row_id.to_numpy(str))
        values = original['x']
        product_values = {}
        for generator, unique, path in [('crem', old_unique, BASE / f'full/counterfactual_representations/{representation}.npz'), ('brics', new_unique, OUT / f'representations/{representation}.npz')]:
            data = np.load(path, allow_pickle=False)
            assert np.array_equal(data['row_id'].astype(str), unique.row_id.to_numpy(str))
            positions = dict(zip(unique.canonical_smiles, range(len(unique))))
            group = selected[selected.generator.eq(generator)].copy()
            used = sorted(group.mutant_smiles.unique())
            product_values[generator] = data['x'][[positions[s] for s in used]]
            index = {s:i for i,s in enumerate(used)}
            selected.loc[group.index, 'mutant_position'] = group.mutant_smiles.map(index)
            del data
        for target in cfg['confirmation_targets']:
            target_eval = evaluation[evaluation.dataset.eq(target)]
            parent_positions = molecules.index[molecules.row_id.isin(target_eval.row_id)].to_numpy()
            parent_ids = molecules.iloc[parent_positions].row_id.tolist()
            target_pairs = pairs[pairs.dataset.eq(target)]
            offsets = np.cumsum([0] + [sizes[i] for i in target_eval.row_id])
            groups = {g:selected[selected.generator.eq(g) & selected.dataset.eq(target)].copy() for g in ['crem','brics']}
            for group in groups.values():
                group['mutant_position'] = group.mutant_position.astype(int)
            for predictor in ['xgboost', 'mlp']:
                arrays = np.zeros((2, 2, 3, int(offsets[-1])))
                for vi, variant in enumerate(['trained', 'label_permuted']):
                    for si, seed in enumerate([11,23,42]):
                        entry = models[models.dataset.eq(target) & models.representation.eq(representation) & models.predictor.eq(predictor) & models.variant.eq(variant) & models.seed.eq(seed)]
                        assert len(entry) == 1
                        entry = next(entry.itertuples(index=False))
                        loaded = load_predictor(predictor, entry, device)
                        if predictor == 'xgboost':
                            loaded.set_params(device='cpu')
                        parent_pred = dict(zip(parent_ids, predict(predictor, loaded, values[parent_positions], device)))
                        for gi, generator in enumerate(['crem','brics']):
                            group = groups[generator]
                            use = np.sort(group.mutant_position.unique())
                            pred = np.full(len(product_values[generator]), np.nan)
                            if len(use):
                                pred[use] = predict(predictor, loaded, product_values[generator][use], device)
                            atom, support = mean_replacement_arrays(target_eval, group, pred, parent_pred, 0)
                            arrays[gi,vi,si] = np.concatenate([atom[i] for i in target_eval.row_id])
                            for pair in target_pairs.itertuples():
                                left, right = f'{target}:{pair.left_index}', f'{target}:{pair.right_index}'
                                labels = np.r_[np.isin(np.arange(sizes[left]), json.loads(pair.rationale_left)), np.isin(np.arange(sizes[right]), json.loads(pair.rationale_right))]
                                score = np.r_[atom[left], atom[right]]
                                result = metrics(labels, score, [sizes[left],sizes[right]])
                                rows.append({'dataset':target,'component_id':int(pair.component_id),'representation':representation,'predictor':predictor,'generator':generator,'variant':variant,'seed':seed,'common_both_endpoints':bool(pair.common_both_endpoints),**result})
                        manifests.append({'dataset':target,'representation':representation,'predictor':predictor,'variant':variant,'seed':seed,'path':entry.path,'sha256':entry.sha256})
                np.savez_compressed(atomdir / f'{target}--{representation}--{predictor}.npz', row_ids=target_eval.row_id.to_numpy(str), offsets=offsets, scores=arrays, generators=np.array(['crem','brics']), variants=np.array(['trained','label_permuted']), seeds=np.array([11,23,42]))
            print(f'scored {representation} {target} rows={len(rows)} elapsed={time.time()-started:.1f}s',flush=True)
        del values, original, product_values
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'pair_metrics.csv', index=False)
    pd.DataFrame(manifests).to_csv(OUT / 'scoring_models.csv', index=False)
    keys = ['dataset','component_id','representation','predictor','generator','common_both_endpoints']
    mean = frame.groupby(keys+['variant'])[['raw_nap','endpoint_exact_nap']].mean().unstack('variant')
    effects = pd.DataFrame({v:mean[(v,'trained')]-mean[(v,'label_permuted')] for v in ['raw_nap','endpoint_exact_nap']}).reset_index()
    effects.to_csv(OUT / 'paired_label_effects.csv',index=False)
    summaries, draws = [], []
    for metric in ['raw_nap','endpoint_exact_nap']:
        p = effects.pivot(index=keys[:4]+['common_both_endpoints'], columns='generator', values=metric).reset_index()
        p['brics_minus_crem'] = p.brics-p.crem
        for scope, sub in [('full617',p),('common272',p[p.common_both_endpoints])]:
            for effect in ['crem','brics','brics_minus_crem']:
                dist = two_stage_bootstrap(sub,effect,5000,20261019,f'brics-{scope}')
                summaries.append({'population':scope,'metric':metric,'effect':effect,'point':sub.groupby('dataset')[effect].median().median(),'ci_low':np.quantile(dist,.025),'ci_high':np.quantile(dist,.975),'datasets':sub.dataset.nunique(),'pairs':len(sub[['dataset','component_id']].drop_duplicates())})
                draws.extend({'population':scope,'metric':metric,'effect':effect,'draw':i,'value':float(x)} for i,x in enumerate(dist))
    pd.DataFrame(summaries).to_csv(OUT / 'comparison_summary.csv',index=False)
    pd.DataFrame(draws).to_csv(OUT / 'paired_bootstrap_draws.csv.gz',index=False)
    summary = {'status':'PASS_COMPUTATION','scored_model_heads':len(manifests),'pair_realizations':len(frame),'elapsed_seconds':time.time()-started,'backend':{'xgboost':'cpu','mlp':'cuda'},'model_fitting':'none; frozen original seeds11,23,42; their label permutation IDs remain matched to seed','claim_boundary':'Matched terminal single-bond regions and product budgets; post-confirmation generator sensitivity, not independent bioactivity validation. Original primary617 result unchanged.','output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}}
    (OUT / 'comparison_manifest.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
