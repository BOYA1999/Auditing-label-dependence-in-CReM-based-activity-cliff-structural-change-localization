import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from score_factorial_explanations import load_npz, load_predictor, predict, mean_replacement_arrays, normalized_ap

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research_v2/artifacts/experiment/crossed_20261019'
FULL = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/full'
PREP = FULL.parent / 'prepared'
SEEDS = [11, 23, 42]


def main():
    start = time.time()
    source_dir = RUN / 'atom_scores_source'
    if not source_dir.exists():
        assert json.loads((RUN / 'atom_export_manifest.json').read_text())['status'] == 'pass'
        shutil.copytree(RUN / 'atom_scores', source_dir)
        for name in ['atom_export_manifest.json', 'atom_export_validation.csv']:
            shutil.copy2(RUN / name, RUN / name.replace('.', '_source.', 1))
    (RUN / 'atom_export_manifest.json').write_text(json.dumps({'status':'processing_uniform_backend','files':0,'permuted_models':2430}))
    randomized = pd.read_csv(RUN / 'consolidated_model_manifest.csv')
    trained = pd.read_csv(FULL / 'sanity_heads/model_manifest.csv')
    trained = trained[trained.variant.eq('trained') & trained.dataset.isin(randomized.dataset)]
    molecules = pd.read_csv(PREP / 'model_molecules.csv')
    evaluation = pd.read_csv(PREP / 'evaluation_molecules.csv')
    unique = pd.read_csv(FULL / 'counterfactuals/unique_scored_mutants_model.csv')
    scored = pd.read_csv(FULL / 'counterfactuals/scored_mutants_model.csv')
    scored['mutant_position'] = scored.mutant_smiles.map(dict(zip(unique.canonical_smiles, range(len(unique)))))
    original_positions = dict(zip(molecules.row_id, range(len(molecules))))
    pairs = pd.read_csv(PREP / 'sentinel_pairs.csv')
    expected = pd.read_csv(RUN / 'consolidated_pair_explanations.csv')
    keys = expected[expected.component_split.eq('test')][['dataset', 'component_id']].drop_duplicates()
    pairs = pairs.merge(keys, on=['dataset', 'component_id'], validate='one_to_one')
    validation, diagnostics = [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for predictor in ['mlp', 'xgboost']:
        for representation in ['morgan', 'molformer', 'unimol']:
            values = load_npz(FULL / 'representations' / f'{representation}.npz', molecules.row_id.astype(str).to_numpy())
            mutants = load_npz(FULL / 'counterfactual_representations' / f'{representation}.npz', unique.row_id.astype(str).to_numpy())
            for dataset in sorted(randomized.dataset.unique()):
                path = RUN / 'atom_scores' / f'{dataset}--{representation}--{predictor}.npz'
                source = dict(np.load(source_dir / path.name))
                ev = evaluation[evaluation.dataset.eq(dataset)]
                geom = scored[scored.row_id.isin(ev.row_id)]
                mutant_indices = np.sort(geom.mutant_position.unique()).astype(int)
                parent_indices = [original_positions[i] for i in ev.row_id]
                def score(row):
                    loaded = load_predictor(predictor, row, device)
                    if predictor == 'xgboost':
                        loaded.set_params(device='cpu')
                    pred = np.full(len(unique), np.nan)
                    pred[mutant_indices] = predict(predictor, loaded, mutants[mutant_indices], device)
                    parents = dict(zip(ev.row_id, predict(predictor, loaded, values[parent_indices], device)))
                    arrays, support = mean_replacement_arrays(ev, geom, pred, parents, 0.)
                    assert np.array_equal(np.concatenate([support[i] for i in source['row_ids']]), source['supported_flat'])
                    return np.concatenate([arrays[i] for i in source['row_ids']])
                t = trained[trained.dataset.eq(dataset) & trained.representation.eq(representation) & trained.predictor.eq(predictor)].set_index('seed')
                observed = np.stack([score(next(t.loc[[s]].itertuples())) for s in SEEDS])
                uniform = source['scores'].copy()
                if predictor == 'xgboost':
                    r = randomized[randomized.dataset.eq(dataset) & randomized.representation.eq(representation) & randomized.predictor.eq(predictor)]
                    for si, seed in enumerate(SEEDS):
                        for pi, perm in enumerate(range(42, 47)):
                            uniform[si, pi] = score(next(r[r.model_initialization_seed.eq(seed) & r.permutation_id.eq(perm)].itertuples()))
                assert np.isfinite(uniform).all() and np.isfinite(observed).all()
                observed_max = float(np.max(np.abs(observed-source['observed_scores'])))
                assert observed_max < 1e-7, (path.name, observed_max)
                for p in pairs[pairs.dataset.eq(dataset)].itertuples():
                    indices, labels = [], []
                    for i, rationale in [(p.left_index,p.rationale_left),(p.right_index,p.rationale_right)]:
                        pos = list(source['row_ids']).index(f'{dataset}:{int(i)}')
                        a,b = source['offsets'][pos:pos+2]
                        indices.extend(range(a,b)); y=np.zeros(b-a,int);y[json.loads(rationale)]=1;labels.extend(y)
                    for variant, old, new in [('observed',source['observed_scores'][:,None,:],observed[:,None,:]),('permuted',source['scores'],uniform)]:
                        for si, seed in enumerate(SEEDS):
                            for pi in range(new.shape[1]):
                                x,y=old[si,pi,indices],new[si,pi,indices]
                                ap0,_,n0=normalized_ap(labels,x);ap1,_,n1=normalized_ap(labels,y)
                                diagnostics.append({'dataset':dataset,'component_id':int(p.component_id),'representation':representation,'predictor':predictor,'variant':variant,'initialization':seed,'permutation':pi+42 if variant=='permuted' else -1,'source_nap':n0,'uniform_nap':n1,'source_ap':ap0,'uniform_ap':ap1,'rank_changed':not np.array_equal(rankdata(x),rankdata(y)),'ap_changed':ap0!=ap1,'maximum_atom_difference':float(np.max(np.abs(x-y)))})
                np.savez_compressed(path, row_ids=source['row_ids'],offsets=source['offsets'],supported_flat=source['supported_flat'],initializations=source['initializations'],permutations=source['permutations'],scores=uniform,observed_scores=observed,scores_source=source['scores'],observed_scores_source=source['observed_scores'])
                validation.append({'dataset':dataset,'representation':representation,'predictor':predictor,'observed_max_atom_difference':observed_max,'permuted_max_atom_difference':float(np.max(np.abs(uniform-source['scores']))),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                print(len(validation),path.name,flush=True)
        if predictor == 'mlp':
            print('GPU_RELEASED: all MLP observed reconstruction complete; remaining XGBoost runs on CPU.',flush=True)
            (RUN / 'GPU_RELEASED.txt').write_text('MLP scoring complete; remaining XGBoost inference uses CPU.\n')
    details=pd.DataFrame(diagnostics)
    details.to_csv(RUN/'backend_pair_diagnostics.csv.gz',index=False)
    summary=details.groupby(['predictor','variant']).agg(pair_realizations=('rank_changed','size'),rank_changed=('rank_changed','sum'),ap_changed=('ap_changed','sum'),maximum_atom_difference=('maximum_atom_difference','max'))
    summary.to_csv(RUN/'backend_difference_summary.csv')
    pd.DataFrame(validation).to_csv(RUN/'atom_export_validation.csv',index=False)
    estimates=[]
    for kind in ['source','uniform']:
        paired=details.groupby(['dataset','component_id','representation','predictor','variant'])[kind+'_nap'].mean().unstack('variant').reset_index()
        paired['delta']=paired.observed-paired.permuted
        estimates.append({'backend':kind,'raw_nap_contrast':float(paired.groupby('dataset').delta.median().median())})
    manifest={'status':'pass','files':len(validation),'permuted_models':2430,'observed_models':486,'elapsed_seconds':time.time()-start,'backend_primary':'XGBoost CPU for all observed and permuted models; MLP original GPU backend for all models','source_compatible_sensitive_fields':['scores_source','observed_scores_source'],'source_validation':'atom_export_validation_source.csv; historical source-compatible per-pair nAP checks preserved','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'raw_aggregate_comparison':estimates,'retraining':False}
    (RUN/'atom_export_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
