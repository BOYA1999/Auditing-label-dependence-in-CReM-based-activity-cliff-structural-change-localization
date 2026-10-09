import json
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem

from score_factorial_explanations import (BASE, COUNTERFACTUALS, PREPARED,
    REPRESENTATIONS, MUTANT_REPRESENTATIONS, XGB_HEADS, MLP_HEADS,
    control_labels, mean_replacement_arrays, add_control_rows, load_npz,
    load_predictor, predict)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--oracle-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    molecules=pd.read_csv(PREPARED/'molecules.csv')
    scored=pd.read_csv(COUNTERFACTUALS/'scored_mutants.csv')
    unique=pd.read_csv(COUNTERFACTUALS/'unique_scored_mutants.csv')
    pairs=pd.read_csv(PREPARED/'sentinel_pairs.csv')
    selected=set(scored.row_id)|{f'{p.dataset}:{int(i)}' for p in pairs.itertuples() for i in [p.left_index,p.right_index]}
    evaluation=molecules[molecules.row_id.isin(selected)].copy()
    positions={s:i for i,s in enumerate(unique.canonical_smiles)}
    scored['mutant_position']=scored.mutant_smiles.map(positions)
    assert scored.mutant_position.notna().all()
    parent_labels={r.row_id:control_labels(Chem.MolFromSmiles(r.canonical_smiles)) for r in evaluation.itertuples()}
    mutant_labels=[control_labels(Chem.MolFromSmiles(s)) for s in unique.canonical_smiles]
    concepts=['halogen_count','aromatic_atom_count','heteroatom_count']
    oracle_rows=[]
    for concept in concepts:
        parents={i:float(v[concept].sum()) for i,v in parent_labels.items()}
        mutants=np.array([v[concept].sum() for v in mutant_labels],float)
        arrays,support=mean_replacement_arrays(evaluation,scored,mutants,parents,0)
        add_control_rows(oracle_rows,evaluation,arrays,support,
            {'control':concept,'representation':'exact_count','predictor':'oracle','explainer':'crem_mean','seed':0},1.0)
    oracle=pd.DataFrame(oracle_rows)
    oracle.to_csv(OUT/'count_oracle_molecules.csv',index=False)
    oracle.groupby('control').agg(molecules=('row_id','size'),median_oracle_nap=('normalized_ap_lift','median'),
        median_rationale_support=('rationale_supported_fraction','median')).reset_index().to_csv(OUT/'count_oracle_summary.csv',index=False)
    calibrated=pd.read_csv(BASE/'full/calibrated_probe_controls/calibrated_positive_controls.csv')
    calibrated=calibrated[calibrated.explainer.eq('crem_mean')]
    joint=calibrated.merge(oracle[['row_id','control','normalized_ap_lift']],on=['row_id','control'],suffixes=('','_oracle'),validate='many_to_one')
    joint['oracle_minus_randomized_nap']=joint.normalized_ap_lift_oracle-joint.mean_randomized_nap
    summaries=[]
    for (rep,pred,concept),part in joint.groupby(['representation','predictor','control']):
        summaries.append({'representation':rep,'predictor':pred,'control':concept,'molecules':len(part),
            'trained_nap':part.normalized_ap_lift.median(),'randomized_nap':part.mean_randomized_nap.median(),
            'trained_minus_randomized_nap':part.randomization_calibrated_nap.median(),
            'oracle_nap':part.normalized_ap_lift_oracle.median(),
            'oracle_minus_randomized_nap':part.oracle_minus_randomized_nap.median()})
    summary=pd.DataFrame(summaries)
    if args.oracle_only:
        summary.to_csv(OUT/'count_oracle_matched_summary.csv',index=False)
        print('Exact oracle replay completed; learned-model mutant errors not rerun.',flush=True)
        return
    device=torch.device('cpu'); errors=[]
    original_ids=molecules.row_id.astype(str).to_numpy()
    original_positions={i:p for p,i in enumerate(original_ids)}
    test_ids=set(evaluation.loc[evaluation.component_split.eq('test'),'row_id'])
    test_scored=scored[scored.row_id.isin(test_ids)].copy()
    for rep,tasks in summary.groupby('representation'):
        x=load_npz(REPRESENTATIONS/f'{rep}.npz',original_ids)
        z=load_npz(MUTANT_REPRESENTATIONS/f'{rep}.npz',unique.row_id.astype(str).to_numpy())
        for task in tasks.itertuples():
            head=XGB_HEADS if task.predictor=='xgboost' else MLP_HEADS
            manifest=pd.read_csv(head/'model_manifest.csv')
            entry=manifest[manifest.kind.eq('probe')&manifest.representation.eq(rep)&manifest.control.eq(task.control)].iloc[0]
            model=load_predictor(task.predictor,entry,device)
            mutant_prediction=predict(task.predictor,model,z,device)
            parent_prediction=predict(task.predictor,model,x,device)
            mutant_truth=np.array([v[task.control].sum() for v in mutant_labels],float)
            ix=test_scored.mutant_position.to_numpy(int)
            px=np.array([original_positions[i] for i in test_scored.row_id])
            truth_parent=np.array([parent_labels[i][task.control].sum() for i in test_scored.row_id])
            true_effect=np.abs(mutant_truth[ix]-truth_parent)
            predicted_effect=np.abs(mutant_prediction[ix]-parent_prediction[px])
            unique_ix=np.unique(ix)
            errors.append({'representation':rep,'predictor':task.predictor,'control':task.control,
                'test_parent_mutant_rows':len(ix),'unique_test_mutants':len(unique_ix),
                'mutant_count_mae':float(np.mean(np.abs(mutant_prediction[unique_ix]-mutant_truth[unique_ix]))),
                'absolute_effect_mae':float(np.mean(np.abs(predicted_effect-true_effect)))})
            print('count diagnostic',rep,task.predictor,task.control,flush=True)
    summary.merge(pd.DataFrame(errors),on=['representation','predictor','control'],validate='one_to_one').to_csv(OUT/'count_oracle_task_diagnostics.csv',index=False)
    (OUT/'count_oracle_validation.json').write_text(json.dumps({'status':'completed','model_refit':False,
        'known_function':'Exact atom count queried on the original retained mutants; original mask-to-atom mean and support rules.',
        'heteroatom_atomic_numbers':[7,8,15,16],'halogen_atomic_numbers':[9,17,35,53],
        'qualified_tasks':len(summary),'scope':'Development-target positive controls; not biological validation.'},indent=2)+'\n')

if __name__=='__main__':main()
