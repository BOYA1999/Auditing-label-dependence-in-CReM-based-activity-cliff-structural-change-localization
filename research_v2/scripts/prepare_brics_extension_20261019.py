import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/brics'
BASE = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/full'


def main():
    coverage = pd.read_csv(OUT / 'mask_coverage.csv')
    keys = coverage[coverage.common_supported][['row_id','mask_id']]
    brics = pd.read_csv(OUT / 'products.csv')
    brics = brics[brics.scored].merge(keys,on=['row_id','mask_id'],validate='many_to_one')
    crem = pd.read_csv(BASE / 'counterfactuals/scored_mutants.csv').merge(keys,on=['row_id','mask_id'],validate='many_to_one')
    counts = pd.concat([crem.groupby(['row_id','mask_id']).size().rename('crem_n'),brics.groupby(['row_id','mask_id']).size().rename('brics_n')],axis=1)
    assert counts.notna().all().all()
    counts['matched_n'] = counts.min(axis=1).astype(int)
    selected = []
    for label, frame in [('crem',crem),('brics',brics)]:
        frame['order'] = [hashlib.sha256(f'{r.row_id}:{r.mask_id}:{r.mutant_smiles}'.encode()).hexdigest() for r in frame.itertuples()]
        frame = frame.sort_values(['row_id','mask_id','order'])
        frame['selection_rank'] = frame.groupby(['row_id','mask_id']).cumcount()
        frame = frame.merge(counts[['matched_n']].reset_index(),on=['row_id','mask_id'])
        frame = frame[frame.selection_rank < frame.matched_n].copy()
        frame['generator'] = label
        frame['mask_size'] = frame.mask_atoms.map(lambda s:len(json.loads(s)))
        selected.append(frame.drop(columns=['order']))
    allselected = pd.concat(selected,ignore_index=True)
    allselected.to_csv(OUT / 'matched_scored_products.csv',index=False)
    counts.to_csv(OUT / 'matched_budgets.csv')
    unique = sorted(allselected.loc[allselected.generator.eq('brics'),'mutant_smiles'].unique())
    pd.DataFrame({'row_id':[f'brics:{i}' for i in range(len(unique))],'canonical_smiles':unique}).to_csv(OUT / 'unique_brics_mutants.csv',index=False)
    summary = {'common_masks':len(counts),'budget_per_generator':len(allselected)//2,'unique_brics_mutants':len(unique),'count_distribution':counts.matched_n.value_counts().sort_index().to_dict(),'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'matched_scored_products.csv',OUT/'frozen_pair_population.csv',OUT/'matched_budgets.csv',OUT/'unique_brics_mutants.csv']}}
    (OUT / 'frozen_comparison_inputs.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
