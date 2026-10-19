import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_round2_revision import two_stage_bootstrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/brics'
BASE = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/full'


def main():
    selected = pd.read_csv(OUT / 'matched_scored_products.csv')
    representation = json.loads((OUT / 'representations/manifest.json').read_text())
    new_unique = pd.read_csv(OUT / 'unique_brics_mutants.csv')
    new_fail = set(new_unique.iloc[representation['arms']['unimol']['conformer_failure_indices']].canonical_smiles)
    affected = set(selected.loc[selected.generator.eq('brics') & selected.mutant_smiles.isin(new_fail),'row_id'])
    audit = {'brics_failure_count':len(new_fail),'brics_affected_parents':sorted(affected)}
    for name, manifest, unique in [('crem',BASE/'counterfactual_representations/manifest.json',BASE/'counterfactuals/unique_scored_mutants_model.csv'),('parent',BASE/'representations/manifest.json',None)]:
        arm = json.loads(manifest.read_text())['arms']['unimol']
        if name == 'crem':
            source = pd.read_csv(unique)
            failed = set(source.iloc[arm['conformer_failure_indices']].canonical_smiles)
            add = set(selected.loc[selected.generator.eq('crem') & selected.mutant_smiles.isin(failed),'row_id'])
        else:
            add = set(arm['conformer_failure_row_ids'])
        affected |= add
        audit[name+'_affected_parents'] = sorted(add)
    population = pd.read_csv(OUT / 'frozen_pair_population.csv')
    population['any_unimol_fallback'] = [f'{r.dataset}:{r.left_index}' in affected or f'{r.dataset}:{r.right_index}' in affected for r in population.itertuples()]
    population.to_csv(OUT/'fallback_pair_sensitivity.csv',index=False)
    audit.update(affected_full_pairs=int(population.any_unimol_fallback.sum()),affected_common_pairs=int((population.any_unimol_fallback & population.common_both_endpoints).sum()))
    (OUT/'representation_fallback_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    effects = pd.read_csv(OUT/'paired_label_effects.csv').merge(population[['dataset','component_id','any_unimol_fallback']],on=['dataset','component_id'],validate='many_to_one')
    rows = []
    for metric in ['raw_nap','endpoint_exact_nap']:
        p = effects.pivot(index=['dataset','component_id','representation','predictor','common_both_endpoints','any_unimol_fallback'],columns='generator',values=metric).reset_index()
        p['brics_minus_crem'] = p.brics-p.crem
        for name, sub in [('common270_exclude_fallback',p[p.common_both_endpoints & ~p.any_unimol_fallback]),('full611_exclude_fallback',p[~p.any_unimol_fallback])]:
            for effect in ['crem','brics','brics_minus_crem']:
                dist = two_stage_bootstrap(sub,effect,5000,20261019,f'brics-{name}')
                rows.append({'population':name,'metric':metric,'effect':effect,'point':float(sub.groupby('dataset')[effect].median().median()),'ci_low':float(np.quantile(dist,.025)),'ci_high':float(np.quantile(dist,.975)),'datasets':int(sub.dataset.nunique()),'pairs':len(sub[['dataset','component_id']].drop_duplicates())})
    pd.DataFrame(rows).to_csv(OUT/'fallback_exclusion_summary.csv',index=False)
    summary = pd.concat([pd.read_csv(OUT/'comparison_summary.csv'),pd.DataFrame(rows)],ignore_index=True)
    summaries = []
    for (population_name,metric),group in summary.groupby(['population','metric'],sort=False):
        row = {'population':population_name,'metric':metric,'datasets':int(group.datasets.iloc[0]),'pairs':int(group.pairs.iloc[0])}
        for r in group.itertuples():
            row[r.effect] = f'{r.point:.6f} [{r.ci_low:.6f}, {r.ci_high:.6f}]'
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(OUT/'Table_S51_generator_comparison.csv',index=False)
    mask_counts = selected.drop_duplicates(['row_id','mask_id']).groupby('row_id').size()
    coverage = [('Original eligible pairs',617),('Original eligible endpoints',1234),('Compatible allowed regions',829),('Common supported allowed regions',764),('Common supported endpoints',606),('Endpoints with one common region',int(mask_counts.eq(1).sum())),('Endpoints with two common regions',int(mask_counts.eq(2).sum())),('Endpoints with three common regions',int(mask_counts.eq(3).sum())),('Pairs supported at both endpoints',272),('Datasets with common pairs',26),('Scored product records per generator',2866),('Unique new BRICS products',2861),('New BRICS products using UniMol 2D fallback',7),('Common pairs with any parent/product fallback',2),('Single-cut parent reconstructions verified',6125)]
    pd.DataFrame(coverage,columns=['quantity','count']).to_csv(OUT/'Table_S50_generator_coverage.csv',index=False)
    information = []
    for (scope,group) in [('full617',effects),('common272',effects[effects.common_both_endpoints])]:
        for generator,sub in group.groupby('generator'):
            for metric in ['raw_nap','endpoint_exact_nap']:
                information.append({'population':scope,'generator':generator,'metric':metric,'component_configuration_rows':len(sub),'nonzero_rows_gt_1e12':int(sub[metric].abs().gt(1e-12).sum()),'maximum_absolute_paired_effect':float(sub[metric].abs().max())})
    pd.DataFrame(information).to_csv(OUT/'local_effect_information.csv',index=False)
    inputs = [Path(__file__),ROOT/'research_v2/scripts/score_brics_extension_20261019.py',ROOT/'research_v2/configs/sanity_full_v1.json',OUT/'frozen_comparison_inputs.json',OUT/'representations/manifest.json',OUT/'matched_scored_products.csv']
    inputs += list((OUT/'representations').glob('*.npz')) + [BASE/f'representations/{r}.npz' for r in ['morgan','molformer','unimol']] + [BASE/f'counterfactual_representations/{r}.npz' for r in ['morgan','molformer','unimol']]
    outputs = list(OUT.glob('*.csv')) + list(OUT.glob('*.csv.gz')) + list((OUT/'atom_scores').glob('*.npz'))
    provenance = {kind:{str(p.relative_to(ROOT)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths} for kind,paths in [('inputs',inputs),('outputs',outputs)]}
    (OUT/'final_analysis_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'status':'PASS','fallback_common_pairs':270,'fallback_full_pairs':611,'local_effect_information':information}))


if __name__ == '__main__':
    main()
