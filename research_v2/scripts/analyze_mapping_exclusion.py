import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from analyze_round2_revision import two_stage_bootstrap

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_review_20261008'
OUT = ROOT / 'research_v2/artifacts/analysis/xai_sanity_cliffs_rereview_20261008'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(SOURCE / 'mcs_mapping_ambiguity.csv')
    keys = ['dataset', 'component_id']
    failed = mapping.loc[~mapping.enumeration_eligible, keys]
    assert len(mapping) == 617 and len(failed) == 6
    counts = mapping.groupby('dataset').agg(original_pairs=('component_id', 'size'),
        assessed_pairs=('enumeration_eligible', 'sum'), ambiguous_pairs=('positive_atom_set_ambiguous', 'sum')).reset_index()
    counts['unreproduced_pairs'] = counts.original_pairs - counts.assessed_pairs
    counts['ambiguous_fraction_assessed'] = counts.ambiguous_pairs / counts.assessed_pairs
    counts.to_csv(OUT / 'mapping_status_by_dataset.csv', index=False)
    prepared = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared'
    pairs = pd.read_csv(prepared / 'sentinel_pairs.csv').set_index(keys)
    molecules = pd.read_csv(prepared / 'model_molecules.csv').set_index('row_id')
    evidence = []
    for row in mapping.itertuples():
        pair = pairs.loc[(row.dataset, row.component_id)]
        stored = json.loads(pair['match'])
        atom_map = dict(stored)
        common_atoms, common_bonds = [], []
        if row.enumeration_eligible:
            query = Chem.MolFromSmarts(row.query_smarts)
            left = Chem.MolFromSmiles(molecules.loc[f'{row.dataset}:{int(pair.left_index)}', 'canonical_smiles'])
            right = Chem.MolFromSmiles(molecules.loc[f'{row.dataset}:{int(pair.right_index)}', 'canonical_smiles'])
            common_atoms = [[a.GetAtomMapNum() - 1, atom_map[a.GetAtomMapNum() - 1]] for a in query.GetAtoms()]
            assert set(map(tuple, common_atoms)) == set(map(tuple, stored))
            for bond in query.GetBonds():
                a, b = [query.GetAtomWithIdx(i).GetAtomMapNum() - 1 for i in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
                lb, rb = left.GetBondBetweenAtoms(a, b), right.GetBondBetweenAtoms(atom_map[a], atom_map[b])
                assert lb.GetBondType() == rb.GetBondType() and lb.IsInRing() == rb.IsInRing()
                common_bonds.append([lb.GetIdx(), rb.GetIdx(), a, b, atom_map[a], atom_map[b]])
            assert len(common_bonds) == row.shared_bonds
        evidence.append({'dataset': row.dataset, 'component_id': row.component_id,
            'status': 'reproduced' if row.enumeration_eligible else 'unreproduced',
            'reason': '' if row.enumeration_eligible else 'Stored atom sets were not reproduced by RASCAL with the recorded original options.',
            'stored_atom_mapping': json.dumps(stored), 'reproduced_common_atoms': json.dumps(common_atoms),
            'reproduced_common_bonds': json.dumps(common_bonds), 'shared_bonds': len(common_bonds)})
    pd.DataFrame(evidence).to_csv(OUT / 'mapping_atom_bond_evidence.csv', index=False)
    components = pd.read_csv(SOURCE / 'component_metrics.csv')
    components = components[components.denominator.eq('supported')].merge(
        failed.assign(unreproduced=True), on=keys, how='left', validate='many_to_one')
    components['unreproduced'] = components.unreproduced.eq(True)
    generators = pd.read_csv(SOURCE / 'generator_null_metrics.csv').merge(
        failed.assign(unreproduced=True), on=keys, how='left', validate='many_to_one')
    generators['unreproduced'] = generators.unreproduced.eq(True)
    original = pd.read_csv(SOURCE / 'primary_summary.csv')
    summaries, absolute, paired = [], [], []
    draws_out = pd.DataFrame({'draw': np.arange(5000)})
    for rationale, part in components.groupby('rationale'):
        kept = part[~part.unreproduced]
        excluded_count = part.loc[part.unreproduced, keys].drop_duplicates().shape[0]
        for metric in ['raw_nap', 'endpoint_exact_nap']:
            full = original[(original.rationale == rationale) & original.denominator.eq('supported') & original.metric.eq(metric)].iloc[0]
            summaries.append({'population': 'all_frozen', 'rationale': rationale, 'metric': metric,
                **{k: full[k] for k in ['estimate', 'ci_low', 'ci_high', 'datasets', 'components']}, 'excluded_pairs': 0})
            label = 'mapping-exclusion:' + rationale
            draws = two_stage_bootstrap(kept, metric + '_delta', 5000, 20261008, label)
            draws_out[rationale + ':' + metric] = draws
            result = {'population': 'reproduced_only', 'rationale': rationale, 'metric': metric,
                'estimate': float(kept.groupby('dataset')[metric + '_delta'].median().median()),
                'ci_low': float(np.quantile(draws, .025)), 'ci_high': float(np.quantile(draws, .975)),
                'datasets': kept.dataset.nunique(), 'components': kept[keys].drop_duplicates().shape[0], 'excluded_pairs': excluded_count}
            summaries.append(result)
            print(json.dumps(result), flush=True)
        for population, model_part in [('all_frozen', part), ('reproduced_only', kept)]:
            gen_part = generators[generators.rationale.eq(rationale)]
            if population == 'reproduced_only':
                gen_part = gen_part[~gen_part.unreproduced]
            for metric in ['raw_nap', 'endpoint_exact_nap']:
                for (rep, pred), cell in model_part.groupby(['representation', 'predictor']):
                    for variant in ['trained', 'permuted']:
                        absolute.append({'population': population, 'rationale': rationale, 'metric': metric,
                            'representation': rep, 'predictor': pred, 'source': variant,
                            'estimate': cell.groupby('dataset')[metric + '_' + variant].median().median()})
                    for baseline, gen in gen_part.groupby('baseline'):
                        joint = cell.merge(gen[keys + [metric]], on=keys, validate='one_to_one')
                        delta = joint.assign(difference=joint[metric + '_trained'] - joint[metric])
                        paired.append({'population': population, 'rationale': rationale, 'metric': metric,
                            'representation': rep, 'predictor': pred, 'baseline': baseline,
                            'paired_difference': delta.groupby('dataset').difference.median().median()})
                for baseline, gen in gen_part.groupby('baseline'):
                    absolute.append({'population': population, 'rationale': rationale, 'metric': metric,
                        'representation': '', 'predictor': '', 'source': baseline,
                        'estimate': gen.groupby('dataset')[metric].median().median()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'mapping_exclusion_primary_summary.csv', index=False)
    draws_out.to_csv(OUT / 'mapping_exclusion_bootstrap.csv', index=False)
    pd.DataFrame(absolute).to_csv(OUT / 'mapping_exclusion_absolute.csv', index=False)
    pd.DataFrame(paired).to_csv(OUT / 'mapping_exclusion_model_minus_generator.csv', index=False)
    components.loc[components.unreproduced].to_csv(OUT / 'excluded_component_rows.csv', index=False)
    by_dataset = components.groupby(['dataset', 'rationale']).apply(lambda x: pd.Series({
        'original_pairs': x.component_id.nunique(), 'retained_pairs': x.loc[~x.unreproduced, 'component_id'].nunique(),
        **{metric + '_' + pop: group[metric + '_delta'].median() for metric in ['raw_nap', 'endpoint_exact_nap']
           for pop, group in [('all', x), ('retained', x[~x.unreproduced])]}}), include_groups=False).reset_index()
    by_dataset.to_csv(OUT / 'mapping_exclusion_by_dataset.csv', index=False)
    raw = summary[summary.rationale.eq('boundary_extended') & summary.metric.eq('raw_nap')]
    assert np.ptp(raw.estimate) < 1e-14
    status = {'status': 'completed', 'review_item': 'P2-P3', 'original_pairs': 617, 'assessed_pairs': 611,
        'unreproduced_pairs': 6, 'ambiguous_pairs': int(counts.ambiguous_pairs.sum()),
        'strict_original_pairs': 581,
        'strict_excluded_pairs': int(summary[summary.rationale.eq('strict_unmatched')].excluded_pairs.max()),
        'strict_retained_pairs': int(summary[summary.rationale.eq('strict_unmatched') & summary.population.eq('reproduced_only')].components.iloc[0]),
        'raw_point_unchanged': True, 'bootstrap_draws': 5000, 'bootstrap_seed': 20261008,
        'bootstrap_label': 'mapping-exclusion:<rationale>', 'full_population_intervals': 'Existing revision draws; historical principal raw interval remains unchanged in Table 1.',
        'scope': 'Frozen scores and labels; component-safe target bootstrap; no model refit; unreproduced does not mean invalid label.'}
    (OUT / 'evaluation_summary.json').write_text(json.dumps(status, indent=2) + '\n')
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE / n for n in
        ['mcs_mapping_ambiguity.csv', 'component_metrics.csv', 'generator_null_metrics.csv', 'primary_summary.csv']]}
    inputs.update({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [prepared / 'sentinel_pairs.csv', prepared / 'model_molecules.csv']})
    (OUT / 'input_provenance.json').write_text(json.dumps(inputs, indent=2) + '\n')
    (OUT / 'manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'manifest.json'}, indent=2) + '\n')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
