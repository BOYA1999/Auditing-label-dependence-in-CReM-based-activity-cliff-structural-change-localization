import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import BRICS, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1'
OUT = ROOT / 'research_v2/artifacts/analysis/extension_20261019/brics'
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ZIP = Chem.MolzipParams()
ZIP.label = Chem.MolzipLabel.Isotope
COMPATIBLE = {tuple(sorted((int(re.sub('[^0-9]', '', a)), int(re.sub('[^0-9]', '', b))))) for group in BRICS.reactionDefs for a, b, _ in group}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def filehash(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def smiles(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def join(left, right):
    fragments = [Chem.Mol(left), Chem.Mol(right)]
    for fragment in fragments:
        dummy = [a for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(dummy) == 1
        dummy[0].SetIsotope(1)
    product = Chem.molzip(*fragments, ZIP)
    Chem.SanitizeMol(product)
    return product


def cut(mol, specification):
    indices, labels = specification
    bond = mol.GetBondBetweenAtoms(*indices)
    fragmented = Chem.FragmentOnBonds(mol, [bond.GetIdx()])
    by_atom = dict(zip(indices, map(int, labels)))
    for atom in list(fragmented.GetAtoms())[mol.GetNumAtoms():]:
        atom.SetIsotope(by_atom[atom.GetNeighbors()[0].GetIdx()])
    return Chem.GetMolFrags(fragmented, asMols=True)


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'research_v2/configs/sanity_full_v1.json').read_text())
    molecules = pd.read_csv(BASE / 'prepared/model_molecules.csv').set_index('row_id')
    pairs = pd.read_csv(BASE / 'full/sanity_explanations/pair_explanations.csv')
    pairs = pairs[pairs.dataset.isin(cfg['confirmation_targets']) & pairs.component_split.eq('test') & pairs.representation.eq('morgan') & pairs.predictor.eq('xgboost') & pairs.variant.eq('trained') & pairs.seed.eq(42) & pairs.explainer.eq('crem_mean')]
    assert len(pairs) == 617
    ids = {f'{r.dataset}:{int(i)}' for r in pairs.itertuples() for i in [r.left_index, r.right_index]}
    masks = pd.read_csv(BASE / 'full/counterfactuals/mask_coverage.csv')
    masks = masks[masks.row_id.isin(ids)].copy()
    source = BASE / 'prepared/replacement_library.smi'
    source_smiles = [line.split()[0] for line in source.read_text().splitlines() if line.strip()]
    assert not set(source_smiles) & set(molecules.loc[sorted(ids)].canonical_smiles)
    library_path = OUT / 'brics_library_single_bond.csv'
    if library_path.exists():
        library = pd.read_csv(library_path)
    else:
        counts = Counter()
        for i, value in enumerate(source_smiles):
            mol = Chem.MolFromSmiles(value)
            for bond in BRICS.FindBRICSBonds(mol):
                if mol.GetBondBetweenAtoms(*bond[0]).GetBondType() != Chem.BondType.SINGLE:
                    continue
                for fragment in cut(mol, bond):
                    if 1 <= fragment.GetNumHeavyAtoms() <= 10:
                        counts[Chem.MolToSmiles(fragment, isomericSmiles=True)] += 1
            if i % 2000 == 0:
                print(f'library {i}/{len(source_smiles)} fragments={len(counts)}', flush=True)
        library = pd.DataFrame([{'fragment_smiles': value, 'frequency': count} for value, count in sorted(counts.items()) if count >= 2])
        library.to_csv(library_path, index=False)
    donors = []
    for row in library.itertuples():
        mol = Chem.MolFromSmiles(row.fragment_smiles)
        dummy = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        assert len(dummy) == 1
        donors.append((row.fragment_smiles, mol, dummy[0].GetIsotope(), mol.GetNumHeavyAtoms(), row.frequency))
    thresholds = json.loads((BASE / 'full/counterfactuals/summary.json').read_text())['absolute_domain_thresholds']
    train_fps = {target: [FP.GetFingerprint(Chem.MolFromSmiles(s)) for s in molecules.loc[molecules.dataset.eq(target) & molecules.component_split.isin(['train', 'isolated']), 'canonical_smiles']] for target in cfg['confirmation_targets']}
    coverage, products, failures = [], [], []
    reconstructed = 0
    for k, (row_id, group) in enumerate(masks.groupby('row_id', sort=True)):
        record = molecules.loc[row_id]
        mol = Chem.MolFromSmiles(record.canonical_smiles)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        terminal = {}
        for bond in BRICS.FindBRICSBonds(mol):
            if mol.GetBondBetweenAtoms(*bond[0]).GetBondType() != Chem.BondType.SINGLE:
                continue
            fragments = cut(mol, bond)
            assert len(fragments) == 2
            rebuilt = Chem.MolFromSmiles(smiles(join(*fragments)))
            original = Chem.MolFromSmiles(record.canonical_smiles)
            assert rebuilt.GetNumAtoms() == original.GetNumAtoms() and rebuilt.GetNumBonds() == original.GetNumBonds()
            assert rebuilt.HasSubstructMatch(original, useChirality=True) and original.HasSubstructMatch(rebuilt, useChirality=True)
            reconstructed += 1
            for removed, retained in [fragments, fragments[::-1]]:
                removed_ids = tuple(sorted(a.GetAtomMapNum() - 1 for a in removed.GetAtoms() if a.GetAtomicNum()))
                terminal[removed_ids] = (retained, [a.GetIsotope() for a in retained.GetAtoms() if not a.GetAtomicNum()][0])
        parent_fp = FP.GetFingerprint(Chem.MolFromSmiles(record.canonical_smiles))
        for mask in group.itertuples():
            selected = tuple(json.loads(mask.mask_atoms))
            row = {'row_id': row_id, 'dataset': record.dataset, 'component_id': int(record.component_id), 'mask_id': int(mask.mask_id), 'mask_atoms': mask.mask_atoms, 'crem_supported': bool(mask.covered), 'brics_compatible': selected in terminal, 'valid_products': 0, 'supported_products': 0, 'failure': ''}
            if selected not in terminal or not 1 <= len(selected) <= 8 or mol.GetNumHeavyAtoms() > 60:
                row['failure'] = 'no_matching_terminal_brics_cut'
                coverage.append(row)
                continue
            retained, label = terminal[selected]
            candidates = [d for d in donors if -2 <= d[3] - len(selected) <= 2 and tuple(sorted((label, d[2]))) in COMPATIBLE]
            candidates.sort(key=lambda d: digest(f'{row_id}:{mask.mask_id}:{d[0]}'))
            unique = {}
            for fragment_smiles, donor, _, _, frequency in candidates:
                try:
                    product = join(retained, donor)
                    kept = {a.GetAtomMapNum() - 1 for a in product.GetAtoms() if a.GetAtomMapNum()}
                    assert kept == set(range(mol.GetNumAtoms())) - set(selected)
                    value = smiles(product)
                    if value == record.canonical_smiles or value in unique:
                        continue
                    unique[value] = (product, fragment_smiles, frequency)
                    if len(unique) == 8:
                        break
                except Exception as error:
                    failures.append({'row_id': row_id, 'mask_id': int(mask.mask_id), 'fragment': fragment_smiles, 'error': repr(error)})
            local = []
            for value, (product, fragment_smiles, frequency) in unique.items():
                fp = FP.GetFingerprint(product)
                similarity = DataStructs.TanimotoSimilarity(parent_fp, fp)
                nearest = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps[record.dataset]))
                local.append({'row_id': row_id, 'dataset': record.dataset, 'component_id': int(record.component_id), 'mask_id': int(mask.mask_id), 'mask_atoms': mask.mask_atoms, 'mutant_smiles': value, 'donor_fragment': fragment_smiles, 'donor_frequency': frequency, 'parent_mutant_tanimoto': similarity, 'nearest_train_tanimoto': nearest, 'absolute_in_domain': nearest >= thresholds[record.dataset], 'local_supported': similarity >= 0.6})
            supported = sorted([r for r in local if r['local_supported']], key=lambda r: digest(f"{row_id}:{mask.mask_id}:{r['mutant_smiles']}"))
            scored = {r['mutant_smiles'] for r in supported[:4]}
            for r in local:
                r['scored'] = r['mutant_smiles'] in scored
            row.update(valid_products=len(unique), supported_products=len(supported), failure='' if supported else 'no_local_supported_product')
            coverage.append(row)
            products.extend(local)
        if k % 100 == 0:
            print(f'parents {k}/{len(ids)} products={len(products)} elapsed={time.time()-start:.1f}s', flush=True)
    coverage = pd.DataFrame(coverage)
    products = pd.DataFrame(products)
    coverage['common_supported'] = coverage.crem_supported & coverage.supported_products.gt(0)
    coverage.to_csv(OUT / 'mask_coverage.csv', index=False)
    products.to_csv(OUT / 'products.csv', index=False)
    pd.DataFrame(failures, columns=['row_id', 'mask_id', 'fragment', 'error']).to_csv(OUT / 'generation_failures.csv', index=False)
    support = set(coverage.loc[coverage.common_supported, 'row_id'])
    pairs['common_both_endpoints'] = [f'{r.dataset}:{r.left_index}' in support and f'{r.dataset}:{r.right_index}' in support for r in pairs.itertuples()]
    pairs[['dataset', 'component_id', 'left_index', 'right_index', 'common_both_endpoints']].to_csv(OUT / 'frozen_pair_population.csv', index=False)
    by_target = pairs.groupby('dataset').common_both_endpoints.agg(['size', 'sum'])
    by_target.to_csv(OUT / 'pair_coverage_by_target.csv')
    gate = int(pairs.common_both_endpoints.sum()) >= 50 and int(by_target['sum'].ge(5).sum()) >= 10 and not failures
    summary = {'status': 'PASS_FEASIBILITY' if gate else 'STOP_FEASIBILITY', 'baseline_pairs': 617, 'baseline_parents': len(ids), 'library_source_molecules': len(source_smiles), 'library_fragments_frequency_ge_2': len(library), 'compatible_masks': int(coverage.brics_compatible.sum()), 'common_supported_masks': int(coverage.common_supported.sum()), 'common_supported_parents': len(support), 'both_supported_pairs': int(pairs.common_both_endpoints.sum()), 'targets_with_5_pairs': int(by_target['sum'].ge(5).sum()), 'products': len(products), 'unexpected_failures': len(failures), 'parent_cut_reassembly_checks': reconstructed, 'rdkit': rdBase.rdkitVersion, 'elapsed_seconds': time.time()-start, 'scientific_outcomes_accessed': False, 'source_sha256': {str(p.relative_to(ROOT)): filehash(p) for p in [source, Path(__file__), BASE / 'prepared/model_molecules.csv', BASE / 'prepared/sentinel_pairs.csv', BASE / 'full/counterfactuals/mask_coverage.csv']}}
    (OUT / 'feasibility.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
