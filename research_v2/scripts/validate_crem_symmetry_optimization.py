import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script = ROOT / "research_v2/scripts/generate_counterfactuals.py"
    spec = importlib.util.spec_from_file_location("counterfactual_generator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.read_csv(args.molecules)
    sample = frame[frame.heavy_atoms.le(30)].drop_duplicates("canonical_smiles").sample(n=60, random_state=20260903)
    symmetric = [
        "c1ccccc1",
        "c1ccc(-c2ccccc2)cc1",
        "c1ccc2ccccc2c1",
        "CC(C)(C)c1ccc(C(C)(C)C)cc1",
        "O=C(O)Cc1ccc(CC(=O)O)cc1",
        "CCOc1ccc(OCC)cc1",
    ]
    checked = list(sample.canonical_smiles) + symmetric
    for smiles in checked:
        mol = Chem.MolFromSmiles(smiles)
        reference = module.ORIGINAL_FRAGMENT_MOL(mol, radius=1, return_ids=True, symmetry_fixes=True)
        optimized = module.fragment_mol_fast(mol, radius=1, return_ids=True, symmetry_fixes=True)
        if reference != optimized:
            raise AssertionError(f"fragment tuple or order mismatch: {smiles}")
    result = {
        "status": "pass",
        "random_molecules": 60,
        "symmetric_molecules": 6,
        "checks": ["exact tuple-set equality", "exact returned-list order equality"],
        "molecules_sha256": sha256(args.molecules),
        "generator_script_sha256": sha256(script),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
