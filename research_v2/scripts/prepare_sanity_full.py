import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research_v2/configs/sanity_full_v1.json"
DATA = ROOT / "external/MoleculeACE/MoleculeACE/Data/benchmark_data"
OLD = ROOT / "artifacts/experiment/molcles_v1_main/cache"
DEFAULT_OUT = ROOT / "research_v2/artifacts/experiment/xai_sanity_cliffs_full_v1/prepared"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pair-cache", type=Path, default=OLD)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    split_rows = pd.read_csv(args.pair_cache / "molecule_splits.csv").sort_values(["dataset", "molecule_index"])
    molecule_rows = []
    library_candidates = set()
    heldout = set()
    dataset_hashes = {}
    for target in cfg["targets"]:
        path = DATA / f"{target}.csv"
        frame = pd.read_csv(path)
        splits = split_rows[split_rows.dataset == target].sort_values("molecule_index")
        if len(frame) != len(splits) or not np.array_equal(splits.molecule_index.to_numpy(), np.arange(len(frame))):
            raise ValueError(f"split/data order mismatch: {target}")
        dataset_hashes[target] = sha256(path)
        for index, (smiles, y, component_id, component_split) in enumerate(zip(frame.smiles, frame.y, splits.component_id, splits.component_split)):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"invalid SMILES: {target}:{index}")
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if component_split in {"calibration", "test"}:
                heldout.add(canonical)
            else:
                library_candidates.add(canonical)
            molecule_rows.append({
                "row_id": f"{target}:{index}", "dataset": target, "molecule_index": index,
                "smiles": smiles, "canonical_smiles": canonical, "y": float(y),
                "component_id": int(component_id), "component_split": component_split,
                "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            })
    molecules = pd.DataFrame(molecule_rows).sort_values(["dataset", "molecule_index"])
    pairs = pd.read_csv(args.pair_cache / "pairs.csv")
    selected_pairs = pairs[
        pairs.dataset.isin(cfg["targets"]) & pairs.is_sentinel.astype(bool)
        & pairs.rationale_valid.astype(bool) & pairs.component_split.isin(["calibration", "test"])
    ].copy()
    evaluation_ids = set()
    for pair in selected_pairs.itertuples(index=False):
        evaluation_ids.update([f"{pair.dataset}:{int(pair.left_index)}", f"{pair.dataset}:{int(pair.right_index)}"])
    evaluation = molecules[molecules.row_id.isin(evaluation_ids)].copy()
    library = sorted(library_candidates - heldout)
    molecules.to_csv(args.output / "molecules.csv", index=False)
    evaluation.to_csv(args.output / "evaluation_molecules.csv", index=False)
    selected_pairs.to_csv(args.output / "sentinel_pairs.csv", index=False)
    with (args.output / "replacement_library.smi").open("w", encoding="utf-8", newline="\n") as handle:
        for index, smiles in enumerate(library):
            handle.write(f"{smiles}\tlib_{index}\n")
    audit = {
        "status": "pass" if not (set(library) & heldout) and set(cfg["development_targets"]).isdisjoint(cfg["confirmation_targets"]) else "fail",
        "targets": len(cfg["targets"]), "development_targets": len(cfg["development_targets"]),
        "confirmation_targets": len(cfg["confirmation_targets"]), "molecules": int(len(molecules)),
        "evaluation_molecules": int(len(evaluation)), "sentinel_pairs": int(len(selected_pairs)),
        "calibration_pairs": int((selected_pairs.component_split == "calibration").sum()),
        "test_pairs": int((selected_pairs.component_split == "test").sum()),
        "replacement_library_smiles": len(library), "heldout_smiles": len(heldout),
        "library_heldout_overlap": len(set(library) & heldout), "dataset_sha256": dataset_hashes,
    }
    (args.output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": cfg["run_id"], "config_sha256": sha256(args.config),
        "pair_cache": str(args.pair_cache.resolve()), "audit": audit,
        "outputs": {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()},
    }
    (args.output / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    if audit["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
