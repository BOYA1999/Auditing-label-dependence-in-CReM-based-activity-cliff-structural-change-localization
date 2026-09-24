import argparse
import hashlib
import json
import random
import time
import zlib
from pathlib import Path

import crem.crem as crem_core
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research_v2/configs/pilot_v1.json"
BASE = ROOT / "research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1"
PREPARED = BASE / "prepared"
DEFAULT_DB = BASE / "full/crem_db/replacement_library.db"
DEFAULT_OUT = BASE / "full/counterfactuals"
ORIGINAL_FRAGMENT_MOL = crem_core.__fragment_mol


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equivalent_assignment(source, target, equivalent):
    remaining = set(target)
    ordered = sorted(source, key=lambda atom_idx: len(equivalent[atom_idx] & remaining))

    def assign(position):
        if position == len(ordered):
            return True
        atom_idx = ordered[position]
        for candidate in sorted(equivalent[atom_idx] & remaining):
            remaining.remove(candidate)
            if assign(position + 1):
                return True
            remaining.add(candidate)
        return False

    return assign(0)


def fragment_mol_fast(mol, radius=3, return_ids=True, keep_stereo=False, protected_ids=None, symmetry_fixes=False):
    if not symmetry_fixes:
        return ORIGINAL_FRAGMENT_MOL(
            mol, radius=radius, return_ids=return_ids, keep_stereo=keep_stereo,
            protected_ids=protected_ids, symmetry_fixes=False,
        )
    if protected_ids:
        return_ids = True

    def atom_ids(fragment):
        return tuple(sorted(atom.GetIntProp("Index") for atom in fragment.GetAtoms() if atom.GetAtomicNum()))

    if return_ids:
        for atom in mol.GetAtoms():
            atom.SetIntProp("Index", atom.GetIdx())
    raw = crem_core.rdMMPA.FragmentMol(
        mol, pattern="[!#1]!@!=!#[!#1]", maxCuts=4, resultsAsMols=True, maxCutBonds=30,
    )
    raw += crem_core.rdMMPA.FragmentMol(
        mol, pattern="[!#1]!@!=!#[!#1]", maxCuts=3, resultsAsMols=True, maxCutBonds=30,
    )
    raw += crem_core.rdMMPA.FragmentMol(
        mol, pattern="[#1]!@!=!#[!#1]", maxCuts=1, resultsAsMols=True, maxCutBonds=100,
    )
    output = set()
    for core, chains in raw:
        if core is None:
            components = list(Chem.GetMolFrags(chains, asMols=True))
            ids_0 = atom_ids(components[0]) if return_ids else tuple()
            ids_1 = atom_ids(components[1]) if return_ids else tuple()
            if Chem.MolToSmiles(components[0]) != "[H][*:1]":
                env, fragment = crem_core.get_canon_context_core(components[0], components[1], radius, keep_stereo)
                output.add((env, fragment, ids_1))
            if Chem.MolToSmiles(components[1]) != "[H][*:1]":
                env, fragment = crem_core.get_canon_context_core(components[1], components[0], radius, keep_stereo)
                output.add((env, fragment, ids_0))
        else:
            env, fragment = crem_core.get_canon_context_core(chains, core, radius, keep_stereo)
            output.add((env, fragment, atom_ids(core) if return_ids else tuple()))

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False, includeIsotopes=False))
    equivalent = {}
    for atom_idx, rank in enumerate(ranks):
        peers = [idx for idx, candidate_rank in enumerate(ranks) if candidate_rank == rank and idx != atom_idx]
        if peers:
            equivalent[atom_idx] = set(peers)
    extended = []
    for env, core, ids in output:
        if not ids or not all(atom_idx in equivalent for atom_idx in ids):
            continue
        smarts = crem_core.patt_remove_brackets.sub("", crem_core.patt_remove_map.sub("", core))
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        for match in mol.GetSubstructMatches(query):
            target = set(match)
            if len(target) == len(ids) and equivalent_assignment(ids, target, equivalent):
                extended.append((env, core, tuple(sorted(target))))
    output.update(extended)
    if protected_ids:
        protected = set(protected_ids)
        output = [item for item in output if protected.isdisjoint(item[2])]
    return list(output)


crem_core.__fragment_mol = fragment_mol_fast


def candidate_masks(mol, radius, max_atoms):
    fragments = fragment_mol_fast(mol, radius=radius, return_ids=True, symmetry_fixes=True)
    masks = {tuple(ids) for _, _, ids in fragments if 1 <= len(ids) <= max_atoms}
    masks = sorted(masks, key=lambda value: (len(value), value))
    selected = set()
    for atom_idx in range(mol.GetNumAtoms()):
        containing = [mask for mask in masks if atom_idx in mask]
        if containing:
            selected.add(containing[0])
    return sorted(selected, key=lambda value: (len(value), value))


def read_jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--targets", nargs="+")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    targets = args.targets or cfg["targets"]
    crem_cfg = cfg["crem"]
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"
    molecules = pd.read_csv(args.prepared / "molecules.csv")
    pairs = pd.read_csv(args.prepared / "sentinel_pairs.csv")
    pairs = pairs[pairs.dataset.isin(targets)]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    started = time.time()

    eval_ids = set()
    for pair in pairs.itertuples(index=False):
        eval_ids.add(f"{pair.dataset}:{int(pair.left_index)}")
        eval_ids.add(f"{pair.dataset}:{int(pair.right_index)}")
    evaluation = molecules[molecules.row_id.isin(eval_ids)].sort_values("row_id")
    if len(evaluation) != len(eval_ids):
        raise ValueError("evaluation molecule lookup mismatch")

    target_thresholds = {}
    train_fps = {}
    for target in targets:
        target_rows = molecules[molecules.dataset == target]
        train = target_rows[target_rows.component_split.isin(["train", "isolated"])]
        calibration = target_rows[target_rows.component_split == "calibration"]
        train_fps[target] = [generator.GetFingerprint(Chem.MolFromSmiles(value)) for value in train.canonical_smiles]
        nearest = [
            max(DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(Chem.MolFromSmiles(value)), train_fps[target]))
            for value in calibration.canonical_smiles
        ]
        target_thresholds[target] = float(np.quantile(nearest, 0.05))

    all_mask_rows = []
    all_mutant_rows = []
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        log.write(f"resume={time.strftime('%Y-%m-%dT%H:%M:%S%z')} molecules={len(evaluation)}\n")
        for target in targets:
            target_dir = args.output / target
            target_dir.mkdir(exist_ok=True)
            final_masks = target_dir / "mask_coverage.csv"
            final_mutants = target_dir / "mutants.csv"
            target_eval = evaluation[evaluation.dataset == target]
            target_expected = set(target_eval.row_id)
            seed_mask_rows = []
            seed_mutant_rows = []
            seed_completed = set()
            if final_masks.is_file() and final_mutants.is_file():
                target_mask_rows = pd.read_csv(final_masks).to_dict("records")
                target_mutant_rows = pd.read_csv(final_mutants).to_dict("records")
                final_completed = {row["row_id"] for row in target_mask_rows}
                if target_expected <= final_completed:
                    all_mask_rows.extend(target_mask_rows)
                    all_mutant_rows.extend(target_mutant_rows)
                    print(f"{target} reused masks={len(target_mask_rows)} mutants={len(target_mutant_rows)}", flush=True)
                    continue
                seed_mask_rows = target_mask_rows
                seed_mutant_rows = target_mutant_rows
                seed_completed = final_completed
                print(f"{target} repairing missing={len(target_expected - final_completed)}", flush=True)
            partial = target_dir / "partial"
            partial.mkdir(exist_ok=True)
            mask_jsonl = partial / "mask_rows.jsonl"
            mutant_jsonl = partial / "mutant_rows.jsonl"
            completed_path = partial / "completed_row_ids.txt"
            target_mask_rows = seed_mask_rows + read_jsonl(mask_jsonl)
            target_mutant_rows = seed_mutant_rows + read_jsonl(mutant_jsonl)
            completed = seed_completed | (set(completed_path.read_text(encoding="utf-8").splitlines()) if completed_path.is_file() else set())
            for number, record in enumerate(target_eval.itertuples(index=False), 1):
                if record.row_id in completed:
                    continue
                mol = Chem.MolFromSmiles(record.canonical_smiles)
                parent_fp = generator.GetFingerprint(mol)
                parent_nearest = max(DataStructs.BulkTanimotoSimilarity(parent_fp, train_fps[target]))
                size_guard = int(record.heavy_atoms) > crem_cfg["max_explainer_heavy_atoms"]
                masks = [] if size_guard else candidate_masks(mol, crem_cfg["radius"], crem_cfg["max_replaced_atoms"])
                maskable = set().union(*(set(mask) for mask in masks)) if masks else set()
                molecule_mask_rows = []
                molecule_mutant_rows = []
                log.write(f"BEGIN target={target} row_id={record.row_id} position={number}/{len(target_eval)} masks={len(masks)}\n")
                log.flush()
                if not masks:
                    molecule_mask_rows.append(
                        {
                            "row_id": record.row_id,
                            "dataset": target,
                            "molecule_index": int(record.molecule_index),
                            "component_split": record.component_split,
                            "parent_atoms": mol.GetNumAtoms(),
                            "maskable_atom_fraction": 0.0,
                            "mask_id": -1,
                            "mask_atoms": "[]",
                            "mask_size": 0,
                            "valid_unique_mutants": 0,
                            "local_supported_mutants": 0,
                            "absolute_in_domain_mutants": 0,
                            "covered": False,
                            "failure": "size_guard" if size_guard else "no_replaceable_mask",
                        }
                    )
                for mask_id, mask in enumerate(masks):
                    log.write(f"MASK target={target} row_id={record.row_id} mask_id={mask_id} atoms={mask}\n")
                    log.flush()
                    random.seed(zlib.crc32(f"{record.row_id}:{mask}".encode()))
                    failure = ""
                    try:
                        generated = crem_core.mutate_mol(
                            mol,
                            db_name=str(args.db),
                            radius=crem_cfg["radius"],
                            min_size=1,
                            max_size=crem_cfg["max_replaced_atoms"],
                            min_inc=crem_cfg["min_inc"],
                            max_inc=crem_cfg["max_inc"],
                            max_replacements=crem_cfg["max_replacements_per_mask"],
                            replace_ids=mask,
                            symmetry_fixes=True,
                            min_freq=crem_cfg["min_fragment_frequency"],
                        )
                        unique = {}
                        for value in generated:
                            mutant = Chem.MolFromSmiles(value)
                            if mutant is None or len(Chem.GetMolFrags(mutant)) != 1:
                                continue
                            canonical = Chem.MolToSmiles(mutant, canonical=True, isomericSmiles=True)
                            if canonical != record.canonical_smiles:
                                unique[canonical] = mutant
                    except Exception as error:
                        unique = {}
                        failure = f"{type(error).__name__}:{error}"
                    mutation_rows = []
                    for canonical, mutant in unique.items():
                        mutant_fp = generator.GetFingerprint(mutant)
                        parent_tanimoto = DataStructs.TanimotoSimilarity(parent_fp, mutant_fp)
                        nearest_train = max(DataStructs.BulkTanimotoSimilarity(mutant_fp, train_fps[target]))
                        local_supported = parent_tanimoto >= crem_cfg["local_parent_tanimoto_threshold"]
                        mutation_rows.append(
                            {
                                "row_id": record.row_id,
                                "dataset": target,
                                "molecule_index": int(record.molecule_index),
                                "component_split": record.component_split,
                                "mask_id": mask_id,
                                "mask_atoms": json.dumps(mask),
                                "mask_size": len(mask),
                                "mutant_smiles": canonical,
                                "parent_mutant_tanimoto": parent_tanimoto,
                                "parent_nearest_train_tanimoto": parent_nearest,
                                "nearest_train_tanimoto": nearest_train,
                                "absolute_in_domain": nearest_train >= target_thresholds[target],
                                "local_supported": local_supported,
                            }
                        )
                    supported = [row for row in mutation_rows if row["local_supported"]]
                    supported.sort(key=lambda row: hashlib.sha256(f"{record.row_id}:{mask_id}:{row['mutant_smiles']}".encode()).hexdigest())
                    scored = {row["mutant_smiles"] for row in supported[: crem_cfg["pilot_scored_replacements_per_mask"]]}
                    for row in mutation_rows:
                        row["scored_in_pilot"] = row["mutant_smiles"] in scored
                    molecule_mutant_rows.extend(mutation_rows)
                    molecule_mask_rows.append(
                        {
                            "row_id": record.row_id,
                            "dataset": target,
                            "molecule_index": int(record.molecule_index),
                            "component_split": record.component_split,
                            "parent_atoms": mol.GetNumAtoms(),
                            "maskable_atom_fraction": len(maskable) / mol.GetNumAtoms(),
                            "mask_id": mask_id,
                            "mask_atoms": json.dumps(mask),
                            "mask_size": len(mask),
                            "valid_unique_mutants": len(unique),
                            "local_supported_mutants": len(supported),
                            "absolute_in_domain_mutants": sum(row["absolute_in_domain"] for row in mutation_rows),
                            "covered": bool(supported),
                            "failure": failure,
                        }
                    )
                with mask_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                    for row in molecule_mask_rows:
                        handle.write(json.dumps(row) + "\n")
                with mutant_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                    for row in molecule_mutant_rows:
                        handle.write(json.dumps(row) + "\n")
                with completed_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(record.row_id + "\n")
                completed.add(record.row_id)
                target_mask_rows.extend(molecule_mask_rows)
                target_mutant_rows.extend(molecule_mutant_rows)
                log.write(f"DONE target={target} row_id={record.row_id} masks={len(molecule_mask_rows)} mutants={len(molecule_mutant_rows)}\n")
                log.flush()
                if len(completed) % 10 == 0 or len(completed) == len(target_eval):
                    message = f"{target} molecules={len(completed)}/{len(target_eval)} masks={len(target_mask_rows)} mutants={len(target_mutant_rows)}"
                    print(message, flush=True)
                    log.write(message + "\n")
                    log.flush()
            if len(completed) != len(target_eval):
                raise RuntimeError(f"incomplete target {target}: {len(completed)}/{len(target_eval)}")
            pd.DataFrame(target_mask_rows).to_csv(final_masks, index=False)
            pd.DataFrame(target_mutant_rows).to_csv(final_mutants, index=False)
            all_mask_rows.extend(target_mask_rows)
            all_mutant_rows.extend(target_mutant_rows)

    masks = pd.DataFrame(all_mask_rows)
    mutants = pd.DataFrame(all_mutant_rows)
    masks.to_csv(args.output / "mask_coverage.csv", index=False)
    mutants.to_csv(args.output / "mutants.csv", index=False)
    scored = mutants[mutants.scored_in_pilot.astype(bool)].copy()
    scored.to_csv(args.output / "scored_mutants.csv", index=False)
    unique = scored[["mutant_smiles"]].drop_duplicates().sort_values("mutant_smiles")
    unique.insert(0, "row_id", [f"mutant:{idx}" for idx in range(len(unique))])
    unique.rename(columns={"mutant_smiles": "canonical_smiles"}).to_csv(args.output / "unique_scored_mutants.csv", index=False)
    expected_failures = {"", "size_guard", "no_replaceable_mask"}
    unexpected_failure_count = int((~masks.failure.fillna("").isin(expected_failures)).sum())
    summary = {
        "run_id": cfg["run_id"],
        "targets": targets,
        "status": "success" if len(masks) and unexpected_failure_count == 0 else "partial",
        "evaluation_molecules": int(evaluation.row_id.nunique()),
        "candidate_masks": int(len(masks)),
        "valid_unique_mutants": int(masks.valid_unique_mutants.sum()),
        "local_supported_mutants": int(masks.local_supported_mutants.sum()),
        "scored_mutant_rows": int(len(scored)),
        "unique_scored_mutants": int(len(unique)),
        "mask_coverage": float(masks.covered.mean()),
        "molecule_coverage": float(masks.groupby("row_id").covered.any().mean()),
        "test_mask_coverage": float(masks[masks.component_split == "test"].covered.mean()),
        "absolute_mask_coverage": float((masks.absolute_in_domain_mutants > 0).mean()),
        "median_maskable_atom_fraction": float(masks.groupby("row_id").maskable_atom_fraction.first().median()),
        "generation_failures": int(masks.failure.fillna("").astype(bool).sum()),
        "size_guard_failures": int((masks.failure == "size_guard").sum()),
        "unexpected_generation_failures": unexpected_failure_count,
        "absolute_domain_thresholds": target_thresholds,
        "database_sha256": sha256(args.db),
        "config_sha256": sha256(args.config),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] == "partial":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
