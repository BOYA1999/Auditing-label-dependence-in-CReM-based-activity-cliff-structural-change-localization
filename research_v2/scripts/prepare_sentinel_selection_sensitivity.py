import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.prepare_molcles_pairs import rationale


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"input hash mismatch: {path}: {observed} != {expected}")


def expected_preparation_counts(spec):
    legacy = {
        "targets": 6,
        "source_test_components": 140,
        "selected_components": 139,
        "source_components_outside_frozen_analysis": 1,
        "localization_eligible_components": 139,
        "changed_from_original": 76,
        "unchanged_from_original": 63,
        "evaluation_endpoints": 278,
        "seed42_heads": 72,
    }
    return spec.get("expected_counts", {}).get("preparation", legacy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["sentinel_selection_sensitivity"]
    inputs = {key: resolve(value) for key, value in spec["inputs"].items()}
    expected = spec["input_sha256"]
    rationale_config = resolve(spec["rationale"]["config"])
    checks = {
        "base_config": inputs["base_config"],
        "rationale_config": rationale_config,
        "pair_cache": inputs["pair_cache"],
        "molecule_splits": inputs["molecule_splits"],
        "source_molecules": inputs["source_molecules"],
        "model_molecules": inputs["model_molecules"],
        "original_sentinel_pairs": inputs["original_sentinel_pairs"],
        "replacement_library": inputs["replacement_library"],
        "crem_database": inputs["crem_database"],
        "original_representation_manifest": inputs["original_representations"] / "manifest.json",
        "original_unique_mutants": inputs["original_unique_mutants"],
        "original_mutant_representation_manifest": inputs["original_mutant_representations"] / "manifest.json",
        "model_manifest": inputs["model_heads"] / "model_manifest.csv",
        "original_pair_explanations": inputs["original_pair_explanations"],
    }
    for name, path in checks.items():
        require_hash(path, expected[name])

    output = resolve(spec["experiment_root"])
    prepared = output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    targets = spec["representative_targets"]
    source_pairs = pd.read_csv(inputs["pair_cache"])
    candidate = source_pairs[
        source_pairs.dataset.isin(targets) & (source_pairs.component_split == "test")
    ].copy()
    selected_rows = []
    for (dataset, component_id), frame in candidate.groupby(["dataset", "component_id"], sort=True):
        center = float(frame.potency_difference.median())
        ranked = frame.assign(
            distance_to_component_median=(frame.potency_difference - center).abs()
        ).sort_values(
            ["distance_to_component_median", "left_index", "right_index", "pair_index"],
            kind="mergesort",
        )
        row = ranked.iloc[0].copy()
        row["component_pair_count"] = len(frame)
        row["component_median_potency_difference"] = center
        selected_rows.append(row)
    selected_all_source_components = pd.DataFrame(selected_rows).sort_values(["dataset", "component_id"]).reset_index(drop=True)

    original = pd.read_csv(inputs["original_sentinel_pairs"])
    original = original[original.dataset.isin(targets) & (original.component_split == "test")][
        ["dataset", "component_id", "pair_index", "left_index", "right_index", "potency_difference"]
    ].rename(columns={
        "pair_index": "original_pair_index",
        "left_index": "original_left_index",
        "right_index": "original_right_index",
        "potency_difference": "original_potency_difference",
    })
    frozen_components = original[["dataset", "component_id"]]
    outside = selected_all_source_components.merge(
        frozen_components, on=["dataset", "component_id"], how="left", indicator=True
    )
    outside = outside[outside._merge == "left_only"].drop(columns="_merge")
    source_sentinel_failures = source_pairs[
        source_pairs.dataset.isin(targets)
        & (source_pairs.component_split == "test")
        & source_pairs.is_sentinel.astype(bool)
    ][["dataset", "component_id", "pair_index", "rationale_failure_reason"]].rename(columns={
        "pair_index": "original_pair_index",
        "rationale_failure_reason": "original_rationale_failure_reason",
    })
    outside = outside.merge(source_sentinel_failures, on=["dataset", "component_id"], how="left", validate="one_to_one")
    outside["exclusion_reason"] = "outside frozen original MCS-valid sentinel population"
    selected = selected_all_source_components.merge(original, on=["dataset", "component_id"], validate="one_to_one")
    selected["same_as_original"] = selected.pair_index == selected.original_pair_index

    rationale_cfg = json.loads(rationale_config.read_text(encoding="utf-8"))["mcs"]
    raw_dir = ROOT / "external/MoleculeACE/MoleculeACE/Data/benchmark_data"
    RDLogger.DisableLog("rdApp.warning")
    proxy_rows = []
    atom_counts = {}
    for target in targets:
        raw = pd.read_csv(raw_dir / f"{target}.csv")
        mols = [Chem.MolFromSmiles(value) for value in raw.smiles]
        target_selected = selected[selected.dataset == target]
        for row in target_selected.itertuples(index=False):
            left = mols[int(row.left_index)]
            right = mols[int(row.right_index)]
            proxy, reason = rationale(left, right, rationale_cfg)
            positive_atoms = 0 if proxy is None else len(proxy["rationale_a"]) + len(proxy["rationale_b"])
            total_atoms = left.GetNumAtoms() + right.GetNumAtoms()
            proxy_rows.append({
                "dataset": target,
                "component_id": int(row.component_id),
                "mcs_atoms": np.nan if proxy is None else proxy["mcs_atoms"],
                "match": "" if proxy is None else json.dumps(proxy["match"]),
                "rationale_left": "" if proxy is None else json.dumps(proxy["rationale_a"]),
                "rationale_right": "" if proxy is None else json.dumps(proxy["rationale_b"]),
                "rationale_valid": proxy is not None,
                "rationale_failure_reason": reason,
                "positive_atoms": positive_atoms,
                "total_atoms": total_atoms,
                "valid_normalized_ap_prevalence": 0 < positive_atoms < total_atoms,
            })
            atom_counts[f"{target}:{int(row.left_index)}"] = left.GetNumAtoms()
            atom_counts[f"{target}:{int(row.right_index)}"] = right.GetNumAtoms()
    proxies = pd.DataFrame(proxy_rows)
    selected = selected.drop(columns=[
        "mcs_atoms", "match", "rationale_left", "rationale_right", "rationale_valid", "rationale_failure_reason"
    ]).merge(proxies, on=["dataset", "component_id"], validate="one_to_one")
    selected["is_sentinel"] = True

    molecule_splits = pd.read_csv(inputs["molecule_splits"])
    split_lookup = molecule_splits.set_index(["dataset", "molecule_index"])
    for row in selected.itertuples(index=False):
        for index in [int(row.left_index), int(row.right_index)]:
            split = split_lookup.loc[(row.dataset, index)]
            if int(split.component_id) != int(row.component_id) or split.component_split != "test":
                raise ValueError(f"selected endpoint split mismatch: {row.dataset}:{index}")

    molecules = pd.read_csv(inputs["source_molecules"])
    model_molecules = pd.read_csv(inputs["model_molecules"])
    eligible = selected[selected.valid_normalized_ap_prevalence].copy()
    evaluation_ids = {
        f"{row.dataset}:{int(index)}"
        for row in eligible.itertuples(index=False)
        for index in [row.left_index, row.right_index]
    }
    evaluation = model_molecules[model_molecules.row_id.astype(str).isin(evaluation_ids)].copy()
    if len(evaluation) != len(evaluation_ids):
        raise ValueError("not every selected endpoint is in the frozen shared model population")

    library = {
        line.strip().split()[0]
        for line in inputs["replacement_library"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    heldout = set(molecules[molecules.component_split.isin(["calibration", "test"])].canonical_smiles.astype(str))
    selected_smiles = set(evaluation.canonical_smiles.astype(str))
    leakage_all = library & heldout
    leakage_selected = library & selected_smiles

    for arm in cfg["representations"]:
        with np.load(inputs["original_representations"] / f"{arm}.npz", allow_pickle=False) as data:
            ids = set(data["row_id"].astype(str))
        if not evaluation_ids <= ids:
            raise ValueError(f"missing frozen parent representations for {arm}")

    models = pd.read_csv(inputs["model_heads"] / "model_manifest.csv")
    models = models[
        models.dataset.isin(targets)
        & (models.seed == spec["model_seed"])
        & models.representation.isin(cfg["representations"])
        & models.predictor.isin(spec["predictors"])
        & models.variant.isin(spec["variants"])
    ].copy()
    model_key = ["dataset", "representation", "predictor", "seed", "variant"]
    expected_models = len(targets) * len(cfg["representations"]) * len(spec["predictors"]) * len(spec["variants"])
    if len(models) != expected_models or models.duplicated(model_key).any():
        raise ValueError("incomplete or non-unique seed-42 head crossing")
    for row in models.itertuples(index=False):
        path = resolve(row.path)
        require_hash(path, row.sha256)

    base_columns = list(source_pairs.columns)
    selected.to_csv(prepared / "selected_pairs_all.csv", index=False)
    outside.to_csv(prepared / "source_components_outside_frozen_analysis.csv", index=False)
    eligible[base_columns].to_csv(prepared / "sentinel_pairs.csv", index=False)
    evaluation.to_csv(prepared / "evaluation_molecules.csv", index=False)
    shutil.copy2(inputs["source_molecules"], prepared / "molecules.csv")
    shutil.copy2(inputs["model_molecules"], prepared / "model_molecules.csv")

    target_rows = selected.groupby("dataset", as_index=False).agg(
        selected_components=("component_id", "size"),
        localization_eligible_components=("valid_normalized_ap_prevalence", "sum"),
        unchanged_from_original=("same_as_original", "sum"),
        median_selected_potency_difference=("potency_difference", "median"),
        median_original_potency_difference=("original_potency_difference", "median"),
    )
    target_rows.to_csv(prepared / "selection_by_target.csv", index=False)
    observed_counts = {
        "targets": len(targets),
        "source_test_components": len(selected_all_source_components),
        "selected_components": len(selected),
        "source_components_outside_frozen_analysis": len(outside),
        "localization_eligible_components": len(eligible),
        "changed_from_original": int((~selected.same_as_original).sum()),
        "unchanged_from_original": int(selected.same_as_original.sum()),
        "evaluation_endpoints": len(evaluation),
        "seed42_heads": len(models),
    }
    expected_counts = expected_preparation_counts(spec)
    if set(expected_counts) != set(observed_counts):
        raise ValueError("preparation expected-count keys do not match the validation contract")
    count_checks = {key: observed_counts[key] == int(value) for key, value in expected_counts.items()}
    status = "pass" if (
        all(count_checks.values())
        and selected[["dataset", "component_id"]].duplicated().sum() == 0
        and not leakage_all
        and not leakage_selected
        and len(evaluation_ids) == len(evaluation)
        and len(models) == expected_models
    ) else "fail"
    outputs = [
        "selected_pairs_all.csv", "source_components_outside_frozen_analysis.csv",
        "sentinel_pairs.csv", "evaluation_molecules.csv",
        "molecules.csv", "model_molecules.csv", "selection_by_target.csv",
    ]
    audit = {
        "run_id": cfg["run_id"],
        "status": status,
        "selected_components": int(len(selected)),
        "source_test_components": int(len(selected_all_source_components)),
        "source_components_outside_frozen_analysis": int(len(outside)),
        "localization_eligible_components": int(len(eligible)),
        "changed_from_original": int((~selected.same_as_original).sum()),
        "unchanged_from_original": int(selected.same_as_original.sum()),
        "evaluation_endpoints": int(len(evaluation)),
        "replacement_library_smiles": int(len(library)),
        "all_heldout_smiles": int(len(heldout)),
        "library_all_heldout_overlap": int(len(leakage_all)),
        "library_selected_endpoint_overlap": int(len(leakage_selected)),
        "seed42_heads": int(len(models)),
        "expected_seed42_heads": int(expected_models),
        "count_validation": {
            "expected": {key: int(value) for key, value in expected_counts.items()},
            "observed": {key: int(value) for key, value in observed_counts.items()},
            "checks": count_checks,
        },
        "input_hashes": {name: sha256(path) for name, path in checks.items()},
        "output_hashes": {name: sha256(prepared / name) for name in outputs},
    }
    (prepared / "feasibility.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
