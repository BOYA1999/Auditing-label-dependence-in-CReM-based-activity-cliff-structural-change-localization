import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr

from score_factorial_explanations import (
    add_atom_rows,
    add_pair_rows,
    build_lime_neighborhoods,
    lime_arrays,
    load_npz,
    load_predictor,
    mean_replacement_arrays,
    predict,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_metrics(pairs, target, prediction_map, observed_map):
    rows = []
    for pair in pairs[pairs.dataset == target].itertuples(index=False):
        true_delta = float(observed_map[pair.left_index] - observed_map[pair.right_index])
        predicted_delta = float(prediction_map[pair.left_index] - prediction_map[pair.right_index])
        error = abs(predicted_delta - true_delta)
        rows.append({
            "component_id": int(pair.component_id), "component_split": pair.component_split,
            "true_delta": true_delta, "predicted_delta": predicted_delta,
            "absolute_signed_delta_error": error, "gap_score": -error,
        })
    return pd.DataFrame(rows)


def rank_row(a, b, support):
    def correlation(x, y):
        return spearmanr(x, y).statistic if len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0 else np.nan

    all_value = correlation(a, b)
    supported_value = correlation(a[support], b[support])
    return (
        float(all_value) if np.isfinite(all_value) else np.nan,
        float(supported_value) if np.isfinite(supported_value) else np.nan,
        int(support.sum()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--mutant-representations", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exclude-unimol-fallback-mutants", action="store_true")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    molecules = pd.read_csv(args.prepared / "model_molecules.csv")
    evaluation = pd.read_csv(args.prepared / "evaluation_molecules.csv")
    pairs = pd.read_csv(args.prepared / "sentinel_pairs.csv")
    scored = pd.read_csv(args.counterfactuals / "scored_mutants_model.csv")
    unique = pd.read_csv(args.counterfactuals / "unique_scored_mutants_model.csv")
    models = pd.read_csv(args.heads / "model_manifest.csv")
    original_ids = molecules.row_id.astype(str).to_numpy()
    mutant_ids = unique.row_id.astype(str).to_numpy()
    original_position = {row_id: index for index, row_id in enumerate(original_ids)}
    atom_counts = {
        row.row_id: Chem.MolFromSmiles(row.canonical_smiles).GetNumAtoms()
        for row in evaluation.itertuples(index=False)
    }
    rationale_rows = []
    for pair in pairs.itertuples(index=False):
        left_id = f"{pair.dataset}:{int(pair.left_index)}"
        right_id = f"{pair.dataset}:{int(pair.right_index)}"
        positives = len(json.loads(pair.rationale_left)) + len(json.loads(pair.rationale_right))
        total = atom_counts[left_id] + atom_counts[right_id]
        rationale_rows.append({
            "dataset": pair.dataset, "component_id": int(pair.component_id),
            "component_split": pair.component_split, "left_index": int(pair.left_index),
            "right_index": int(pair.right_index), "positive_atoms": positives,
            "total_atoms": total, "valid_normalized_ap_prevalence": 0 < positives < total,
        })
    rationale_audit = pd.DataFrame(rationale_rows)
    rationale_audit.to_csv(args.output / "rationale_prevalence_audit.csv", index=False)
    excluded_rationales = rationale_audit[~rationale_audit.valid_normalized_ap_prevalence]
    excluded_rationales.to_csv(args.output / "excluded_rationale_pairs.csv", index=False)
    pairs = pairs[rationale_audit.valid_normalized_ap_prevalence.to_numpy()].copy()
    mutant_position = {smiles: index for index, smiles in enumerate(unique.canonical_smiles)}
    scored["mutant_position"] = scored.mutant_smiles.map(mutant_position)
    if scored.mutant_position.isna().any():
        raise ValueError("counterfactual representation mapping mismatch")
    scored["mutant_position"] = scored.mutant_position.astype(int)
    original_morgan = load_npz(args.representations / "morgan.npz", original_ids)
    mutant_morgan = load_npz(args.mutant_representations / "morgan.npz", mutant_ids)
    neighborhoods = build_lime_neighborhoods(
        evaluation, scored, original_morgan, mutant_morgan, original_position, cfg["crem_lime"],
    )
    scored_without_unimol_fallback = scored
    neighborhoods_without_unimol_fallback = neighborhoods
    excluded_unimol_fallback_rows = 0
    if args.exclude_unimol_fallback_mutants:
        representation_manifest = json.loads((args.mutant_representations / "manifest.json").read_text(encoding="utf-8"))
        failure_indices = representation_manifest["arms"]["unimol"]["conformer_failure_indices"]
        fallback_smiles = set(unique.iloc[failure_indices].canonical_smiles)
        scored_without_unimol_fallback = scored[~scored.mutant_smiles.isin(fallback_smiles)].copy()
        excluded_unimol_fallback_rows = len(scored) - len(scored_without_unimol_fallback)
        neighborhoods_without_unimol_fallback = build_lime_neighborhoods(
            evaluation, scored_without_unimol_fallback, original_morgan, mutant_morgan,
            original_position, cfg["crem_lime"],
        )
    selected_targets = args.targets or cfg["targets"]
    if not set(selected_targets) <= set(cfg["targets"]):
        raise ValueError("requested target is absent from config")
    targets = selected_targets[:1] if args.smoke else selected_targets
    representations = list(cfg["representations"])[:1] if args.smoke else list(cfg["representations"])
    predictors = ["xgboost"] if args.smoke else ["xgboost", "mlp"]
    seeds = [42] if args.smoke else (args.seeds or cfg["seeds"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pair_rows = []
    fidelity_rows = []
    similarity_rows = []
    output_audit_rows = []
    atom_path = args.output / "atom_scores.csv.gz"
    atom_fields = [
        "dataset", "representation", "predictor", "explainer", "seed", "variant",
        "row_id", "component_split", "atom_index", "atom_score", "supported",
    ]
    with gzip.open(atom_path, "wt", encoding="utf-8", newline="") as atom_handle:
        atom_writer = csv.DictWriter(atom_handle, fieldnames=atom_fields)
        atom_writer.writeheader()
        for representation in representations:
            original_values = load_npz(args.representations / f"{representation}.npz", original_ids)
            mutant_values = load_npz(args.mutant_representations / f"{representation}.npz", mutant_ids)
            representation_scored = scored_without_unimol_fallback if representation == "unimol" else scored
            representation_neighborhoods = neighborhoods_without_unimol_fallback if representation == "unimol" else neighborhoods
            for target in targets:
                target_eval = evaluation[evaluation.dataset == target]
                target_scored = representation_scored[representation_scored.dataset == target]
                target_mutant_positions = np.sort(target_scored.mutant_position.unique())
                parent_positions = [original_position[row_id] for row_id in target_eval.row_id]
                observed_map = molecules[molecules.dataset == target].set_index("molecule_index").y.to_dict()
                for predictor in predictors:
                    entries = models[
                        (models.dataset == target) & (models.representation == representation)
                        & (models.predictor == predictor) & models.seed.isin(seeds)
                    ]
                    if len(entries) != len(seeds) * 2:
                        raise ValueError(f"incomplete model crossing: {target} {representation} {predictor}")
                    score_cache = {}
                    for entry in entries.itertuples(index=False):
                        loaded = load_predictor(predictor, entry, device)
                        mutant_predictions = np.full(len(unique), np.nan, dtype=float)
                        mutant_predictions[target_mutant_positions] = predict(
                            predictor, loaded, mutant_values[target_mutant_positions], device,
                        )
                        parent_values = predict(predictor, loaded, original_values[parent_positions], device)
                        parent_predictions = dict(zip(target_eval.row_id, parent_values))
                        prediction_map = {
                            int(row.molecule_index): float(parent_predictions[row.row_id])
                            for row in target_eval.itertuples(index=False)
                        }
                        metrics = pair_metrics(pairs, target, prediction_map, observed_map)
                        local_mutant_predictions = mutant_predictions[target_mutant_positions]
                        combined_predictions = np.concatenate([parent_values, local_mutant_predictions])
                        output_audit_rows.append({
                            "dataset": target, "representation": representation,
                            "predictor": predictor, "seed": int(entry.seed), "variant": entry.variant,
                            "parent_prediction_range": float(np.ptp(parent_values)),
                            "mutant_prediction_range": float(np.ptp(local_mutant_predictions)),
                            "local_domain_prediction_range": float(np.ptp(combined_predictions)),
                            "finite_local_domain": bool(np.isfinite(combined_predictions).all()),
                            "parent_rows": len(parent_values), "mutant_rows": len(local_mutant_predictions),
                        })
                        mean_scores, mean_support = mean_replacement_arrays(
                            target_eval, target_scored, mutant_predictions, parent_predictions,
                            cfg["mask_size_exponent"],
                        )
                        lime_scores, lime_support, fidelity = lime_arrays(
                            target_eval, representation_neighborhoods, original_morgan, mutant_morgan,
                            original_position, mutant_predictions, parent_predictions, cfg["crem_lime"],
                        )
                        for explainer, arrays, supports in [
                            ("crem_mean", mean_scores, mean_support),
                            ("crem_lime", lime_scores, lime_support),
                        ]:
                            metadata = {
                                "dataset": target, "representation": representation,
                                "predictor": predictor, "explainer": explainer,
                                "seed": int(entry.seed), "variant": entry.variant,
                            }
                            add_pair_rows(pair_rows, pairs, arrays, supports, metadata, metrics)
                            atom_buffer = []
                            add_atom_rows(atom_buffer, target_eval, arrays, supports, metadata)
                            atom_writer.writerows(atom_buffer)
                            score_cache[(entry.variant, int(entry.seed), explainer)] = (arrays, supports)
                        for row in fidelity:
                            row.update({
                                "representation": representation, "predictor": predictor,
                                "seed": int(entry.seed), "variant": entry.variant,
                            })
                            fidelity_rows.append(row)
                    test_eval = target_eval[target_eval.component_split == "test"]
                    for explainer in cfg["explainers"]:
                        random_maps = [score_cache[("label_permuted", seed, explainer)][0] for seed in seeds]
                        random_supports = [score_cache[("label_permuted", seed, explainer)][1] for seed in seeds]
                        for trained_seed in seeds:
                            trained_map, trained_support = score_cache[("trained", trained_seed, explainer)]
                            for random_seed in seeds + ["mean"]:
                                for record in test_eval.itertuples(index=False):
                                    if random_seed == "mean":
                                        random_score = np.mean([value[record.row_id] for value in random_maps], axis=0)
                                        random_support = np.any([value[record.row_id] for value in random_supports], axis=0)
                                    else:
                                        random_score = score_cache[("label_permuted", random_seed, explainer)][0][record.row_id]
                                        random_support = score_cache[("label_permuted", random_seed, explainer)][1][record.row_id]
                                    support = trained_support[record.row_id] | random_support
                                    all_atoms, supported_atoms, supported_count = rank_row(
                                        trained_map[record.row_id], random_score, support,
                                    )
                                    similarity_rows.append({
                                        "row_id": record.row_id, "dataset": target,
                                        "representation": representation, "predictor": predictor,
                                        "explainer": explainer, "trained_seed": trained_seed,
                                        "random_seed": random_seed, "comparison": "random_mean" if random_seed == "mean" else "replicate",
                                        "matched_seed": random_seed == trained_seed,
                                        "spearman_supported": supported_atoms,
                                        "spearman_all_atoms": all_atoms,
                                        "supported_atoms": supported_count,
                                    })
                    print(f"{target} {representation} {predictor}", flush=True)

    pair_frame = pd.DataFrame(pair_rows)
    fidelity_frame = pd.DataFrame(fidelity_rows)
    similarity_frame = pd.DataFrame(similarity_rows)
    output_audit = pd.DataFrame(output_audit_rows)
    pair_frame.to_csv(args.output / "pair_explanations.csv", index=False)
    fidelity_frame.to_csv(args.output / "local_fidelity.csv", index=False)
    similarity_frame.to_csv(args.output / "atom_rank_similarity.csv", index=False)
    output_audit.to_csv(args.output / "model_output_audit.csv", index=False)
    expected_pairs = len(pairs[pairs.dataset.isin(targets)]) * len(representations) * len(predictors) * 2 * len(seeds) * 2
    randomized = output_audit[output_audit.variant == "label_permuted"]
    randomized_local_nonconstant = randomized.local_domain_prediction_range > 1e-8
    randomized_gate = bool(randomized.finite_local_domain.all() and randomized_local_nonconstant.all())
    summary = {
        "run_id": cfg["run_id"], "smoke": args.smoke,
        "status": "pass" if len(pair_frame) == expected_pairs and np.isfinite(pair_frame.normalized_ap_lift).all() and randomized_gate else "fail",
        "pair_rows": int(len(pair_frame)), "expected_pair_rows": int(expected_pairs),
        "atom_rows": sum(1 for _ in gzip.open(atom_path, "rt", encoding="utf-8")) - 1,
        "fidelity_rows": int(len(fidelity_frame)), "rank_similarity_rows": int(len(similarity_frame)),
        "finite_pair_metrics": bool(np.isfinite(pair_frame.normalized_ap_lift).all()),
        "localization_pairs": int(len(pairs)),
        "excluded_rationale_pairs": int(len(excluded_rationales)),
        "exclude_unimol_fallback_mutants": args.exclude_unimol_fallback_mutants,
        "excluded_unimol_fallback_scored_rows": excluded_unimol_fallback_rows,
        "nonconstant_randomized_local_models": int(randomized_local_nonconstant.sum()),
        "randomized_local_models": int(len(randomized)),
        "randomized_local_nonconstant_gate": randomized_gate,
        "input_hashes": {
            "config": sha256(args.config), "model_manifest": sha256(args.heads / "model_manifest.csv"),
            "scored_mutants": sha256(args.counterfactuals / "scored_mutants_model.csv"),
            "original_representation_manifest": sha256(args.representations / "manifest.json"),
            "mutant_representation_manifest": sha256(args.mutant_representations / "manifest.json"),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if summary["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
