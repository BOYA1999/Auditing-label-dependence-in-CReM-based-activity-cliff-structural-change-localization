import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import scipy
import sklearn
from rdkit import Chem
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

from score_factorial_explanations import (
    build_lime_neighborhoods,
    load_npz,
    load_predictor,
    predict,
    weighted_r2,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def stable_seed(seed, label):
    offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")
    return (int(seed) + offset) % (2**32)


def parse_atoms(value):
    result = json.loads(value)
    if not isinstance(result, list) or any(not isinstance(atom, int) for atom in result):
        raise ValueError(f"invalid atom list: {value}")
    return result


def nap(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    prevalence = float(labels.mean())
    if not 0.0 < prevalence < 1.0:
        return np.nan, prevalence, np.nan
    ap = float(average_precision_score(labels, np.asarray(scores, dtype=float)))
    return ap, prevalence, float((ap - prevalence) / (1.0 - prevalence))


def interval(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def two_stage_bootstrap(frame, value, draws, seed, label):
    targets = {
        target: [part[value].to_numpy(float) for _, part in target_frame.groupby("component_id", sort=True)]
        for target, target_frame in frame.groupby("dataset", sort=True)
    }
    names = sorted(targets)
    rng = np.random.default_rng(stable_seed(seed, label))
    distribution = np.empty(draws, dtype=float)
    for draw in range(draws):
        estimates = []
        for target_index in rng.integers(0, len(names), len(names)):
            components = targets[names[target_index]]
            sampled = [components[index] for index in rng.integers(0, len(components), len(components))]
            estimates.append(np.median(np.concatenate(sampled)))
        distribution[draw] = np.median(estimates)
    return distribution


def compact_representation(path, expected_ids, positions):
    values = load_npz(path, expected_ids)
    selected = values[np.asarray(positions, dtype=int)].copy()
    del values
    return selected


def reference_predictions(path, targets, endpoints):
    columns = ["row_id", "dataset", "component_split", "representation", "predictor", "seed", "variant", "prediction"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=200000):
        keep = chunk.dataset.isin(targets) & chunk.row_id.isin(endpoints)
        if keep.any():
            chunks.append(chunk[keep])
    return pd.concat(chunks, ignore_index=True)


def exact_selected_mutants(raw, threshold, count):
    selected = raw[raw.parent_mutant_tanimoto >= threshold].copy()
    selected["selection_hash"] = [
        hashlib.sha256(f"{row.row_id}:{int(row.mask_id)}:{row.mutant_smiles}".encode()).hexdigest()
        for row in selected.itertuples(index=False)
    ]
    return selected.sort_values(["row_id", "mask_id", "selection_hash"]).groupby(["row_id", "mask_id"], sort=False).head(count)


def weighted_ridge_diagnostics(matrix, outcomes, weights, alpha, folds, cv_seed):
    samples = len(matrix)
    variables = matrix.shape[1]
    if variables == 0:
        intercept = np.average(outcomes, axis=0, weights=weights)
        fitted = np.repeat(intercept[None, :], samples, axis=0)
        coefficients = np.empty((outcomes.shape[1], 0), dtype=float)
        in_sample = np.array([weighted_r2(outcomes[:, index], fitted[:, index], weights) for index in range(outcomes.shape[1])])
        press_predictions = np.empty_like(outcomes)
        if samples > 1:
            for index in range(samples):
                retained = np.arange(samples) != index
                press_predictions[index] = np.average(outcomes[retained], axis=0, weights=weights[retained])
        else:
            press_predictions[:] = np.nan
        status = "intercept_only"
        effective_df = 1.0
        maximum_leverage = float(weights.max() / weights.sum())
        hat_mismatch = 0.0
        fitted_difference = np.zeros(outcomes.shape[1], dtype=float)
        coefficient_max_difference = np.zeros(outcomes.shape[1], dtype=float)
        coefficient_relative_l2_difference = np.zeros(outcomes.shape[1], dtype=float)
        exact_coefficients = coefficients.copy()
    else:
        fitted = np.empty_like(outcomes)
        coefficients = np.empty((outcomes.shape[1], variables), dtype=float)
        for outcome_index in range(outcomes.shape[1]):
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(matrix, outcomes[:, outcome_index], sample_weight=weights)
            fitted[:, outcome_index] = model.predict(matrix)
            coefficients[outcome_index] = model.coef_
        in_sample = np.array([weighted_r2(outcomes[:, index], fitted[:, index], weights) for index in range(outcomes.shape[1])])
        sqrt_weights = np.sqrt(weights)
        weight_sum = float(weights.sum())
        mean_x = np.average(matrix, axis=0, weights=weights)
        centered = matrix - mean_x
        weighted_centered = sqrt_weights[:, None] * centered
        kernel = weighted_centered @ weighted_centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        shrinkage = eigenvalues / (eigenvalues + alpha)
        centered_hat = (eigenvectors * shrinkage) @ eigenvectors.T
        intercept_vector = sqrt_weights / np.sqrt(weight_sum)
        hat = np.outer(intercept_vector, intercept_vector) + centered_hat
        weighted_outcomes = sqrt_weights[:, None] * outcomes
        hat_fitted = (hat @ weighted_outcomes) / sqrt_weights[:, None]
        hat_mismatch = float(np.max(np.abs(fitted - hat_fitted)))
        outcome_means = np.average(outcomes, axis=0, weights=weights)
        weighted_centered_outcomes = sqrt_weights[:, None] * (outcomes - outcome_means)
        dual = eigenvectors @ (
            (eigenvectors.T @ weighted_centered_outcomes) / (eigenvalues[:, None] + alpha)
        )
        exact_coefficients = (weighted_centered.T @ dual).T
        exact_fitted = outcome_means + centered @ exact_coefficients.T
        fitted_difference = np.max(np.abs(fitted - exact_fitted), axis=0)
        coefficient_difference = coefficients - exact_coefficients
        coefficient_max_difference = np.max(np.abs(coefficient_difference), axis=1)
        exact_norm = np.linalg.norm(exact_coefficients, axis=1)
        coefficient_relative_l2_difference = np.divide(
            np.linalg.norm(coefficient_difference, axis=1), exact_norm,
            out=np.full_like(exact_norm, np.nan), where=exact_norm > 1e-12,
        )
        leverage = np.diag(hat)
        residual_weighted = sqrt_weights[:, None] * (outcomes - hat_fitted)
        safe = 1.0 - leverage
        press_predictions = outcomes - residual_weighted / (sqrt_weights[:, None] * safe[:, None]) if np.all(safe > 1e-10) else np.full_like(outcomes, np.nan)
        effective_df = float(np.trace(hat))
        maximum_leverage = float(leverage.max())
        status = "ridge"
    means = np.average(outcomes, axis=0, weights=weights)
    denominator = np.sum(weights[:, None] * np.square(outcomes - means), axis=0)
    press_error = np.sum(weights[:, None] * np.square(outcomes - press_predictions), axis=0)
    press_r2 = np.divide(denominator - press_error, denominator, out=np.full_like(denominator, np.nan), where=denominator > 1e-12)
    cv_predictions = np.full_like(outcomes, np.nan)
    cv_status = "available"
    if samples >= folds:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=int(cv_seed))
        for train, test in splitter.split(matrix):
            if variables:
                model = Ridge(alpha=alpha, solver="lsqr")
                model.fit(matrix[train], outcomes[train], sample_weight=weights[train])
                cv_predictions[test] = model.predict(matrix[test])
            else:
                cv_predictions[test] = np.average(outcomes[train], axis=0, weights=weights[train])
    else:
        cv_status = "fewer_than_five_samples"
    cv_error = np.sum(weights[:, None] * np.square(outcomes - cv_predictions), axis=0)
    cv_r2 = np.divide(denominator - cv_error, denominator, out=np.full_like(denominator, np.nan), where=(denominator > 1e-12) & np.isfinite(cv_error))
    return {
        "fitted": fitted,
        "coefficients": coefficients,
        "in_sample_r2": in_sample,
        "press_r2": press_r2,
        "cv_r2": cv_r2,
        "effective_df": effective_df,
        "maximum_leverage": maximum_leverage,
        "hat_fit_max_abs_difference": hat_mismatch,
        "lsqr_vs_svd_fitted_max_abs_difference": fitted_difference,
        "lsqr_vs_svd_coefficient_max_abs_difference": coefficient_max_difference,
        "lsqr_vs_svd_coefficient_relative_l2_difference": coefficient_relative_l2_difference,
        "svd_equivalent_coefficients": exact_coefficients,
        "status": status,
        "cv_status": cv_status,
    }


def mean_response_arrays(evaluation, scored, mutant_matrix, parent_matrix, exponents):
    model_count = parent_matrix.shape[1]
    grouped = {row_id: frame for row_id, frame in scored.groupby("row_id", sort=False)}
    absolute = {exponent: {} for exponent in exponents}
    signed = {exponent: {} for exponent in exponents}
    supports = {}
    for row_index, record in enumerate(evaluation.itertuples(index=False)):
        atoms = Chem.MolFromSmiles(record.canonical_smiles).GetNumAtoms()
        support_count = np.zeros(atoms, dtype=int)
        absolute_sums = {exponent: np.zeros((model_count, atoms), dtype=float) for exponent in exponents}
        signed_sums = {exponent: np.zeros((model_count, atoms), dtype=float) for exponent in exponents}
        frame = grouped.get(record.row_id, pd.DataFrame())
        if not frame.empty:
            for (_, mask_atoms, mask_size), mask in frame.groupby(["mask_id", "mask_atoms", "mask_size"], sort=False):
                atom_indices = parse_atoms(mask_atoms)
                positions = mask.mutant_position.to_numpy(int)
                effects = mutant_matrix[positions] - parent_matrix[row_index]
                absolute_effect = np.mean(np.abs(effects), axis=0)
                signed_effect = np.mean(effects, axis=0)
                for atom in atom_indices:
                    support_count[atom] += 1
                    for exponent in exponents:
                        scale = float(mask_size) ** float(exponent)
                        absolute_sums[exponent][:, atom] += absolute_effect / scale
                        signed_sums[exponent][:, atom] += signed_effect / scale
        for exponent in exponents:
            absolute[exponent][record.row_id] = np.divide(
                absolute_sums[exponent], support_count[None, :],
                out=np.zeros_like(absolute_sums[exponent]), where=support_count[None, :] > 0,
            )
            signed[exponent][record.row_id] = np.divide(
                signed_sums[exponent], support_count[None, :],
                out=np.zeros_like(signed_sums[exponent]), where=support_count[None, :] > 0,
            )
        supports[record.row_id] = support_count > 0
    return absolute, signed, supports


def lime_arrays_and_diagnostics(evaluation, neighborhoods, original_morgan, mutant_morgan, original_position, mutant_matrix, parent_matrix, model_metadata, ridge_cfg, scenario, representation):
    absolute, signed_loss, supports, rows = {}, {}, {}, []
    alpha = float(ridge_cfg["alpha"])
    for row_index, record in enumerate(evaluation.itertuples(index=False)):
        neighborhood = neighborhoods[record.row_id]
        positions = neighborhood["mutant_positions"]
        bits = neighborhood["variable_bits"]
        matrix = np.vstack([original_morgan[original_position[record.row_id]], mutant_morgan[positions]])[:, bits]
        outcomes = np.vstack([parent_matrix[row_index], mutant_matrix[positions]])
        weights = neighborhood["weights"]
        diagnostics = weighted_ridge_diagnostics(
            matrix, outcomes, weights, alpha, int(ridge_cfg["cv_folds"]),
            stable_seed(ridge_cfg["cv_seed"], f"{scenario}:{record.row_id}"),
        )
        coefficients = diagnostics["coefficients"]
        atom_absolute = np.zeros((len(model_metadata), len(neighborhood["atom_columns"])), dtype=float)
        atom_signed_loss = np.zeros_like(atom_absolute)
        atom_support = np.zeros(atom_absolute.shape[1], dtype=bool)
        for atom, columns in enumerate(neighborhood["atom_columns"]):
            if columns:
                atom_absolute[:, atom] = np.mean(np.abs(coefficients[:, columns]), axis=1)
                atom_signed_loss[:, atom] = -np.mean(coefficients[:, columns], axis=1)
                atom_support[atom] = True
        absolute[record.row_id] = atom_absolute
        signed_loss[record.row_id] = atom_signed_loss
        supports[record.row_id] = atom_support
        for model_index, metadata in enumerate(model_metadata):
            rows.append({
                **metadata, "scenario": scenario, "representation": representation,
                "row_id": record.row_id, "dataset": record.dataset, "component_split": record.component_split,
                "local_samples": len(outcomes), "variable_bits": len(bits),
                "ridge_effective_df": diagnostics["effective_df"],
                "maximum_leverage": diagnostics["maximum_leverage"],
                "in_sample_weighted_r2": diagnostics["in_sample_r2"][model_index],
                "press_weighted_r2": diagnostics["press_r2"][model_index],
                "five_fold_weighted_r2": diagnostics["cv_r2"][model_index],
                "hat_fit_max_abs_difference": diagnostics["hat_fit_max_abs_difference"],
                "lsqr_vs_svd_fitted_max_abs_difference": diagnostics["lsqr_vs_svd_fitted_max_abs_difference"][model_index],
                "lsqr_vs_svd_coefficient_max_abs_difference": diagnostics["lsqr_vs_svd_coefficient_max_abs_difference"][model_index],
                "lsqr_vs_svd_coefficient_relative_l2_difference": diagnostics["lsqr_vs_svd_coefficient_relative_l2_difference"][model_index],
                "fit_status": diagnostics["status"], "five_fold_status": diagnostics["cv_status"],
                "supported_atom_fraction": float(atom_support.mean()),
            })
    return absolute, signed_loss, supports, rows


def rationale_audit(pairs, molecules):
    atom_counts = {
        (row.dataset, int(row.molecule_index)): Chem.MolFromSmiles(row.canonical_smiles).GetNumAtoms()
        for row in molecules.itertuples(index=False)
    }
    rows = []
    labels = {}
    for pair in pairs.itertuples(index=False):
        left_atoms = atom_counts[(pair.dataset, int(pair.left_index))]
        right_atoms = atom_counts[(pair.dataset, int(pair.right_index))]
        match = json.loads(pair.match)
        matched_left = {int(left) for left, _ in match}
        matched_right = {int(right) for _, right in match}
        definitions = {
            "boundary_extended": (parse_atoms(pair.rationale_left), parse_atoms(pair.rationale_right)),
            "strict_unmatched": (
                sorted(set(range(left_atoms)) - matched_left),
                sorted(set(range(right_atoms)) - matched_right),
            ),
        }
        for name, (left_positive, right_positive) in definitions.items():
            left = np.isin(np.arange(left_atoms), left_positive)
            right = np.isin(np.arange(right_atoms), right_positive)
            combined = np.concatenate([left, right])
            valid = 0 < combined.sum() < len(combined)
            labels[(pair.dataset, int(pair.component_id), name)] = (left, right, valid)
            rows.append({
                "dataset": pair.dataset, "component_id": int(pair.component_id), "rationale": name,
                "left_atoms": left_atoms, "right_atoms": right_atoms, "positive_atoms": int(combined.sum()),
                "prevalence": float(combined.mean()), "valid_normalized_ap": bool(valid),
                "failure": "" if valid else ("no_positive_atoms" if combined.sum() == 0 else "all_atoms_positive"),
            })
    return pd.DataFrame(rows), labels


def add_pair_scores(rows, signed_rows, pairs, labels, arrays, signed_arrays, supports, metadata, parent_matrix, evaluation):
    row_position = {row_id: index for index, row_id in enumerate(evaluation.row_id)}
    outcome = dict(zip(evaluation.row_id, evaluation.y))
    for pair in pairs.itertuples(index=False):
        left_id = f"{pair.dataset}:{int(pair.left_index)}"
        right_id = f"{pair.dataset}:{int(pair.right_index)}"
        true_delta = float(outcome[left_id] - outcome[right_id])
        predicted_delta = parent_matrix[row_position[left_id]] - parent_matrix[row_position[right_id]]
        for rationale in ["boundary_extended", "strict_unmatched"]:
            left_labels, right_labels, valid = labels[(pair.dataset, int(pair.component_id), rationale)]
            if not valid:
                continue
            pair_labels = np.concatenate([left_labels, right_labels])
            for explainer, endpoint_arrays in arrays.items():
                pair_scores = np.concatenate([endpoint_arrays[left_id], endpoint_arrays[right_id]], axis=1)
                pair_support = np.concatenate([supports[explainer][left_id], supports[explainer][right_id]])
                for model_index, model in enumerate(metadata):
                    ap, prevalence, lift = nap(pair_labels, pair_scores[model_index])
                    rows.append({
                        **model, "scenario": model["scenario"], "explainer": explainer, "rationale": rationale,
                        "dataset": pair.dataset, "component_id": int(pair.component_id),
                        "left_index": int(pair.left_index), "right_index": int(pair.right_index),
                        "average_precision": ap, "prevalence": prevalence, "normalized_ap_lift": lift,
                        "supported_atom_fraction": float(pair_support.mean()), "true_delta": true_delta,
                        "predicted_delta": float(predicted_delta[model_index]),
                    })
            left_orientation = -np.sign(true_delta)
            right_orientation = np.sign(true_delta)
            for explainer, endpoint_arrays in signed_arrays.items():
                oriented = np.concatenate([
                    left_orientation * endpoint_arrays[left_id],
                    right_orientation * endpoint_arrays[right_id],
                ], axis=1)
                for model_index, model in enumerate(metadata):
                    score = oriented[model_index]
                    ap, prevalence, lift = nap(pair_labels, score)
                    signed_rows.append({
                        **model, "scenario": model["scenario"], "explainer": explainer, "rationale": rationale,
                        "dataset": pair.dataset, "component_id": int(pair.component_id),
                        "left_index": int(pair.left_index), "right_index": int(pair.right_index),
                        "average_precision": ap, "prevalence": prevalence, "normalized_ap_lift": lift,
                        "changed_mean_score": float(score[pair_labels].mean()),
                        "unchanged_mean_score": float(score[~pair_labels].mean()),
                        "changed_minus_unchanged_mean_score": float(score[pair_labels].mean() - score[~pair_labels].mean()),
                        "changed_positive_fraction": float((score[pair_labels] > 0).mean()),
                        "unchanged_positive_fraction": float((score[~pair_labels] > 0).mean()),
                        "true_delta": true_delta, "predicted_delta": float(predicted_delta[model_index]),
                    })


def summarize_scores(frame, draws, seed, label_prefix):
    keys = ["scenario", "rationale", "explainer", "dataset", "component_id", "representation", "predictor"]
    variant = frame.groupby(keys + ["variant"], as_index=False).agg(
        normalized_ap_lift=("normalized_ap_lift", "mean"), average_precision=("average_precision", "mean"),
        prevalence=("prevalence", "mean"), seeds=("seed", "nunique"),
    )
    trained = variant[variant.variant == "trained"].drop(columns=["variant", "seeds"]).rename(columns={
        "normalized_ap_lift": "trained_nap", "average_precision": "trained_ap", "prevalence": "trained_prevalence",
    })
    randomized = variant[variant.variant == "label_permuted"].drop(columns=["variant", "seeds"]).rename(columns={
        "normalized_ap_lift": "permuted_nap", "average_precision": "permuted_ap", "prevalence": "permuted_prevalence",
    })
    component = trained.merge(randomized, on=keys, validate="one_to_one")
    component["delta_nap"] = component.trained_nap - component.permuted_nap
    rows = []
    for key, group in component.groupby(["scenario", "rationale", "explainer"], sort=True):
        targets = group.groupby("dataset", as_index=False).agg(
            trained_nap=("trained_nap", "median"), permuted_nap=("permuted_nap", "median"),
            delta_nap=("delta_nap", "median"), components=("component_id", "nunique"),
        )
        distribution = two_stage_bootstrap(group, "delta_nap", draws, seed, label_prefix + ":" + ":".join(key))
        ci = interval(distribution)
        rows.append({
            "scenario": key[0], "rationale": key[1], "explainer": key[2],
            "target_balanced_median_trained_nap": float(targets.trained_nap.median()),
            "target_balanced_median_permuted_nap": float(targets.permuted_nap.median()),
            "target_balanced_median_delta_nap": float(targets.delta_nap.median()),
            "delta_ci95_low": ci[0], "delta_ci95_high": ci[1],
            "targets": group.dataset.nunique(), "components": group[["dataset", "component_id"]].drop_duplicates().shape[0],
        })
    return component, pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "research_v2/configs/major_revision_rescore_v1.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source_cfg = json.loads(resolve(cfg["source_config"]).read_text(encoding="utf-8"))
    paths = {name: resolve(path) for name, path in cfg["inputs"].items()}
    output = resolve(cfg["output"]) / "smoke" if args.smoke else resolve(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    targets = cfg["targets"][:1] if args.smoke else cfg["targets"]
    if not set(targets).issubset(source_cfg["confirmation_targets"]):
        raise ValueError("rescore targets must be confirmation targets")

    model_molecules = pd.read_csv(paths["model_molecules"])
    evaluation_all = pd.read_csv(paths["evaluation_molecules"])
    pairs = pd.read_csv(paths["sentinel_pairs"])
    pairs = pairs[pairs.dataset.isin(targets) & (pairs.component_split == "test")].copy()
    endpoint_ids = {
        f"{pair.dataset}:{int(index)}"
        for pair in pairs.itertuples(index=False)
        for index in (pair.left_index, pair.right_index)
    }
    evaluation = evaluation_all[evaluation_all.row_id.isin(endpoint_ids)].sort_values(["dataset", "molecule_index"]).copy()
    if set(evaluation.row_id) != endpoint_ids or len(evaluation) != 2 * len(pairs):
        raise ValueError("test sentinel endpoint mapping is not one-to-two")
    evaluation_scope = evaluation_all[evaluation_all.dataset.isin(targets)].sort_values(["dataset", "molecule_index"]).copy()
    scope_endpoint_ids = set(evaluation_scope.row_id)
    scored = pd.read_csv(paths["scored_mutants"])
    scored = scored[scored.row_id.isin(scope_endpoint_ids)].copy()
    scored_test = scored[scored.row_id.isin(endpoint_ids)].copy()
    raw = pd.read_csv(paths["raw_mutants"])
    raw = raw[raw.row_id.isin(endpoint_ids)].copy()
    unique = pd.read_csv(paths["unique_mutants"])
    mutant_global = {smiles: index for index, smiles in enumerate(unique.canonical_smiles)}
    scored["global_mutant_position"] = scored.mutant_smiles.map(mutant_global)
    if scored.global_mutant_position.isna().any():
        raise ValueError("selected mutant is absent from the representation index")
    selected_global_mutants = sorted(scored.global_mutant_position.astype(int).unique())
    compact_mutant = {position: index for index, position in enumerate(selected_global_mutants)}
    scored["mutant_position"] = scored.global_mutant_position.astype(int).map(compact_mutant)
    model_row_ids = model_molecules.row_id.astype(str).to_numpy()
    original_global = {row_id: index for index, row_id in enumerate(model_row_ids)}
    selected_global_originals = [original_global[row_id] for row_id in evaluation_scope.row_id]
    original_position = {row_id: index for index, row_id in enumerate(evaluation_scope.row_id)}
    unique_row_ids = unique.row_id.astype(str).to_numpy()

    rationale_frame, rationale_labels = rationale_audit(pairs, evaluation)
    rationale_frame.to_csv(output / "rationale_audit.csv", index=False)
    reference = reference_predictions(paths["reference_predictions"], targets, scope_endpoint_ids)
    models = pd.read_csv(paths["model_manifest"])
    models = models[models.dataset.isin(targets)].copy()
    expected_models = len(targets) * len(source_cfg["representations"]) * 2 * len(source_cfg["seeds"]) * 2
    if len(models) != expected_models:
        raise ValueError(f"incomplete saved-model crossing: {len(models)} != {expected_models}")

    availability_rows = []
    available_smiles = set(unique.canonical_smiles)
    for threshold in [0.5, 0.6, 0.7]:
        exact = exact_selected_mutants(raw, threshold, int(source_cfg["crem"]["pilot_scored_replacements_per_mask"]))
        missing = ~exact.mutant_smiles.isin(available_smiles)
        restricted = scored_test[scored_test.parent_mutant_tanimoto >= threshold]
        availability_rows.append({
            "threshold": threshold, "exact_reselection_candidate_rows": int((raw.parent_mutant_tanimoto >= threshold).sum()),
            "exact_reselection_selected_rows": len(exact), "exact_reselection_missing_representation_rows": int(missing.sum()),
            "exact_reselection_missing_unique_smiles": int(exact.loc[missing, "mutant_smiles"].nunique()),
            "exact_reselection_affected_masks": int(exact.loc[missing, ["row_id", "mask_id"]].drop_duplicates().shape[0]),
            "exact_reselection_affected_endpoints": int(exact.loc[missing, "row_id"].nunique()),
            "fixed_selected_restriction_rows": len(restricted),
            "used_as_rescore_scenario": threshold in {0.6, 0.7},
        })
    availability = pd.DataFrame(availability_rows)
    if availability.loc[availability.threshold == 0.6, "exact_reselection_missing_representation_rows"].iloc[0] != 0:
        raise ValueError("frozen 0.60 selected neighborhood is not representation-complete")
    availability.to_csv(output / "support_threshold_availability.csv", index=False)

    original_morgan = compact_representation(
        paths["original_representations"] / "morgan.npz", model_row_ids, selected_global_originals,
    )
    mutant_morgan = compact_representation(
        paths["mutant_representations"] / "morgan.npz", unique_row_ids, selected_global_mutants,
    )
    scenarios = {}
    coverage_rows = []
    for scenario in cfg["neighborhood_scenarios"]:
        subset = scored[scored.parent_mutant_tanimoto >= float(scenario["minimum_parent_tanimoto"])].copy()
        if scenario["absolute_domain_only"]:
            subset = subset[subset.absolute_in_domain.astype(bool)].copy()
        scenarios[scenario["name"]] = {
            "scored": subset,
            "neighborhoods": build_lime_neighborhoods(
                evaluation, subset, original_morgan, mutant_morgan, original_position, cfg["ridge_diagnostics"],
            ),
        }
        for target, target_eval in evaluation.groupby("dataset", sort=True):
            target_subset = subset[(subset.dataset == target) & subset.row_id.isin(target_eval.row_id)]
            samples = [1 + target_subset[target_subset.row_id == row_id].mutant_position.nunique() for row_id in target_eval.row_id]
            coverage_rows.append({
                "scenario": scenario["name"], "dataset": target, "endpoints": len(target_eval),
                "selected_mutant_rows": len(target_subset), "unique_selected_mutants": target_subset.mutant_position.nunique(),
                "endpoints_with_any_mutant": target_subset.row_id.nunique(), "median_local_samples": float(np.median(samples)),
                "minimum_local_samples": int(np.min(samples)), "endpoints_with_fewer_than_five_samples": int((np.asarray(samples) < 5).sum()),
            })
    pd.DataFrame(coverage_rows).to_csv(output / "neighborhood_coverage.csv", index=False)

    pair_rows, signed_rows, fidelity_rows, inference_audit = [], [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    representations = list(source_cfg["representations"])
    for representation in representations:
        if representation == "morgan":
            original_values, mutant_values = original_morgan, mutant_morgan
        else:
            original_values = compact_representation(
                paths["original_representations"] / f"{representation}.npz", model_row_ids, selected_global_originals,
            )
            mutant_values = compact_representation(
                paths["mutant_representations"] / f"{representation}.npz", unique_row_ids, selected_global_mutants,
            )
        for target in targets:
            target_eval = evaluation[evaluation.dataset == target].copy()
            target_eval_scope = evaluation_scope[evaluation_scope.dataset == target].copy()
            target_pairs = pairs[pairs.dataset == target]
            eval_positions = [original_position[row_id] for row_id in target_eval_scope.row_id]
            target_mutant_positions = np.sort(scored[scored.dataset == target].mutant_position.unique())
            target_models = models[(models.dataset == target) & (models.representation == representation)].sort_values(["predictor", "seed", "variant"])
            metadata, parent_columns, mutant_columns = [], [], []
            for entry in target_models.itertuples(index=False):
                loaded = load_predictor(entry.predictor, entry, device)
                parent_prediction_scope = predict(entry.predictor, loaded, original_values[eval_positions], device)
                mutant_prediction = np.full(len(selected_global_mutants), np.nan, dtype=float)
                mutant_prediction[target_mutant_positions] = predict(entry.predictor, loaded, mutant_values[target_mutant_positions], device)
                expected = reference[
                    (reference.dataset == target) & (reference.representation == representation)
                    & (reference.predictor == entry.predictor) & (reference.seed == entry.seed)
                    & (reference.variant == entry.variant)
                ].set_index("row_id").loc[target_eval_scope.row_id].prediction.to_numpy(float)
                maximum_difference = float(np.max(np.abs(parent_prediction_scope - expected)))
                if maximum_difference > float(cfg["reference_reproduction_tolerances"]["parent_prediction_absolute"]):
                    raise ValueError(f"saved-model re-inference mismatch: {target}/{representation}/{entry.predictor}/{entry.seed}/{entry.variant}")
                metadata.append({
                    "representation": representation, "predictor": entry.predictor,
                    "seed": int(entry.seed), "variant": entry.variant,
                    "scenario": "",
                })
                scope_position = {row_id: index for index, row_id in enumerate(target_eval_scope.row_id)}
                parent_columns.append(parent_prediction_scope[[scope_position[row_id] for row_id in target_eval.row_id]])
                mutant_columns.append(mutant_prediction)
                inference_audit.append({
                    "dataset": target, "representation": representation, "predictor": entry.predictor,
                    "seed": int(entry.seed), "variant": entry.variant, "parent_rows": len(parent_prediction_scope),
                    "mutant_rows": len(target_mutant_positions), "reference_max_abs_difference": maximum_difference,
                })
            parent_matrix = np.column_stack(parent_columns)
            mutant_matrix = np.column_stack(mutant_columns)
            for scenario_name, scenario_data in scenarios.items():
                target_scored = scenario_data["scored"][scenario_data["scored"].dataset == target]
                neighborhoods = {row_id: scenario_data["neighborhoods"][row_id] for row_id in target_eval.row_id}
                for item in metadata:
                    item["scenario"] = scenario_name
                mean_absolute, mean_signed, mean_support = mean_response_arrays(
                    target_eval, target_scored, mutant_matrix, parent_matrix, cfg["mask_size_exponents"],
                )
                lime_absolute, lime_signed, lime_support, local_rows = lime_arrays_and_diagnostics(
                    target_eval, neighborhoods, original_morgan, mutant_morgan, original_position,
                    mutant_matrix, parent_matrix, metadata, cfg["ridge_diagnostics"], scenario_name, representation,
                )
                fidelity_rows.extend(local_rows)
                arrays = {
                    **{f"crem_mean_abs_exponent_{float(exponent):g}": mean_absolute[exponent] for exponent in cfg["mask_size_exponents"]},
                    "crem_lime_abs": lime_absolute,
                }
                signed_arrays = {
                    **{f"crem_mean_signed_exponent_{float(exponent):g}": mean_signed[exponent] for exponent in cfg["mask_size_exponents"]},
                    "crem_lime_signed_parent_bit_loss": lime_signed,
                }
                combined_support = {
                    **{name: mean_support for name in arrays if name.startswith("crem_mean")},
                    "crem_lime_abs": lime_support,
                }
                add_pair_scores(
                    pair_rows, signed_rows, target_pairs, rationale_labels, arrays, signed_arrays,
                    combined_support,
                    metadata, parent_matrix, target_eval,
                )
            print(f"completed {target} {representation}", flush=True)
        if representation != "morgan":
            del original_values, mutant_values

    pair_frame = pd.DataFrame(pair_rows)
    signed_frame = pd.DataFrame(signed_rows)
    fidelity_frame = pd.DataFrame(fidelity_rows)
    inference_frame = pd.DataFrame(inference_audit)
    pair_frame.to_csv(output / "localization_pair_scores.csv", index=False)
    signed_frame.to_csv(output / "signed_local_sensitivity.csv", index=False)
    fidelity_frame.to_csv(output / "local_fidelity_diagnostics.csv", index=False)
    inference_frame.to_csv(output / "reinference_audit.csv", index=False)

    component, localization_summary = summarize_scores(
        pair_frame, int(cfg["bootstrap"]["draws"]), int(cfg["bootstrap"]["seed"]), "localization",
    )
    signed_component, signed_summary = summarize_scores(
        signed_frame, int(cfg["bootstrap"]["draws"]), int(cfg["bootstrap"]["seed"]), "signed",
    )
    component.to_csv(output / "localization_symmetric_component_deltas.csv", index=False)
    localization_summary.to_csv(output / "localization_summary.csv", index=False)
    signed_component.to_csv(output / "signed_symmetric_component_deltas.csv", index=False)
    signed_summary.to_csv(output / "signed_summary.csv", index=False)

    fidelity_target = fidelity_frame.groupby(["scenario", "dataset", "variant"], as_index=False).agg(
        median_in_sample_r2=("in_sample_weighted_r2", "median"), median_press_r2=("press_weighted_r2", "median"),
        median_five_fold_r2=("five_fold_weighted_r2", "median"), median_effective_df=("ridge_effective_df", "median"),
        median_local_samples=("local_samples", "median"), median_variable_bits=("variable_bits", "median"),
        five_fold_available_fraction=("five_fold_weighted_r2", lambda values: float(values.notna().mean())),
        endpoints=("row_id", "nunique"),
    )
    fidelity_target.to_csv(output / "local_fidelity_target_summary.csv", index=False)
    fidelity_summary = fidelity_target.groupby(["scenario", "variant"], as_index=False).agg(
        target_balanced_median_in_sample_r2=("median_in_sample_r2", "median"),
        target_balanced_median_press_r2=("median_press_r2", "median"),
        target_balanced_median_five_fold_r2=("median_five_fold_r2", "median"),
        target_balanced_median_effective_df=("median_effective_df", "median"),
        target_balanced_median_local_samples=("median_local_samples", "median"),
        target_balanced_median_variable_bits=("median_variable_bits", "median"),
        target_balanced_median_five_fold_available_fraction=("five_fold_available_fraction", "median"),
        targets=("dataset", "nunique"),
    )
    fidelity_summary.to_csv(output / "local_fidelity_summary.csv", index=False)
    solver_target = fidelity_frame.groupby(["scenario", "dataset", "variant"], as_index=False).agg(
        median_lsqr_vs_svd_fitted_max_abs_difference=("lsqr_vs_svd_fitted_max_abs_difference", "median"),
        maximum_lsqr_vs_svd_fitted_max_abs_difference=("lsqr_vs_svd_fitted_max_abs_difference", "max"),
        median_lsqr_vs_svd_coefficient_relative_l2_difference=("lsqr_vs_svd_coefficient_relative_l2_difference", "median"),
        q95_lsqr_vs_svd_coefficient_relative_l2_difference=("lsqr_vs_svd_coefficient_relative_l2_difference", lambda values: float(values.quantile(0.95))),
        maximum_lsqr_vs_svd_coefficient_relative_l2_difference=("lsqr_vs_svd_coefficient_relative_l2_difference", "max"),
        local_fits=("row_id", "size"),
    )
    solver_target.to_csv(output / "ridge_solver_target_diagnostics.csv", index=False)
    solver_summary = solver_target.groupby(["scenario", "variant"], as_index=False).agg(
        target_balanced_median_fitted_difference=("median_lsqr_vs_svd_fitted_max_abs_difference", "median"),
        maximum_fitted_difference=("maximum_lsqr_vs_svd_fitted_max_abs_difference", "max"),
        target_balanced_median_coefficient_relative_l2_difference=("median_lsqr_vs_svd_coefficient_relative_l2_difference", "median"),
        target_balanced_q95_coefficient_relative_l2_difference=("q95_lsqr_vs_svd_coefficient_relative_l2_difference", "median"),
        maximum_coefficient_relative_l2_difference=("maximum_lsqr_vs_svd_coefficient_relative_l2_difference", "max"),
        targets=("dataset", "nunique"),
    )
    solver_summary.to_csv(output / "ridge_solver_summary.csv", index=False)

    reference_pairs = pd.read_csv(paths["reference_pair_explanations"])
    reference_pairs = reference_pairs[reference_pairs.dataset.isin(targets) & (reference_pairs.component_split == "test")]
    reproduction = pair_frame[
        (pair_frame.scenario == "fixed_selected_support_0_60")
        & (pair_frame.rationale == "boundary_extended")
        & pair_frame.explainer.isin(["crem_mean_abs_exponent_0", "crem_lime_abs"])
    ].copy()
    reproduction["reference_explainer"] = reproduction.explainer.map({"crem_mean_abs_exponent_0": "crem_mean", "crem_lime_abs": "crem_lime"})
    reproduction_keys = ["dataset", "component_id", "representation", "predictor", "seed", "variant"]
    reproduced = reproduction.merge(
        reference_pairs[reproduction_keys + ["explainer", "average_precision", "normalized_ap_lift"]],
        left_on=reproduction_keys + ["reference_explainer"], right_on=reproduction_keys + ["explainer"],
        suffixes=("_rescore", "_reference"), validate="one_to_one",
    )
    tolerances = cfg["reference_reproduction_tolerances"]
    lime_tolerance = float(tolerances["pair_ap_or_nap_absolute"])
    reproduced["average_precision_abs_difference"] = (
        reproduced.average_precision_rescore - reproduced.average_precision_reference
    ).abs()
    reproduced["normalized_ap_lift_abs_difference"] = (
        reproduced.normalized_ap_lift_rescore - reproduced.normalized_ap_lift_reference
    ).abs()
    reproduced["exceeds_frozen_tolerance"] = (
        (reproduced.average_precision_abs_difference > lime_tolerance)
        | (reproduced.normalized_ap_lift_abs_difference > lime_tolerance)
    )
    reproduced.to_csv(output / "reference_pair_reproduction_diagnostics.csv", index=False)
    reproduction_by_explainer = reproduced.groupby("explainer_rescore", as_index=False).agg(
        rows=("component_id", "size"),
        maximum_ap_abs_difference=("average_precision_abs_difference", "max"),
        median_ap_abs_difference=("average_precision_abs_difference", "median"),
        mean_ap_abs_difference=("average_precision_abs_difference", "mean"),
        maximum_nap_abs_difference=("normalized_ap_lift_abs_difference", "max"),
        median_nap_abs_difference=("normalized_ap_lift_abs_difference", "median"),
        mean_nap_abs_difference=("normalized_ap_lift_abs_difference", "mean"),
        rows_above_frozen_tolerance=("exceeds_frozen_tolerance", "sum"),
    )
    reproduction_by_explainer.to_csv(output / "reference_pair_reproduction_summary.csv", index=False)
    mean_reproduction = reproduced[reproduced.explainer_rescore == "crem_mean_abs_exponent_0"]
    lime_reproduction = reproduced[reproduced.explainer_rescore == "crem_lime_abs"]
    mean_pair_ap_difference = float(mean_reproduction.average_precision_abs_difference.max())
    mean_pair_nap_difference = float(mean_reproduction.normalized_ap_lift_abs_difference.max())
    lime_pair_ap_difference = float(lime_reproduction.average_precision_abs_difference.max())
    lime_pair_nap_difference = float(lime_reproduction.normalized_ap_lift_abs_difference.max())
    affected_lime = lime_reproduction[lime_reproduction.exceeds_frozen_tolerance]

    reference_lime = reference_pairs[reference_pairs.explainer == "crem_lime"].copy()
    reference_lime["scenario"] = "fixed_selected_support_0_60"
    reference_lime["rationale"] = "boundary_extended"
    reference_lime["explainer"] = "crem_lime_abs"
    reference_component, _ = summarize_scores(
        reference_lime, int(cfg["bootstrap"]["draws"]), int(cfg["bootstrap"]["seed"]), "reference_localization",
    )
    comparison_keys = ["scenario", "rationale", "explainer", "dataset", "component_id", "representation", "predictor"]
    rescore_component = component[
        (component.scenario == "fixed_selected_support_0_60")
        & (component.rationale == "boundary_extended")
        & (component.explainer == "crem_lime_abs")
    ]
    estimand_components = rescore_component.merge(
        reference_component[comparison_keys + ["delta_nap"]], on=comparison_keys,
        suffixes=("_rescore", "_reference"), validate="one_to_one",
    )
    estimand_components["rescore_minus_reference_delta_nap"] = (
        estimand_components.delta_nap_rescore - estimand_components.delta_nap_reference
    )
    estimand_components.to_csv(output / "lime_estimand_reproduction_components.csv", index=False)
    bootstrap_draws = int(cfg["bootstrap"]["draws"])
    bootstrap_seed = int(cfg["bootstrap"]["seed"])
    estimand_values = {}
    for column, label in [
        ("delta_nap_reference", "lime_reference"),
        ("delta_nap_rescore", "localization:fixed_selected_support_0_60:boundary_extended:crem_lime_abs"),
        ("rescore_minus_reference_delta_nap", "lime_rescore_minus_reference"),
    ]:
        target_values = estimand_components.groupby("dataset")[column].median()
        ci = interval(two_stage_bootstrap(estimand_components, column, bootstrap_draws, bootstrap_seed, label))
        estimand_values[column] = {
            "estimate": float(target_values.median()), "ci95_low": ci[0], "ci95_high": ci[1],
            "conclusion": "positive" if ci[0] > 0 else ("negative" if ci[1] < 0 else "includes_zero"),
        }
    estimand_summary = {
        "reference_target_balanced_median_delta_nap": estimand_values["delta_nap_reference"]["estimate"],
        "reference_ci95_low": estimand_values["delta_nap_reference"]["ci95_low"],
        "reference_ci95_high": estimand_values["delta_nap_reference"]["ci95_high"],
        "reference_conclusion": estimand_values["delta_nap_reference"]["conclusion"],
        "rescore_target_balanced_median_delta_nap": estimand_values["delta_nap_rescore"]["estimate"],
        "rescore_ci95_low": estimand_values["delta_nap_rescore"]["ci95_low"],
        "rescore_ci95_high": estimand_values["delta_nap_rescore"]["ci95_high"],
        "rescore_conclusion": estimand_values["delta_nap_rescore"]["conclusion"],
        "rescore_minus_reference_target_balanced_median_delta_nap": estimand_values["rescore_minus_reference_delta_nap"]["estimate"],
        "difference_ci95_low": estimand_values["rescore_minus_reference_delta_nap"]["ci95_low"],
        "difference_ci95_high": estimand_values["rescore_minus_reference_delta_nap"]["ci95_high"],
        "conclusion_changed": bool(estimand_values["delta_nap_reference"]["conclusion"] != estimand_values["delta_nap_rescore"]["conclusion"]),
    }
    estimand_comparison = pd.DataFrame([estimand_summary])
    estimand_comparison.to_csv(output / "lime_estimand_reproduction_summary.csv", index=False)
    reference_fidelity = pd.read_csv(paths["reference_local_fidelity"])
    reference_fidelity = reference_fidelity[reference_fidelity.dataset.isin(targets) & (reference_fidelity.component_split == "test")]
    primary_fidelity = fidelity_frame[fidelity_frame.scenario == "fixed_selected_support_0_60"]
    fidelity_keys = ["dataset", "row_id", "representation", "predictor", "seed", "variant"]
    reproduced_fidelity = primary_fidelity.merge(
        reference_fidelity[fidelity_keys + ["local_weighted_r2"]], on=fidelity_keys, validate="one_to_one",
    )
    fidelity_difference = float((reproduced_fidelity.in_sample_weighted_r2 - reproduced_fidelity.local_weighted_r2).abs().max())
    maximum_prediction_difference = float(inference_frame.reference_max_abs_difference.max())
    maximum_hat_difference = float(fidelity_frame.hat_fit_max_abs_difference.max())
    core_gate_failed = (
        maximum_prediction_difference > float(tolerances["parent_prediction_absolute"])
        or max(mean_pair_ap_difference, mean_pair_nap_difference) > float(tolerances["pair_ap_or_nap_absolute"])
        or fidelity_difference > float(tolerances["local_r2_absolute"])
    )
    if core_gate_failed:
        raise ValueError("primary rescore failed prediction, mean-explainer, or fidelity reproduction")
    lime_rank_instability = max(lime_pair_ap_difference, lime_pair_nap_difference) > lime_tolerance
    run_status = "completed_with_failed_lime_reproduction_gate" if lime_rank_instability else "success"

    summary = {
        "run_id": cfg["run_id"], "status": run_status, "smoke": args.smoke,
        "targets": targets, "test_pairs": len(pairs), "test_endpoints": len(evaluation),
        "saved_models_reinferred": len(inference_frame), "neighborhood_scenarios": list(scenarios),
        "rationale_pairs": {
            name: {
                "valid": int(rationale_frame[(rationale_frame.rationale == name) & rationale_frame.valid_normalized_ap].shape[0]),
                "invalid": int(rationale_frame[(rationale_frame.rationale == name) & ~rationale_frame.valid_normalized_ap].shape[0]),
            }
            for name in cfg["rationales"]
        },
        "support_0_50": {
            "status": "unavailable_exactly_from_cached_representations",
            "missing_selected_mutant_rows": int(availability.loc[availability.threshold == 0.5, "exact_reselection_missing_representation_rows"].iloc[0]),
            "affected_endpoints": int(availability.loc[availability.threshold == 0.5, "exact_reselection_affected_endpoints"].iloc[0]),
        },
        "reference_reproduction": {
            "hard_gate": "failed_lime_attribution_rank_reproduction" if lime_rank_instability else "passed",
            "core_prediction_mean_fidelity_gate": "passed",
            "maximum_parent_prediction_abs_difference": maximum_prediction_difference,
            "maximum_mean_pair_ap_abs_difference": mean_pair_ap_difference,
            "maximum_mean_pair_nap_abs_difference": mean_pair_nap_difference,
            "maximum_lime_pair_ap_abs_difference": lime_pair_ap_difference,
            "maximum_lime_pair_nap_abs_difference": lime_pair_nap_difference,
            "lime_rows_above_pair_tolerance": len(affected_lime),
            "lime_rows_compared": len(lime_reproduction),
            "affected_target_pairs": int(affected_lime[["dataset", "component_id"]].drop_duplicates().shape[0]),
            "affected_target_specific_model_instances": int(affected_lime[["dataset", "representation", "predictor", "seed", "variant"]].drop_duplicates().shape[0]),
            "affected_global_model_configurations": int(affected_lime[["representation", "predictor", "seed", "variant"]].drop_duplicates().shape[0]),
            "affected_targets": int(affected_lime.dataset.nunique()),
            "affected_rows_by_variant": {str(key): int(value) for key, value in affected_lime.groupby("variant").size().items()},
            "maximum_local_in_sample_r2_abs_difference": fidelity_difference,
            "maximum_closed_form_hat_fit_abs_difference": maximum_hat_difference,
            "tolerances": tolerances,
        },
        "lime_estimand_reproduction": estimand_summary,
        "ridge_solver_numerical_diagnostic": {
            "comparison": "lsqr_primary_vs_analytic_svd_equivalent_ridge",
            "maximum_fitted_abs_difference": float(fidelity_frame.lsqr_vs_svd_fitted_max_abs_difference.max()),
            "median_coefficient_relative_l2_difference": float(fidelity_frame.lsqr_vs_svd_coefficient_relative_l2_difference.median()),
            "q95_coefficient_relative_l2_difference": float(fidelity_frame.lsqr_vs_svd_coefficient_relative_l2_difference.quantile(0.95)),
            "maximum_coefficient_relative_l2_difference": float(fidelity_frame.lsqr_vs_svd_coefficient_relative_l2_difference.max()),
            "role": "numerical diagnostic only; never substituted for the primary LSQR explanation",
        },
        "missing_diagnostics": {
            "press_weighted_r2": int(fidelity_frame.press_weighted_r2.isna().sum()),
            "five_fold_weighted_r2": int(fidelity_frame.five_fold_weighted_r2.isna().sum()),
            "coefficient_relative_l2_difference": int(fidelity_frame.lsqr_vs_svd_coefficient_relative_l2_difference.isna().sum()),
        },
        "rows": {
            "localization_pair_scores": len(pair_frame), "signed_local_sensitivity": len(signed_frame),
            "local_fidelity_diagnostics": len(fidelity_frame),
        },
        "input_hashes": {
            name: sha256(path) for name, path in paths.items() if path.is_file()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    dictionary = """# Rescore-control data dictionary

This is a result-blind six-target sensitivity panel that reuses frozen trained and label-permuted heads. It performs new inference on cached parent and mutant representations; it does not train a model, regenerate a CReM library, or modify the original outputs.

- `fixed_selected_support_0_60` reproduces the frozen four-selected-mutant neighborhood.
- `fixed_selected_support_0_70_restriction` removes cached selected mutants below parent Morgan Tanimoto 0.70. It is a restriction of the frozen selected set, not an exact re-selection from all generated mutants.
- `fixed_selected_absolute_domain_only` retains frozen selected mutants that also meet the target-specific absolute-domain criterion.
- An exact 0.50 sensitivity cannot be reconstructed from cached representations. Lowering the threshold changes deterministic mutant selection, and the missing selected-mutant representations are counted in `support_threshold_availability.csv`. No zero-filled or proxy representation is used.
- `boundary_extended` is the original RASCAL unmatched/bond/boundary proxy. `strict_unmatched` labels only atoms absent from the stored atom match. Pairs with zero or all positive atoms remain visible in `rationale_audit.csv` and are excluded only from normalized AP.
- `crem_mean_abs_exponent_0` and `_0.5` are mean absolute mutant-minus-parent responses divided by mask size to the named exponent. `crem_lime_abs` reproduces the original absolute parent-bit coefficient mapping.
- Signed mean scores retain mutant-minus-parent direction. The left endpoint is multiplied by `-sign(y_left-y_right)` and the right endpoint by `+sign(y_left-y_right)`, so positive values indicate a local perturbation moving a prediction toward the observed partner direction. The signed CReM-LIME score uses the negative signed parent-bit coefficient as an approximate parent-bit-loss response before the same orientation. These are direction-aligned local sensitivities, not complete pairwise attributions, causal SAR decompositions, or conservation identities.
- `local_fidelity_diagnostics.csv` reports the original in-sample distance-weighted R2, deterministic five-fold out-of-fold R2, analytic weighted ridge PRESS R2, and effective degrees of freedom `trace(H)`. Ridge slopes use alpha 1 and the intercept is unpenalized. PRESS uses weighted residuals divided by `1-h_ii`. Five-fold and PRESS values remain missing for insufficient samples or constant response rather than being replaced by zero.
- `ridge_solver_target_diagnostics.csv` and `ridge_solver_summary.csv` compare the primary LSQR ridge fit with an analytic SVD-equivalent weighted ridge solution using identical cached predictions. Coefficient and fitted-value differences are pure numerical diagnostics; the SVD-equivalent coefficients never replace the primary explanation.
- `localization_summary.csv` and `signed_summary.csv` use a symmetric mean-trained-seed minus mean-permuted-seed component contrast, equal target weighting, and a target/component two-stage bootstrap. With six targets these intervals are exploratory sensitivity intervals, not replacements for the 27-target primary inference.

The primary cached-neighborhood rescore is checked against saved parent predictions, mean-replacement AP/nAP, and in-sample fidelity as hard gates. `reference_pair_reproduction_diagnostics.csv` also retains every original-versus-rescore LIME AP/nAP difference. GPU rounding near 1e-6 can move the LSQR solution within a collinear local design and change ranks among tied or nearly tied atom scores discontinuously; such LIME differences are reported as attribution-rank instability rather than hidden by relaxing the frozen tolerance.
"""
    (output / "DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")

    input_files = [args.config.resolve(), resolve(cfg["source_config"])]
    for representation in source_cfg["representations"]:
        input_files.extend([
            paths["original_representations"] / f"{representation}.npz",
            paths["mutant_representations"] / f"{representation}.npz",
        ])
    input_files.extend(path for path in paths.values() if path.is_file())
    outputs = sorted(path for path in output.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "run_id": cfg["run_id"], "status": run_status, "smoke": args.smoke,
        "command": "python research_v2/scripts/rescore_major_revision_controls.py --config research_v2/configs/major_revision_rescore_v1.json" + (" --smoke" if args.smoke else ""),
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        },
        "script_sha256": sha256(Path(__file__)), "input_hashes": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in dict.fromkeys(input_files)
        },
        "output_hashes": {path.name: sha256(path) for path in outputs},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
