import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.metrics import average_precision_score


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


def require(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def interval(values, level=0.95):
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def rng_for(seed, label):
    offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")
    return np.random.default_rng((int(seed) + offset) % (2**32))


def two_stage_bootstrap(frame, value, draws, seed, label):
    grouped = {
        target: [part[value].to_numpy(float) for _, part in target_frame.groupby("component_id", sort=True)]
        for target, target_frame in frame.groupby("dataset", sort=True)
    }
    names = sorted(grouped)
    if not names:
        return np.array([], dtype=float)
    rng = rng_for(seed, label)
    estimates = np.empty(draws, dtype=float)
    for draw in range(draws):
        target_values = []
        for target_index in rng.integers(0, len(names), len(names)):
            components = grouped[names[target_index]]
            sampled = [components[index] for index in rng.integers(0, len(components), len(components))]
            target_values.append(np.median(np.concatenate(sampled)))
        estimates[draw] = np.median(target_values)
    return estimates


def median_bootstrap(values, draws, seed, label):
    values = np.asarray(values, dtype=float)
    rng = rng_for(seed, label)
    return np.median(values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1)


def mean_bootstrap(values, draws, seed, label):
    values = np.asarray(values, dtype=float)
    rng = rng_for(seed, label)
    return np.mean(values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1)


def normalized_ap(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    prevalence = float(labels.mean())
    if not 0.0 < prevalence < 1.0:
        raise ValueError("normalized AP requires 0 < prevalence < 1")
    ap = float(average_precision_score(labels, np.asarray(scores, dtype=float)))
    return ap, prevalence, float((ap - prevalence) / (1.0 - prevalence))


def prediction_metrics(frame):
    true = frame.true_delta.to_numpy(float)
    predicted = frame.predicted_delta.to_numpy(float)
    rho = stats.spearmanr(true, predicted).statistic if np.ptp(predicted) > 0 else np.nan
    slope, intercept = np.polyfit(true, predicted, 1) if np.ptp(true) > 0 else (np.nan, np.nan)
    error = predicted - true
    return pd.Series({
        "pairs": len(frame),
        "spearman": rho,
        "direction_accuracy": float(np.mean(np.sign(true) == np.sign(predicted))),
        "delta_mae": float(np.mean(np.abs(error))),
        "delta_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "mean_absolute_predicted_delta": float(np.mean(np.abs(predicted))),
    })


def parse_atoms(value):
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(atom, int) for atom in parsed):
        raise ValueError(f"invalid atom-index list: {value}")
    return parsed


def quantile(values, q):
    return float(np.quantile(values, q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "research_v2/configs/major_revision_v1.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source_cfg_path = resolve(cfg["source_config"])
    source_cfg = json.loads(source_cfg_path.read_text(encoding="utf-8"))
    paths = {name: resolve(path) for name, path in cfg["inputs"].items()}
    output = resolve(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    draws = int(cfg["bootstrap"]["draws"])
    seed = int(cfg["bootstrap"]["seed"])
    confirmation = list(source_cfg["confirmation_targets"])
    confirmation_set = set(confirmation)
    representations = list(source_cfg["representations"])
    predictors = ["xgboost", "mlp"]
    explainers = list(source_cfg["explainers"])
    seeds = list(source_cfg["seeds"])

    pairs = pd.read_csv(paths["pair_explanations"])
    pair_predictions = pd.read_csv(paths["pair_predictions"])
    local = pd.read_csv(paths["local_fidelity"])
    rationale_audit = pd.read_csv(paths["rationale_audit"])
    masks = pd.read_csv(paths["mask_coverage"])
    scored = pd.read_csv(paths["scored_mutants"])
    sentinels = pd.read_csv(paths["sentinel_pairs"])
    molecules = pd.read_csv(paths["molecules"])
    model_molecules = pd.read_csv(paths["model_molecules"])
    evaluation = pd.read_csv(paths["evaluation_molecules"])

    require(pairs, ["dataset", "component_id", "component_split", "left_index", "right_index", "representation", "predictor", "explainer", "seed", "variant", "average_precision", "prevalence", "normalized_ap_lift", "predicted_delta"], "pair_explanations")
    require(pair_predictions, ["dataset", "component_id", "component_split", "representation", "predictor", "seed", "variant", "true_delta", "predicted_delta"], "pair_predictions")
    require(local, ["row_id", "dataset", "component_split", "local_weighted_r2", "local_samples", "variable_bits", "supported_atom_fraction", "representation", "predictor", "seed", "variant"], "local_fidelity")
    require(rationale_audit, ["dataset", "component_id", "component_split", "left_index", "right_index", "positive_atoms", "total_atoms", "valid_normalized_ap_prevalence"], "rationale_audit")
    require(masks, ["row_id", "dataset", "component_split", "parent_atoms", "mask_id", "mask_atoms", "mask_size", "valid_unique_mutants", "local_supported_mutants", "absolute_in_domain_mutants", "covered"], "mask_coverage")
    require(scored, ["row_id", "dataset", "mask_id", "mask_atoms", "parent_mutant_tanimoto", "nearest_train_tanimoto", "local_supported", "scored_in_pilot"], "scored_mutants")
    require(sentinels, ["dataset", "component_id", "component_split", "left_index", "right_index", "rationale_left", "rationale_right"], "sentinel_pairs")
    if set(pairs.variant) != {"trained", "label_permuted"} or set(pair_predictions.variant) != {"trained", "label_permuted"}:
        raise ValueError("unexpected model variants")
    if set(pairs.representation) != set(representations) or set(pairs.predictor) != set(predictors) or set(pairs.explainer) != set(explainers) or set(pairs.seed) != set(seeds):
        raise ValueError("pair-explanation factorial crossing differs from source contract")
    pair_key = ["dataset", "component_id", "representation", "predictor", "explainer", "seed", "variant"]
    prediction_key = ["dataset", "component_id", "representation", "predictor", "seed", "variant"]
    local_key = ["row_id", "representation", "predictor", "seed", "variant"]
    if pairs.duplicated(pair_key).any() or pair_predictions.duplicated(prediction_key).any() or local.duplicated(local_key).any():
        raise ValueError("non-unique analysis key in a source table")

    exp = pairs[(pairs.dataset.isin(confirmation_set)) & (pairs.component_split == "test")].copy()
    pred = pair_predictions[(pair_predictions.dataset.isin(confirmation_set)) & (pair_predictions.component_split == "test")].copy()
    fidelity = local[(local.dataset.isin(confirmation_set)) & (local.component_split == "test")].copy()
    config_keys = ["representation", "predictor", "explainer"]
    component_keys = ["dataset", "component_id", "left_index", "right_index"] + config_keys

    component_variant = exp.groupby(component_keys + ["variant"], as_index=False).agg(
        average_precision=("average_precision", "mean"),
        normalized_ap_lift=("normalized_ap_lift", "mean"),
        prevalence=("prevalence", "mean"),
        seeds=("seed", "nunique"),
    )
    if not (component_variant.seeds == len(seeds)).all():
        raise ValueError("incomplete seed crossing in confirmation explanations")
    trained = component_variant[component_variant.variant == "trained"].drop(columns=["variant", "seeds"]).rename(columns={
        "average_precision": "trained_ap", "normalized_ap_lift": "trained_nap", "prevalence": "trained_prevalence",
    })
    randomized = component_variant[component_variant.variant == "label_permuted"].drop(columns=["variant", "seeds"]).rename(columns={
        "average_precision": "permuted_ap", "normalized_ap_lift": "permuted_nap", "prevalence": "permuted_prevalence",
    })
    component = trained.merge(randomized, on=component_keys, validate="one_to_one")
    if not np.allclose(component.trained_prevalence, component.permuted_prevalence, atol=1e-12):
        raise ValueError("rationale prevalence differs by variant")
    component["prevalence"] = component.trained_prevalence
    component["delta_ap"] = component.trained_ap - component.permuted_ap
    component["delta_nap"] = component.trained_nap - component.permuted_nap
    component.drop(columns=["trained_prevalence", "permuted_prevalence"], inplace=True)
    component.to_csv(output / "symmetric_component_metrics.csv", index=False)

    target_config = component.groupby(["dataset"] + config_keys, as_index=False).agg(
        components=("component_id", "nunique"),
        trained_ap=("trained_ap", "median"),
        permuted_ap=("permuted_ap", "median"),
        trained_nap=("trained_nap", "median"),
        permuted_nap=("permuted_nap", "median"),
        prevalence=("prevalence", "median"),
        median_delta_ap=("delta_ap", "median"),
        median_delta_nap=("delta_nap", "median"),
    )
    cell_rows = []
    for key, frame in component.groupby(["dataset"] + config_keys, sort=True):
        distribution = median_bootstrap(frame.delta_nap, draws, seed, "cell:" + ":".join(map(str, key)))
        ci90 = interval(distribution, 0.90)
        ci95 = interval(distribution, 0.95)
        cell_rows.append({
            "dataset": key[0], "representation": key[1], "predictor": key[2], "explainer": key[3],
            "ci90_low": ci90[0], "ci90_high": ci90[1], "ci95_low": ci95[0], "ci95_high": ci95[1],
        })
    target_config = target_config.merge(pd.DataFrame(cell_rows), on=["dataset"] + config_keys, validate="one_to_one")
    target_config.to_csv(output / "target_configuration_absolute_metrics.csv", index=False)

    configuration_rows = []
    for key, frame in component.groupby(config_keys, sort=True):
        targets = target_config[(target_config.representation == key[0]) & (target_config.predictor == key[1]) & (target_config.explainer == key[2])]
        distribution = two_stage_bootstrap(frame, "delta_nap", draws, seed, "configuration:" + ":".join(key))
        ci90 = interval(distribution, 0.90)
        ci95 = interval(distribution, 0.95)
        configuration_rows.append({
            "representation": key[0], "predictor": key[1], "explainer": key[2],
            "target_balanced_median_trained_ap": float(targets.trained_ap.median()),
            "target_balanced_median_permuted_ap": float(targets.permuted_ap.median()),
            "target_balanced_median_trained_nap": float(targets.trained_nap.median()),
            "target_balanced_median_permuted_nap": float(targets.permuted_nap.median()),
            "target_balanced_median_prevalence": float(targets.prevalence.median()),
            "target_balanced_median_delta_nap": float(targets.median_delta_nap.median()),
            "delta_ci90_low": ci90[0], "delta_ci90_high": ci90[1],
            "delta_ci95_low": ci95[0], "delta_ci95_high": ci95[1],
            "nominally_equivalent_targets_ci90_margin_0_05": int(((targets.ci90_low >= -0.05) & (targets.ci90_high <= 0.05)).sum()),
            "nominally_equivalent_targets_ci95_margin_0_05": int(((targets.ci95_low >= -0.05) & (targets.ci95_high <= 0.05)).sum()),
            "targets": len(targets), "components": int(frame[["dataset", "component_id"]].drop_duplicates().shape[0]),
        })
    configuration_summary = pd.DataFrame(configuration_rows)
    configuration_summary.to_csv(output / "configuration_absolute_metrics.csv", index=False)

    target_primary = component.groupby("dataset", as_index=False).agg(
        median_delta_nap=("delta_nap", "median"), components=("component_id", "nunique"), rows=("delta_nap", "size"),
    )
    target_primary.to_csv(output / "symmetric_target_estimates.csv", index=False)
    primary_bootstrap = two_stage_bootstrap(component, "delta_nap", draws, seed, "symmetric-primary")
    pd.DataFrame({"draw": np.arange(draws), "estimate": primary_bootstrap}).to_csv(output / "symmetric_primary_bootstrap.csv", index=False)
    primary_ci90 = interval(primary_bootstrap, 0.90)
    primary_ci95 = interval(primary_bootstrap, 0.95)

    margin_rows = []
    for margin in cfg["equivalence_margins"]:
        for level, low_name, high_name in [(0.90, "ci90_low", "ci90_high"), (0.95, "ci95_low", "ci95_high")]:
            equivalent = (target_config[low_name] >= -float(margin)) & (target_config[high_name] <= float(margin))
            primary_ci = primary_ci90 if level == 0.90 else primary_ci95
            margin_rows.append({
                "margin": float(margin), "confidence_level": level,
                "primary_ci_low": primary_ci[0], "primary_ci_high": primary_ci[1],
                "primary_equivalent": primary_ci[0] >= -float(margin) and primary_ci[1] <= float(margin),
                "nominally_equivalent_target_configuration_cells": int(equivalent.sum()),
                "target_configuration_cells": len(target_config),
                "nominally_equivalent_fraction": float(equivalent.mean()),
            })
    pd.DataFrame(margin_rows).to_csv(output / "margin_and_ci_sensitivity.csv", index=False)

    seed_group = ["dataset", "component_id"] + config_keys
    trained_seed = exp[exp.variant == "trained"].groupby(seed_group).normalized_ap_lift.std().rename("trained_seed_sd")
    permuted_seed = exp[exp.variant == "label_permuted"].groupby(seed_group).normalized_ap_lift.std().rename("permuted_seed_sd")
    matched_seed = exp.pivot(index=seed_group + ["seed"], columns="variant", values="normalized_ap_lift").reset_index()
    matched_seed["matched_delta"] = matched_seed.trained - matched_seed.label_permuted
    matched_sd = matched_seed.groupby(seed_group).matched_delta.std().rename("matched_delta_seed_sd")
    seed_variability = pd.concat([trained_seed, permuted_seed, matched_sd], axis=1).reset_index()
    seed_variability.to_csv(output / "seed_variability_component.csv", index=False)
    seed_target = seed_variability.groupby("dataset", as_index=False).agg(
        median_trained_seed_sd=("trained_seed_sd", "median"),
        median_permuted_seed_sd=("permuted_seed_sd", "median"),
        median_matched_delta_seed_sd=("matched_delta_seed_sd", "median"),
    )
    seed_target.to_csv(output / "seed_variability_target_summary.csv", index=False)
    seed_summary = pd.DataFrame([{
        "target_balanced_median_trained_seed_sd": float(seed_target.median_trained_seed_sd.median()),
        "target_balanced_median_permuted_seed_sd": float(seed_target.median_permuted_seed_sd.median()),
        "target_balanced_median_matched_delta_seed_sd": float(seed_target.median_matched_delta_seed_sd.median()),
        "frozen_margin": float(source_cfg["inference"]["equivalence_margin"]),
        "frozen_margin_over_trained_seed_sd": float(source_cfg["inference"]["equivalence_margin"] / seed_target.median_trained_seed_sd.median()),
    }])
    seed_summary.to_csv(output / "seed_variability_summary.csv", index=False)

    model_cell = pred.groupby(["dataset", "representation", "predictor", "seed", "variant"], sort=True).apply(prediction_metrics, include_groups=False).reset_index()
    model_cell.to_csv(output / "prediction_model_cell_metrics.csv", index=False)
    prediction_target_config = model_cell.groupby(["dataset", "representation", "predictor", "variant"], as_index=False).agg(
        pairs=("pairs", "max"), spearman=("spearman", "median"), direction_accuracy=("direction_accuracy", "median"),
        delta_mae=("delta_mae", "median"), delta_rmse=("delta_rmse", "median"),
        calibration_slope=("calibration_slope", "median"), calibration_intercept=("calibration_intercept", "median"),
        mean_absolute_predicted_delta=("mean_absolute_predicted_delta", "median"),
    )
    prediction_target_config.to_csv(output / "prediction_target_configuration_metrics.csv", index=False)
    prediction_configuration = prediction_target_config.groupby(["representation", "predictor", "variant"], as_index=False).agg(
        target_balanced_median_spearman=("spearman", "median"),
        target_balanced_median_direction_accuracy=("direction_accuracy", "median"),
        target_balanced_median_delta_mae=("delta_mae", "median"),
        target_balanced_median_delta_rmse=("delta_rmse", "median"),
        target_balanced_median_calibration_slope=("calibration_slope", "median"),
        target_balanced_median_calibration_intercept=("calibration_intercept", "median"),
        target_balanced_median_absolute_predicted_delta=("mean_absolute_predicted_delta", "median"),
        targets=("dataset", "nunique"),
    )
    prediction_configuration.to_csv(output / "prediction_configuration_summary.csv", index=False)
    trained_prediction = prediction_target_config[prediction_target_config.variant == "trained"].drop(columns="variant")
    permuted_prediction = prediction_target_config[prediction_target_config.variant == "label_permuted"].drop(columns="variant")
    prediction_difference = trained_prediction.merge(permuted_prediction, on=["dataset", "representation", "predictor"], suffixes=("_trained", "_permuted"), validate="one_to_one")
    for metric in ["spearman", "direction_accuracy", "calibration_slope", "calibration_intercept", "mean_absolute_predicted_delta"]:
        prediction_difference[f"trained_minus_permuted_{metric}"] = prediction_difference[f"{metric}_trained"] - prediction_difference[f"{metric}_permuted"]
    for metric in ["delta_mae", "delta_rmse"]:
        prediction_difference[f"permuted_minus_trained_{metric}"] = prediction_difference[f"{metric}_permuted"] - prediction_difference[f"{metric}_trained"]
    prediction_difference.to_csv(output / "prediction_randomization_differences.csv", index=False)
    prediction_change_rows = []
    change_metrics = [
        "trained_minus_permuted_spearman", "trained_minus_permuted_direction_accuracy",
        "permuted_minus_trained_delta_mae", "permuted_minus_trained_delta_rmse",
        "trained_minus_permuted_calibration_slope", "trained_minus_permuted_mean_absolute_predicted_delta",
    ]
    for metric in change_metrics:
        target_values = prediction_difference.groupby("dataset")[metric].median()
        distribution = median_bootstrap(target_values, draws, seed, "prediction-change:" + metric)
        ci = interval(distribution)
        prediction_change_rows.append({
            "metric": metric, "target_balanced_median": float(target_values.median()),
            "ci95_low": ci[0], "ci95_high": ci[1], "positive_target_medians": int((target_values > 0).sum()),
            "targets": len(target_values),
        })
    pd.DataFrame(prediction_change_rows).to_csv(output / "prediction_randomization_summary.csv", index=False)

    random_mean = exp[exp.variant == "label_permuted"].groupby(component_keys, as_index=False).normalized_ap_lift.mean().rename(columns={"normalized_ap_lift": "permuted_mean_nap"})
    aligned = exp[exp.variant == "trained"].merge(random_mean, on=component_keys, validate="many_to_one")
    aligned["delta_to_permuted_mean"] = aligned.normalized_ap_lift - aligned.permuted_mean_nap
    aligned["correct_direction"] = aligned.true_delta * aligned.predicted_delta > 0
    criteria = [("all_eligible_cliffs", aligned)]
    for threshold in cfg["prediction_alignment"]["absolute_predicted_delta_thresholds"]:
        label = "correct_direction" if float(threshold) == 0 else f"correct_direction_abs_pred_ge_{float(threshold):g}"
        criteria.append((label, aligned[aligned.correct_direction & (aligned.predicted_delta.abs() >= float(threshold))]))
    aligned_target_rows = []
    aligned_summary_rows = []
    for label, frame in criteria:
        target_values = frame.groupby("dataset", as_index=False).agg(
            median_delta=("delta_to_permuted_mean", "median"),
            components=("component_id", "nunique"), rows=("delta_to_permuted_mean", "size"),
        )
        target_values.insert(0, "criterion", label)
        aligned_target_rows.append(target_values)
        distribution = two_stage_bootstrap(frame, "delta_to_permuted_mean", draws, seed, "alignment:" + label)
        ci = interval(distribution, 0.95)
        aligned_summary_rows.append({
            "criterion": label, "target_balanced_median_delta": float(target_values.median_delta.median()),
            "ci95_low": ci[0], "ci95_high": ci[1], "eligible_targets": frame.dataset.nunique(),
            "eligible_components": frame[["dataset", "component_id"]].drop_duplicates().shape[0],
            "eligible_model_pairs": frame[["dataset", "component_id", "representation", "predictor", "seed"]].drop_duplicates().shape[0],
            "explanation_rows": len(frame),
        })
    pd.concat(aligned_target_rows, ignore_index=True).to_csv(output / "prediction_aligned_target_estimates.csv", index=False)
    pd.DataFrame(aligned_summary_rows).to_csv(output / "prediction_aligned_sensitivity.csv", index=False)

    eligible_audit = rationale_audit[(rationale_audit.dataset.isin(confirmation_set)) & (rationale_audit.component_split == "test") & rationale_audit.valid_normalized_ap_prevalence.astype(bool)]
    eligible_pairs = sentinels.merge(
        eligible_audit[["dataset", "component_id", "left_index", "right_index"]],
        on=["dataset", "component_id", "left_index", "right_index"], validate="one_to_one",
    )
    if len(eligible_pairs) != exp[["dataset", "component_id"]].drop_duplicates().shape[0]:
        raise ValueError("eligible sentinel pairs do not match explanation pairs")
    endpoint_rows = []
    for row in eligible_pairs.itertuples(index=False):
        endpoint_rows.extend([(f"{row.dataset}:{int(row.left_index)}", row.dataset, int(row.component_id), "left"), (f"{row.dataset}:{int(row.right_index)}", row.dataset, int(row.component_id), "right")])
    endpoint_map = pd.DataFrame(endpoint_rows, columns=["row_id", "dataset", "component_id", "pair_side"])
    if endpoint_map.row_id.duplicated().any():
        raise ValueError("an endpoint occurs in more than one eligible sentinel pair")
    eligible_endpoints = set(endpoint_map.row_id)
    endpoint_masks = masks[masks.row_id.isin(eligible_endpoints)].copy()
    if set(endpoint_masks.row_id) != eligible_endpoints:
        raise ValueError("missing mask coverage for an eligible endpoint")
    valid_masks = endpoint_masks[endpoint_masks.mask_id >= 0].copy()
    valid_masks["atoms"] = valid_masks.mask_atoms.map(parse_atoms)
    for row in valid_masks.itertuples(index=False):
        if not row.atoms or len(row.atoms) != int(row.mask_size) or any(atom < 0 or atom >= int(row.parent_atoms) for atom in row.atoms):
            raise ValueError(f"ambiguous or out-of-range mask mapping: {row.row_id}/{row.mask_id}")
    scored_masks = scored[scored.row_id.isin(eligible_endpoints)].groupby(["row_id", "mask_id", "mask_atoms"], as_index=False).agg(
        selected_mutants=("mutant_smiles", "size"),
        selected_mean_parent_tanimoto=("parent_mutant_tanimoto", "mean"),
        selected_mean_nearest_train_tanimoto=("nearest_train_tanimoto", "mean"),
        all_selected_local_supported=("local_supported", "all"),
        all_selected_flagged=("scored_in_pilot", "all"),
    )
    mapping_check = scored_masks.merge(valid_masks[["row_id", "mask_id", "mask_atoms"]], on=["row_id", "mask_id"], how="left", suffixes=("_scored", "_coverage"), indicator=True)
    if (mapping_check._merge != "both").any() or (mapping_check.mask_atoms_scored != mapping_check.mask_atoms_coverage).any() or not scored_masks.all_selected_local_supported.all() or not scored_masks.all_selected_flagged.all():
        raise ValueError("scored-mutant masks cannot be mapped exactly to mask coverage")
    valid_masks = valid_masks.merge(scored_masks.drop(columns=["all_selected_local_supported", "all_selected_flagged"]), on=["row_id", "mask_id", "mask_atoms"], how="left", validate="one_to_one")
    valid_masks["selected_mutants"] = valid_masks.selected_mutants.fillna(0)
    valid_masks["selected_mean_parent_tanimoto"] = valid_masks.selected_mean_parent_tanimoto.fillna(0.0)
    valid_masks["selected_mean_nearest_train_tanimoto"] = valid_masks.selected_mean_nearest_train_tanimoto.fillna(0.0)
    if ((valid_masks.local_supported_mutants > 0) != (valid_masks.selected_mutants > 0)).any():
        raise ValueError("covered masks and selected generator mutants disagree")

    baseline_names = [
        "support_any", "mask_membership_count", "valid_mutant_count", "local_supported_mutant_count",
        "absolute_in_domain_mutant_count", "local_support_fraction", "selected_mutant_count",
        "selected_mean_parent_tanimoto", "selected_mean_nearest_train_tanimoto", "inverse_mask_size",
    ]
    atom_arrays = {}
    for row_id, frame in endpoint_masks.groupby("row_id", sort=True):
        atom_count_values = frame.parent_atoms.unique()
        if len(atom_count_values) != 1:
            raise ValueError(f"inconsistent parent atom count: {row_id}")
        atom_count = int(atom_count_values[0])
        sums = {name: np.zeros(atom_count, dtype=float) for name in baseline_names}
        memberships = np.zeros(atom_count, dtype=float)
        for mask in valid_masks[valid_masks.row_id == row_id].itertuples(index=False):
            local_fraction = float(mask.local_supported_mutants / mask.valid_unique_mutants) if mask.valid_unique_mutants else 0.0
            values = {
                "support_any": float(bool(mask.covered)), "mask_membership_count": 1.0,
                "valid_mutant_count": float(mask.valid_unique_mutants),
                "local_supported_mutant_count": float(mask.local_supported_mutants),
                "absolute_in_domain_mutant_count": float(mask.absolute_in_domain_mutants),
                "local_support_fraction": local_fraction, "selected_mutant_count": float(mask.selected_mutants),
                "selected_mean_parent_tanimoto": float(mask.selected_mean_parent_tanimoto),
                "selected_mean_nearest_train_tanimoto": float(mask.selected_mean_nearest_train_tanimoto),
                "inverse_mask_size": 1.0 / float(mask.mask_size),
            }
            for atom in mask.atoms:
                memberships[atom] += 1.0
                for name, value in values.items():
                    if name == "support_any":
                        sums[name][atom] = max(sums[name][atom], value)
                    else:
                        sums[name][atom] += value
        for name in baseline_names:
            if name not in {"support_any", "mask_membership_count"}:
                sums[name] = np.divide(sums[name], memberships, out=np.zeros(atom_count), where=memberships > 0)
        atom_arrays[row_id] = sums

    pair_lookup = {(row.dataset, int(row.component_id)): row for row in eligible_pairs.itertuples(index=False)}
    generator_atoms = []
    generator_pairs = []
    support_rows = []
    random_rows = []
    random_replicates = int(cfg["random_atom_baseline_replicates"])
    for (dataset, component_id), pair in sorted(pair_lookup.items()):
        left_id = f"{dataset}:{int(pair.left_index)}"
        right_id = f"{dataset}:{int(pair.right_index)}"
        left_count = len(atom_arrays[left_id]["support_any"])
        right_count = len(atom_arrays[right_id]["support_any"])
        left_labels = np.isin(np.arange(left_count), parse_atoms(pair.rationale_left))
        right_labels = np.isin(np.arange(right_count), parse_atoms(pair.rationale_right))
        labels = np.concatenate([left_labels, right_labels])
        if labels.sum() == 0 or labels.all():
            raise ValueError(f"invalid eligible rationale after mapping: {dataset}/{component_id}")
        for side, row_id, side_labels in [("left", left_id, left_labels), ("right", right_id, right_labels)]:
            for atom in range(len(side_labels)):
                generator_atoms.append({
                    "dataset": dataset, "component_id": component_id, "pair_side": side, "row_id": row_id,
                    "atom_index": atom, "changed_atom": bool(side_labels[atom]),
                    **{name: float(atom_arrays[row_id][name][atom]) for name in baseline_names},
                })
        concatenated = {name: np.concatenate([atom_arrays[left_id][name], atom_arrays[right_id][name]]) for name in baseline_names}
        for name, scores_array in concatenated.items():
            ap, prevalence, nap = normalized_ap(labels, scores_array)
            generator_pairs.append({
                "dataset": dataset, "component_id": component_id, "baseline": name,
                "average_precision": ap, "prevalence": prevalence, "normalized_ap_lift": nap,
            })
        changed = labels
        unchanged = ~labels
        support_rows.append({
            "dataset": dataset, "component_id": component_id, "changed_atoms": int(changed.sum()), "unchanged_atoms": int(unchanged.sum()),
            "changed_support_fraction": float(concatenated["support_any"][changed].mean()),
            "unchanged_support_fraction": float(concatenated["support_any"][unchanged].mean()),
            "changed_minus_unchanged_support": float(concatenated["support_any"][changed].mean() - concatenated["support_any"][unchanged].mean()),
            "changed_mean_local_supported_mutants": float(concatenated["local_supported_mutant_count"][changed].mean()),
            "unchanged_mean_local_supported_mutants": float(concatenated["local_supported_mutant_count"][unchanged].mean()),
        })
        random_rng = rng_for(seed, f"random-atoms:{dataset}:{component_id}")
        random_ap = np.array([average_precision_score(labels, random_rng.random(len(labels))) for _ in range(random_replicates)])
        prevalence = float(labels.mean())
        random_nap = (random_ap - prevalence) / (1.0 - prevalence)
        random_rows.append({
            "dataset": dataset, "component_id": component_id, "atoms": len(labels), "positive_atoms": int(labels.sum()),
            "prevalence": prevalence, "replicates": random_replicates,
            "mean_random_ap": float(random_ap.mean()), "median_random_ap": float(np.median(random_ap)),
            "random_ap_ci95_low": interval(random_ap)[0], "random_ap_ci95_high": interval(random_ap)[1],
            "mean_random_nap": float(random_nap.mean()), "median_random_nap": float(np.median(random_nap)),
            "random_nap_ci95_low": interval(random_nap)[0], "random_nap_ci95_high": interval(random_nap)[1],
        })
    generator_atom_frame = pd.DataFrame(generator_atoms)
    generator_pair_frame = pd.DataFrame(generator_pairs)
    support_frame = pd.DataFrame(support_rows)
    random_frame = pd.DataFrame(random_rows)
    generator_atom_frame.to_csv(output / "generator_atom_geometry.csv", index=False)
    generator_pair_frame.to_csv(output / "generator_baseline_pair_metrics.csv", index=False)
    support_frame.to_csv(output / "generator_changed_unchanged_support.csv", index=False)
    random_frame.to_csv(output / "random_atom_baseline.csv", index=False)
    generator_summary_rows = []
    for baseline, frame in generator_pair_frame.groupby("baseline", sort=True):
        target_values = frame.groupby("dataset", as_index=False).agg(median_ap=("average_precision", "median"), median_nap=("normalized_ap_lift", "median"), median_prevalence=("prevalence", "median"))
        distribution = two_stage_bootstrap(frame, "normalized_ap_lift", draws, seed, "generator:" + baseline)
        ci = interval(distribution)
        generator_summary_rows.append({
            "baseline": baseline, "target_balanced_median_ap": float(target_values.median_ap.median()),
            "target_balanced_median_nap": float(target_values.median_nap.median()),
            "target_balanced_median_prevalence": float(target_values.median_prevalence.median()),
            "nap_ci95_low": ci[0], "nap_ci95_high": ci[1], "targets": frame.dataset.nunique(), "pairs": len(frame),
        })
    pd.DataFrame(generator_summary_rows).to_csv(output / "generator_baseline_summary.csv", index=False)
    generator_support_target = support_frame.groupby("dataset", as_index=False).agg(
        median_changed_support=("changed_support_fraction", "median"),
        median_unchanged_support=("unchanged_support_fraction", "median"),
        median_support_gap=("changed_minus_unchanged_support", "median"),
        pairs=("component_id", "nunique"),
    )
    generator_support_target.to_csv(output / "generator_changed_unchanged_target_summary.csv", index=False)
    support_distribution = median_bootstrap(generator_support_target.median_support_gap, draws, seed, "generator-support-gap")
    support_ci = interval(support_distribution)
    pd.DataFrame([{
        "target_balanced_median_changed_support": float(generator_support_target.median_changed_support.median()),
        "target_balanced_median_unchanged_support": float(generator_support_target.median_unchanged_support.median()),
        "target_balanced_median_support_gap": float(generator_support_target.median_support_gap.median()),
        "support_gap_ci95_low": support_ci[0], "support_gap_ci95_high": support_ci[1],
        "positive_target_medians": int((generator_support_target.median_support_gap > 0).sum()),
        "targets": len(generator_support_target),
    }]).to_csv(output / "generator_changed_unchanged_summary.csv", index=False)

    contrast_rows = []
    for target, frame in target_config.groupby("dataset", sort=True):
        values = frame.set_index(config_keys).median_delta_nap.to_dict()
        if len(values) != len(representations) * len(predictors) * len(explainers):
            raise ValueError(f"incomplete target factorial table: {target}")
        def y(rep, predictor, explainer):
            return float(values[(rep, predictor, explainer)])
        for rep in [value for value in representations if value != "morgan"]:
            rp = np.mean([(y(rep, "mlp", ex) - y(rep, "xgboost", ex)) - (y("morgan", "mlp", ex) - y("morgan", "xgboost", ex)) for ex in explainers])
            re = np.mean([(y(rep, pr, "crem_lime") - y(rep, pr, "crem_mean")) - (y("morgan", pr, "crem_lime") - y("morgan", pr, "crem_mean")) for pr in predictors])
            rpe = ((y(rep, "mlp", "crem_lime") - y(rep, "mlp", "crem_mean")) - (y(rep, "xgboost", "crem_lime") - y(rep, "xgboost", "crem_mean"))) - ((y("morgan", "mlp", "crem_lime") - y("morgan", "mlp", "crem_mean")) - (y("morgan", "xgboost", "crem_lime") - y("morgan", "xgboost", "crem_mean")))
            contrast_rows.extend([
                {"dataset": target, "interaction": "representation_by_predictor", "contrast": f"{rep}_vs_morgan", "estimate": rp},
                {"dataset": target, "interaction": "representation_by_explainer", "contrast": f"{rep}_vs_morgan", "estimate": re},
                {"dataset": target, "interaction": "three_way", "contrast": f"{rep}_vs_morgan", "estimate": rpe},
            ])
        pe = np.mean([(y(rep, "mlp", "crem_lime") - y(rep, "mlp", "crem_mean")) - (y(rep, "xgboost", "crem_lime") - y(rep, "xgboost", "crem_mean")) for rep in representations])
        contrast_rows.append({"dataset": target, "interaction": "predictor_by_explainer", "contrast": "average_over_representations", "estimate": pe})
    contrast_frame = pd.DataFrame(contrast_rows)
    contrast_frame.to_csv(output / "target_equal_factorial_contrasts.csv", index=False)
    contrast_summary_rows = []
    for key, frame in contrast_frame.groupby(["interaction", "contrast"], sort=True):
        distribution = mean_bootstrap(frame.estimate, draws, seed, "contrast:" + ":".join(key))
        ci = interval(distribution)
        contrast_summary_rows.append({
            "interaction": key[0], "contrast": key[1], "target_equal_mean": float(frame.estimate.mean()),
            "target_equal_median": float(frame.estimate.median()), "ci95_low": ci[0], "ci95_high": ci[1],
            "absolute_mean_over_margin_0_05": float(abs(frame.estimate.mean()) / 0.05), "target_clusters": len(frame),
        })
    pd.DataFrame(contrast_summary_rows).to_csv(output / "target_equal_factorial_summary.csv", index=False)

    min_components = target_primary.sort_values(["components", "dataset"]).iloc[0].dataset
    max_components = target_primary.sort_values(["components", "dataset"], ascending=[False, True]).iloc[0].dataset
    min_delta = target_primary.sort_values(["median_delta_nap", "dataset"]).iloc[0].dataset
    max_delta = target_primary.sort_values(["median_delta_nap", "dataset"], ascending=[False, True]).iloc[0].dataset
    reason_map = {}
    for target, reason in [(min_components, "fewest_components"), (max_components, "most_components"), (min_delta, "lowest_target_delta"), (max_delta, "highest_target_delta")]:
        reason_map.setdefault(target, []).append(reason)
    loto_rows = []
    for target in sorted(target_primary.dataset):
        retained = target_primary[target_primary.dataset != target]
        loto_rows.append({
            "excluded_target": target, "reason": ";".join(reason_map.get(target, [])),
            "retained_targets": len(retained), "target_balanced_median_delta": float(retained.median_delta_nap.median()),
        })
    pd.DataFrame(loto_rows).to_csv(output / "leave_one_target_out.csv", index=False)
    extreme_rows = []
    for target, reasons in sorted(reason_map.items()):
        retained = component[component.dataset != target]
        distribution = two_stage_bootstrap(retained, "delta_nap", draws, seed, "extreme:" + target)
        ci = interval(distribution)
        extreme_rows.append({
            "excluded_target": target, "reason": ";".join(reasons), "retained_targets": retained.dataset.nunique(),
            "target_balanced_median_delta": float(target_primary[target_primary.dataset != target].median_delta_nap.median()),
            "ci95_low": ci[0], "ci95_high": ci[1],
        })
    pd.DataFrame(extreme_rows).to_csv(output / "extreme_target_sensitivity.csv", index=False)

    fidelity["variable_bits_per_local_sample"] = fidelity.variable_bits / fidelity.local_samples
    fidelity["underdetermined_by_raw_dimension"] = fidelity.variable_bits >= fidelity.local_samples
    fidelity_summary = fidelity.groupby(["variant", "representation", "predictor"], as_index=False).agg(
        model_neighborhood_rows=("row_id", "size"), unique_endpoints=("row_id", "nunique"),
        median_in_sample_weighted_r2=("local_weighted_r2", "median"),
        q05_in_sample_weighted_r2=("local_weighted_r2", lambda x: quantile(x, 0.05)),
        q95_in_sample_weighted_r2=("local_weighted_r2", lambda x: quantile(x, 0.95)),
        median_local_samples=("local_samples", "median"), min_local_samples=("local_samples", "min"), max_local_samples=("local_samples", "max"),
        median_variable_bits=("variable_bits", "median"), min_variable_bits=("variable_bits", "min"), max_variable_bits=("variable_bits", "max"),
        median_variable_bits_per_sample=("variable_bits_per_local_sample", "median"),
        fraction_raw_dimension_ge_samples=("underdetermined_by_raw_dimension", "mean"),
        median_supported_atom_fraction=("supported_atom_fraction", "median"),
    )
    fidelity_summary.to_csv(output / "local_fidelity_descriptives.csv", index=False)
    fidelity_target = fidelity.groupby(["dataset", "variant"], as_index=False).agg(
        median_in_sample_weighted_r2=("local_weighted_r2", "median"),
        median_local_samples=("local_samples", "median"), median_variable_bits=("variable_bits", "median"),
        median_variable_bits_per_sample=("variable_bits_per_local_sample", "median"),
        fraction_raw_dimension_ge_samples=("underdetermined_by_raw_dimension", "mean"),
    )
    fidelity_target.to_csv(output / "local_fidelity_target_summary.csv", index=False)

    prevalence_pairs = exp[["dataset", "component_id", "left_index", "right_index", "prevalence"]].drop_duplicates()
    if prevalence_pairs.duplicated(["dataset", "component_id"]).any():
        raise ValueError("prevalence is not invariant across configurations")
    prevalence_summary = prevalence_pairs.groupby("dataset", as_index=False).agg(
        pairs=("component_id", "nunique"), minimum=("prevalence", "min"), q25=("prevalence", lambda x: quantile(x, 0.25)),
        median=("prevalence", "median"), q75=("prevalence", lambda x: quantile(x, 0.75)), maximum=("prevalence", "max"), mean=("prevalence", "mean"),
    )
    prevalence_summary.to_csv(output / "prevalence_distribution.csv", index=False)
    bins = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
    prevalence_pairs["bin"] = pd.cut(prevalence_pairs.prevalence, bins=bins, include_lowest=True, right=True).astype(str)
    prevalence_pairs.groupby("bin", observed=False).size().rename("pairs").reset_index().to_csv(output / "prevalence_histogram.csv", index=False)

    target_rows = []
    valid_audit = rationale_audit[rationale_audit.valid_normalized_ap_prevalence.astype(bool)]
    base_perm_seed = int(source_cfg["randomization"]["label_permutation_seed"])
    for target_index, target in enumerate(source_cfg["targets"]):
        target_sentinel = sentinels[sentinels.dataset == target]
        target_audit = valid_audit[valid_audit.dataset == target]
        target_rows.append({
            "target_index_zero_based": target_index, "dataset": target,
            "role": "development" if target in source_cfg["development_targets"] else "confirmation",
            "molecules": int((molecules.dataset == target).sum()), "shared_model_molecules": int((model_molecules.dataset == target).sum()),
            "sentinel_pairs": len(target_sentinel), "calibration_sentinel_pairs": int((target_sentinel.component_split == "calibration").sum()),
            "test_sentinel_pairs": int((target_sentinel.component_split == "test").sum()),
            "normalized_ap_eligible_pairs": len(target_audit),
            "normalized_ap_eligible_test_pairs": int((target_audit.component_split == "test").sum()),
            **{f"label_permutation_seed_replicate_{replicate}": base_perm_seed + 1000 * target_index + int(replicate) for replicate in seeds},
        })
    target_map = pd.DataFrame(target_rows)
    target_map.to_csv(output / "target_index_map.csv", index=False)
    flow = pd.DataFrame([
        {"stage": "all_targets", "count": len(source_cfg["targets"]), "unit": "targets"},
        {"stage": "confirmation_targets", "count": len(confirmation), "unit": "targets"},
        {"stage": "source_molecules", "count": len(molecules), "unit": "molecules"},
        {"stage": "shared_model_molecules", "count": len(model_molecules), "unit": "molecules"},
        {"stage": "all_sentinel_pairs", "count": len(sentinels), "unit": "pairs"},
        {"stage": "all_test_sentinel_pairs", "count": int((sentinels.component_split == "test").sum()), "unit": "pairs"},
        {"stage": "confirmation_test_sentinel_pairs", "count": int(((sentinels.dataset.isin(confirmation_set)) & (sentinels.component_split == "test")).sum()), "unit": "pairs"},
        {"stage": "confirmation_test_normalized_ap_eligible_pairs", "count": len(eligible_pairs), "unit": "pairs"},
        {"stage": "confirmation_test_evaluation_endpoints", "count": len(eligible_endpoints), "unit": "molecules"},
        {"stage": "confirmation_test_explanation_rows", "count": len(exp), "unit": "model_explainer_pair_rows"},
    ])
    flow.to_csv(output / "eligibility_flow.csv", index=False)

    generator_atom_class = generator_atom_frame.groupby("changed_atom", as_index=False).agg(
        atoms=("atom_index", "size"), support_fraction=("support_any", "mean"),
        mean_local_supported_mutants=("local_supported_mutant_count", "mean"),
        mean_selected_parent_tanimoto=("selected_mean_parent_tanimoto", "mean"),
    )
    generator_atom_class.to_csv(output / "generator_atom_class_summary.csv", index=False)

    random_target = random_frame.groupby("dataset", as_index=False).agg(
        median_prevalence=("prevalence", "median"), median_empirical_random_ap=("median_random_ap", "median"),
        median_empirical_random_nap=("median_random_nap", "median"), pairs=("component_id", "nunique"),
    )
    random_target.to_csv(output / "random_atom_target_summary.csv", index=False)

    prediction_variant_target = model_cell.groupby(["dataset", "variant"], as_index=False).agg(
        spearman=("spearman", "median"), direction_accuracy=("direction_accuracy", "median"),
        delta_mae=("delta_mae", "median"), delta_rmse=("delta_rmse", "median"),
        calibration_slope=("calibration_slope", "median"), calibration_intercept=("calibration_intercept", "median"),
    )
    prediction_overall = prediction_variant_target.groupby("variant", as_index=False).agg(
        target_balanced_median_spearman=("spearman", "median"),
        target_balanced_median_direction_accuracy=("direction_accuracy", "median"),
        target_balanced_median_delta_mae=("delta_mae", "median"),
        target_balanced_median_delta_rmse=("delta_rmse", "median"),
        target_balanced_median_calibration_slope=("calibration_slope", "median"),
        target_balanced_median_calibration_intercept=("calibration_intercept", "median"),
    )
    prediction_overall.to_csv(output / "prediction_overall_summary.csv", index=False)

    summary = {
        "run_id": cfg["run_id"], "status": "success",
        "confirmation_targets": len(confirmation), "eligible_confirmation_test_pairs": len(eligible_pairs),
        "symmetric_primary": {
            "target_balanced_median_delta_nap": float(target_primary.median_delta_nap.median()),
            "ci90": list(primary_ci90), "ci95": list(primary_ci95),
            "frozen_margin": [-float(source_cfg["inference"]["equivalence_margin"]), float(source_cfg["inference"]["equivalence_margin"])],
            "equivalent_at_ci95": primary_ci95[0] >= -float(source_cfg["inference"]["equivalence_margin"]) and primary_ci95[1] <= float(source_cfg["inference"]["equivalence_margin"]),
        },
        "generator_mapping": {
            "status": "pass", "eligible_endpoints": len(eligible_endpoints), "mapped_mask_rows": len(valid_masks),
            "mapped_selected_mutant_masks": len(scored_masks), "atom_rows": len(generator_atom_frame),
        },
        "empirical_random_atom_baseline": {
            "replicates_per_pair": random_replicates,
            "target_balanced_median_ap": float(random_target.median_empirical_random_ap.median()),
            "target_balanced_median_nap": float(random_target.median_empirical_random_nap.median()),
        },
        "pairwise_prediction": {
            "model_cells": len(model_cell),
            "undefined_spearman_constant_predicted_delta_cells": int(model_cell.spearman.isna().sum()),
            "undefined_spearman_label_permuted_cells": int(model_cell[model_cell.variant == "label_permuted"].spearman.isna().sum()),
        },
        "generator_support": {
            "target_balanced_median_changed_support": float(generator_support_target.median_changed_support.median()),
            "target_balanced_median_unchanged_support": float(generator_support_target.median_unchanged_support.median()),
            "target_balanced_median_changed_minus_unchanged": float(generator_support_target.median_support_gap.median()),
            "target_bootstrap_ci95": list(support_ci),
        },
        "local_fidelity_scope": {
            "reported_metric": "in-sample distance-weighted R2",
            "effective_degrees_of_freedom": "not identifiable from saved summary rows",
            "held-out_or_cross_validated_r2": "not computed by this existing-output reanalysis",
        },
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
        "source_config_sha256": sha256(source_cfg_path), "analysis_config_sha256": sha256(args.config),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    dictionary = """# Robustness reanalysis data dictionary

All results are derived from the frozen `xai_sanity_cliffs_full_v1` row-level outputs. Targets are weighted equally unless a file explicitly reports pair-level rows. The revised primary estimand averages the three trained seeds and the three label-permuted seeds separately within each target/component/configuration, subtracts the two means, takes a median within each target, and then takes the median across confirmation targets. Its interval uses a target-and-component two-stage percentile bootstrap.

- `symmetric_component_metrics.csv`: seed-averaged absolute AP, normalized AP (nAP), prevalence, and trained-minus-permuted differences for every eligible confirmation component and 12-way configuration.
- `target_configuration_absolute_metrics.csv`: target/configuration medians plus 90% and 95% component-bootstrap intervals for the nAP difference. Cell equivalence is nominal because it is a descriptive scope-control result, not the pooled primary decision.
- `configuration_absolute_metrics.csv`: the requested 12-row comparison of absolute trained/permuted AP and nAP, prevalence, target-balanced nAP difference, two-stage intervals, and nominally equivalent target counts.
- `margin_and_ci_sensitivity.csv`: frozen and non-primary margin/CI combinations. Non-frozen margins are sensitivity descriptions, not alternative decision rules.
- `prediction_*`: pairwise cliff prediction Spearman correlation, direction accuracy, delta MAE/RMSE, and linear calibration summaries. Calibration regresses predicted signed delta on observed signed delta. Seed/configuration summaries use medians before the final equal-target median; `prediction_randomization_summary.csv` gives target-bootstrap intervals for trained-versus-permuted changes. Spearman is left missing, rather than imputed, for the eight label-permuted model cells whose predicted pair deltas are constant.
- `prediction_aligned_*`: the original trained-seed-minus-mean-permuted-seed estimand after selecting trained models that predict the cliff direction and, where stated, a minimum absolute predicted delta. All sentinel pairs already have absolute observed delta above 1 log unit.
- `generator_atom_geometry.csv`: model-free atom scores reconstructed exactly from `mask_coverage.csv` and the deterministically selected rows in `scored_mutants.csv`. Mask-level quantities are averaged over masks containing an atom; `support_any` is an OR and `mask_membership_count` is a count. Unmaskable atoms are zero. This is a generator-geometry baseline, not an explanation.
- `generator_baseline_*`: AP/nAP of those model-free scores against the frozen structural-change rationale. `generator_changed_unchanged_*` reports coverage separately for rationale-positive and rationale-negative atoms.
- `random_atom_baseline.csv`: 100 deterministic random atom rankings per pair. The analytical random-ranking reference is AP=prevalence and nAP=0; finite-pair empirical AP can deviate from prevalence.
- `target_equal_factorial_*`: within-target difference-in-differences on target/configuration median nAP differences. Reported uncertainty is a 27-target cluster bootstrap; there are no asymptotic p-value claims.
- `leave_one_target_out.csv` and `extreme_target_sensitivity.csv`: target-balanced influence and prespecified smallest/largest-information and lowest/highest-effect exclusions.
- `local_fidelity_*`: descriptive in-sample distance-weighted R2, local sample count, variable-bit count, and raw dimension/sample ratio for all 619 confirmation-test pair endpoints (including the two pairs excluded only from normalized AP). Saved rows do not identify ridge effective degrees of freedom and do not support held-out/CV fidelity; this reanalysis does not relabel the in-sample statistic.
- `eligibility_flow.csv` and `target_index_map.csv`: analysis flow, target role, zero-based target index, and the exact permutation seeds implied by the frozen formula.
- `prevalence_*`: rationale-positive atom prevalence without configuration duplication.

Generator mapping is accepted only if every eligible endpoint has mask coverage, all atom indices are within the recorded parent atom count, and every selected-mutant mask maps exactly to one mask-coverage row. Any failure aborts the script rather than inferring an atom mapping.
"""
    (output / "DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")

    output_files = sorted(path for path in output.iterdir() if path.is_file() and path.name != "manifest.json")
    manifest = {
        "run_id": cfg["run_id"], "status": "success",
        "command": "python research_v2/scripts/analyze_major_revision.py --config research_v2/configs/major_revision_v1.json",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "script_sha256": sha256(Path(__file__)), "config_sha256": sha256(args.config),
        "input_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths.values()},
        "output_hashes": {path.name: sha256(path) for path in output_files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
