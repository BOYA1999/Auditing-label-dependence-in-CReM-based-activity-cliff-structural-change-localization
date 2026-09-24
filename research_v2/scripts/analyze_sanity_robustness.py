import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def two_stage_target_bootstrap(frame, value, draws, seed):
    groups = {
        target: [group[value].to_numpy() for _, group in target_frame.groupby("component_id")]
        for target, target_frame in frame.groupby("dataset")
    }
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        target_values = []
        for target_index in rng.integers(0, len(names), len(names)):
            components = groups[names[target_index]]
            sampled = [components[index] for index in rng.integers(0, len(components), len(components))]
            target_values.append(np.median(np.concatenate(sampled)))
        estimates[draw] = np.median(target_values)
    return estimates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    confirmation = set(cfg["confirmation_targets"])
    draws = int(cfg["inference"]["bootstrap_draws"])
    seed = int(cfg["inference"]["bootstrap_seed"])

    metrics = pd.read_csv(args.heads / "prediction_metrics.csv")
    metrics = metrics[(metrics.dataset.isin(confirmation)) & (metrics.split == "test")]
    keys = ["dataset", "representation", "predictor", "seed"]
    prediction = metrics.pivot(index=keys, columns="variant", values="rmse_against_observed").reset_index()
    prediction["permuted_minus_trained_rmse"] = prediction.label_permuted - prediction.trained
    prediction.to_csv(args.output / "prediction_randomization_control.csv", index=False)
    target_prediction = prediction.groupby("dataset", as_index=False).permuted_minus_trained_rmse.median()
    target_prediction.to_csv(args.output / "prediction_randomization_target_estimates.csv", index=False)
    rng = np.random.default_rng(seed)
    values = target_prediction.permuted_minus_trained_rmse.to_numpy()
    prediction_bootstrap = np.median(values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1)
    prediction_by_configuration = prediction.groupby(["representation", "predictor"], as_index=False).agg(
        median_rmse_increase=("permuted_minus_trained_rmse", "median"),
        positive_models=("permuted_minus_trained_rmse", lambda x: int((x > 0).sum())),
        models=("permuted_minus_trained_rmse", "size"),
    )
    prediction_by_configuration.to_csv(args.output / "prediction_randomization_configuration_summary.csv", index=False)

    pairs = pd.read_csv(args.scores / "pair_explanations.csv")
    pairs = pairs[(pairs.dataset.isin(confirmation)) & (pairs.component_split == "test")]
    explanation_keys = ["dataset", "component_id", "representation", "predictor", "explainer", "seed"]
    matched = pairs.pivot(index=explanation_keys, columns="variant", values="normalized_ap_lift").reset_index()
    matched["matched_seed_delta"] = matched.trained - matched.label_permuted
    matched.to_csv(args.output / "matched_seed_component_deltas.csv", index=False)
    target_matched = matched.groupby("dataset", as_index=False).matched_seed_delta.median()
    target_matched.to_csv(args.output / "matched_seed_target_estimates.csv", index=False)
    matched_bootstrap = two_stage_target_bootstrap(matched, "matched_seed_delta", draws, seed)

    fidelity = pd.read_csv(args.scores / "local_fidelity.csv")
    fidelity = fidelity[(fidelity.dataset.isin(confirmation)) & (fidelity.component_split == "test")]
    fidelity_keys = ["dataset", "row_id", "representation", "predictor", "seed"]
    fidelity_pairs = fidelity.pivot(index=fidelity_keys, columns="variant", values="local_weighted_r2").reset_index()
    fidelity_pairs["trained_minus_permuted_r2"] = fidelity_pairs.trained - fidelity_pairs.label_permuted
    fidelity_pairs.to_csv(args.output / "local_fidelity_randomization_control.csv", index=False)
    fidelity_configuration = fidelity_pairs.groupby(["representation", "predictor"], as_index=False).agg(
        permuted_median_r2=("label_permuted", "median"), trained_median_r2=("trained", "median"),
        median_r2_difference=("trained_minus_permuted_r2", "median"),
        permuted_fraction_ge_08=("label_permuted", lambda x: float((x >= 0.8).mean())),
    )
    fidelity_configuration.to_csv(args.output / "local_fidelity_configuration_summary.csv", index=False)
    fidelity_targets = fidelity_pairs.groupby("dataset", as_index=False).agg(
        permuted_median_r2=("label_permuted", "median"),
        trained_median_r2=("trained", "median"),
        median_r2_difference=("trained_minus_permuted_r2", "median"),
    )
    fidelity_targets.to_csv(args.output / "local_fidelity_target_summary.csv", index=False)
    fidelity_values = fidelity_targets.permuted_median_r2.to_numpy()
    fidelity_rng = np.random.default_rng(seed)
    fidelity_bootstrap = np.median(
        fidelity_values[fidelity_rng.integers(0, len(fidelity_values), size=(draws, len(fidelity_values)))], axis=1,
    )

    similarity = pd.read_csv(args.scores / "atom_rank_similarity.csv")
    similarity = similarity[(similarity.dataset.isin(confirmation)) & (similarity.comparison == "random_mean")]
    similarity_targets = similarity.groupby("dataset", as_index=False).agg(
        supported_spearman=("spearman_supported", "median"),
        all_atom_spearman=("spearman_all_atoms", "median"),
    ).dropna()
    similarity_targets["all_minus_supported"] = (
        similarity_targets.all_atom_spearman - similarity_targets.supported_spearman
    )
    similarity_targets.to_csv(args.output / "atom_similarity_support_bias.csv", index=False)
    similarity_values = similarity_targets.all_minus_supported.to_numpy()
    similarity_rng = np.random.default_rng(seed)
    similarity_bootstrap = np.median(
        similarity_values[
            similarity_rng.integers(0, len(similarity_values), size=(draws, len(similarity_values)))
        ], axis=1,
    )

    summary = {
        "status": "success", "confirmation_targets": len(confirmation),
        "prediction_randomization": {
            "target_balanced_median_permuted_minus_trained_rmse": float(np.median(values)),
            "target_bootstrap_ci": interval(prediction_bootstrap),
            "positive_target_medians": int((values > 0).sum()), "targets": len(values),
            "positive_model_cells": int((prediction.permuted_minus_trained_rmse > 0).sum()),
            "model_cells": len(prediction),
        },
        "matched_seed_explanation": {
            "target_balanced_median_delta": float(np.median(target_matched.matched_seed_delta)),
            "two_stage_bootstrap_ci": interval(matched_bootstrap),
        },
        "local_fidelity_randomization": {
            "target_balanced_permuted_median_r2": float(np.median(fidelity_values)),
            "target_bootstrap_ci": interval(fidelity_bootstrap),
            "pooled_permuted_fraction_ge_08": float((fidelity_pairs.label_permuted >= 0.8).mean()),
            "pooled_trained_fraction_ge_08": float((fidelity_pairs.trained >= 0.8).mean()),
            "pooled_median_trained_minus_permuted_r2": float(fidelity_pairs.trained_minus_permuted_r2.median()),
        },
        "atom_similarity_support_bias": {
            "target_balanced_median_all_minus_supported_spearman": float(np.median(similarity_values)),
            "target_bootstrap_ci": interval(similarity_bootstrap),
            "targets": len(similarity_values),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
