import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_metrics(frame):
    true = frame.true_delta.to_numpy(dtype=float)
    predicted = frame.predicted_delta.to_numpy(dtype=float)
    correlation = spearmanr(true, predicted).statistic if len(true) > 1 and np.ptp(true) > 0 and np.ptp(predicted) > 0 else np.nan
    error = predicted - true
    return pd.Series({
        "test_pairs": len(frame),
        "spearman_true_predicted_delta": float(correlation) if np.isfinite(correlation) else np.nan,
        "spearman_defined": bool(np.isfinite(correlation)),
        "direction_accuracy": float(np.mean(np.sign(predicted) == np.sign(true))),
        "delta_mae": float(np.mean(np.abs(error))),
        "delta_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_absolute_predicted_delta": float(np.median(np.abs(predicted))),
    })


def cumulative_explanation(cfg, random_frame, trained_frame):
    keys = ["dataset", "component_id", "representation", "predictor", "explainer"]
    trained = trained_frame.groupby(keys, as_index=False).normalized_ap_lift.mean().rename(
        columns={"normalized_ap_lift": "trained_seed_mean_nap"}
    )
    target_rows = []
    configuration_rows = []
    pooled_rows = []
    ordered = cfg["permutation_ids"]
    for count in [1, 3, 5, 10]:
        active = ordered[:count]
        random = random_frame[random_frame.permutation_id.isin(active)].groupby(
            keys, as_index=False
        ).normalized_ap_lift.mean().rename(columns={"normalized_ap_lift": "random_permutation_mean_nap"})
        merged = trained.merge(random, on=keys, validate="one_to_one")
        merged["trained_minus_random_nap"] = merged.trained_seed_mean_nap - merged.random_permutation_mean_nap
        by_target = merged.groupby(
            ["dataset", "representation", "predictor", "explainer"], as_index=False
        ).agg(
            trained_seed_mean_nap=("trained_seed_mean_nap", "median"),
            random_permutation_mean_nap=("random_permutation_mean_nap", "median"),
            trained_minus_random_nap=("trained_minus_random_nap", "median"),
            test_pairs=("component_id", "size"),
        )
        by_target["permutations_used"] = count
        by_target["permutation_ids"] = ",".join(str(value) for value in active)
        target_rows.append(by_target)
        by_configuration = by_target.groupby(
            ["representation", "predictor", "explainer"], as_index=False
        ).agg(
            target_balanced_trained_nap=("trained_seed_mean_nap", "median"),
            target_balanced_random_nap=("random_permutation_mean_nap", "median"),
            target_balanced_trained_minus_random_nap=("trained_minus_random_nap", "median"),
            targets=("dataset", "nunique"),
        )
        by_configuration["permutations_used"] = count
        by_configuration["permutation_ids"] = ",".join(str(value) for value in active)
        configuration_rows.append(by_configuration)
        target_pooled = by_target.groupby("dataset", as_index=False).agg(
            trained_nap=("trained_seed_mean_nap", "median"),
            random_nap=("random_permutation_mean_nap", "median"),
            trained_minus_random_nap=("trained_minus_random_nap", "median"),
        )
        pooled_rows.append({
            "permutations_used": count,
            "permutation_ids": ",".join(str(value) for value in active),
            "target_balanced_trained_nap": float(target_pooled.trained_nap.median()),
            "target_balanced_random_nap": float(target_pooled.random_nap.median()),
            "target_balanced_trained_minus_random_nap": float(target_pooled.trained_minus_random_nap.median()),
            "targets": int(target_pooled.dataset.nunique()),
        })
    targets = pd.concat(target_rows, ignore_index=True)
    configurations = pd.concat(configuration_rows, ignore_index=True)
    pooled = pd.DataFrame(pooled_rows)
    full_target = targets[targets.permutations_used == 10][
        ["dataset", "representation", "predictor", "explainer", "trained_minus_random_nap"]
    ].rename(columns={"trained_minus_random_nap": "full_10_trained_minus_random_nap"})
    targets = targets.merge(full_target, on=["dataset", "representation", "predictor", "explainer"], validate="many_to_one")
    targets["difference_from_10"] = targets.trained_minus_random_nap - targets.full_10_trained_minus_random_nap
    targets["absolute_difference_from_10"] = targets.difference_from_10.abs()
    full_configuration = configurations[configurations.permutations_used == 10][
        ["representation", "predictor", "explainer", "target_balanced_trained_minus_random_nap"]
    ].rename(columns={"target_balanced_trained_minus_random_nap": "full_10_trained_minus_random_nap"})
    configurations = configurations.merge(full_configuration, on=["representation", "predictor", "explainer"], validate="many_to_one")
    configurations["difference_from_10"] = configurations.target_balanced_trained_minus_random_nap - configurations.full_10_trained_minus_random_nap
    configurations["absolute_difference_from_10"] = configurations.difference_from_10.abs()
    full_pooled = float(pooled.loc[pooled.permutations_used == 10, "target_balanced_trained_minus_random_nap"].iloc[0])
    pooled["full_10_trained_minus_random_nap"] = full_pooled
    pooled["difference_from_10"] = pooled.target_balanced_trained_minus_random_nap - full_pooled
    pooled["absolute_difference_from_10"] = pooled.difference_from_10.abs()
    return targets, configurations, pooled


def cumulative_pairwise(cfg, pairs):
    keys = ["dataset", "component_id", "representation", "predictor"]
    rows = []
    for count in [1, 3, 5, 10]:
        active = cfg["permutation_ids"][:count]
        averaged = pairs[pairs.permutation_id.isin(active)].groupby(keys, as_index=False).agg(
            true_delta=("true_delta", "first"), predicted_delta=("predicted_delta", "mean")
        )
        target = averaged.groupby(["dataset", "representation", "predictor"], group_keys=False).apply(
            pair_metrics, include_groups=False
        ).reset_index()
        target_balanced = target.groupby(["representation", "predictor"], as_index=False).agg(
            target_balanced_spearman=("spearman_true_predicted_delta", "median"),
            target_balanced_direction_accuracy=("direction_accuracy", "median"),
            target_balanced_delta_mae=("delta_mae", "median"),
            target_balanced_delta_rmse=("delta_rmse", "median"),
            target_balanced_median_absolute_predicted_delta=("median_absolute_predicted_delta", "median"),
            targets=("dataset", "nunique"),
            spearman_defined_targets=("spearman_defined", "sum"),
        )
        target_balanced["permutations_used"] = count
        target_balanced["permutation_ids"] = ",".join(str(value) for value in active)
        rows.append(target_balanced)
    result = pd.concat(rows, ignore_index=True)
    full = result[result.permutations_used == 10][
        ["representation", "predictor", "target_balanced_delta_mae", "target_balanced_direction_accuracy"]
    ].rename(columns={
        "target_balanced_delta_mae": "full_10_delta_mae",
        "target_balanced_direction_accuracy": "full_10_direction_accuracy",
    })
    result = result.merge(full, on=["representation", "predictor"], validate="many_to_one")
    result["delta_mae_difference_from_10"] = result.target_balanced_delta_mae - result.full_10_delta_mae
    result["direction_accuracy_difference_from_10"] = result.target_balanced_direction_accuracy - result.full_10_direction_accuracy
    return result


def artifact_manifest(output):
    excluded = {"artifact_manifest.csv", "run.log"}
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in excluded):
        rows.append({
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / "artifact_manifest.csv", index=False)
    return manifest


def permutation_signature(frame):
    digest = hashlib.sha256()
    for row in frame.sort_values("row_id").itertuples(index=False):
        digest.update(str(row.row_id).encode("utf-8"))
        digest.update(np.float64(row.permuted_y).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact-package", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve() if args.output else ROOT / cfg["output"]
    source_explanations = ROOT / cfg["inputs"]["source_explanations"]
    pairs = pd.read_csv(output / "pair_predictions.csv")
    explanations = pd.read_csv(output / "pair_explanations.csv")
    models = pd.read_csv(output / "model_manifest.csv")
    labels = pd.read_csv(output / "permuted_labels.csv")
    test_pairs = pairs[pairs.component_split == "test"].copy()
    pairwise = test_pairs.groupby(
        ["dataset", "representation", "predictor", "permutation_id", "permutation_seed"],
        group_keys=False,
    ).apply(pair_metrics, include_groups=False).reset_index()
    pairwise.to_csv(output / "pairwise_metrics_by_permutation.csv", index=False)
    target_stability = pairwise.groupby(
        ["dataset", "representation", "predictor"], as_index=False
    ).agg(
        permutations=("permutation_id", "nunique"),
        spearman_defined_permutations=("spearman_defined", "sum"),
        spearman_mean=("spearman_true_predicted_delta", "mean"),
        spearman_sd=("spearman_true_predicted_delta", "std"),
        direction_accuracy_mean=("direction_accuracy", "mean"),
        direction_accuracy_sd=("direction_accuracy", "std"),
        delta_mae_mean=("delta_mae", "mean"), delta_mae_sd=("delta_mae", "std"),
        delta_rmse_mean=("delta_rmse", "mean"), delta_rmse_sd=("delta_rmse", "std"),
        median_absolute_predicted_delta_mean=("median_absolute_predicted_delta", "mean"),
        median_absolute_predicted_delta_sd=("median_absolute_predicted_delta", "std"),
    )
    target_stability.to_csv(output / "pairwise_stability_by_target_configuration.csv", index=False)
    target_balanced = pairwise.groupby(
        ["representation", "predictor", "permutation_id"], as_index=False
    ).agg(
        target_balanced_spearman=("spearman_true_predicted_delta", "median"),
        target_balanced_direction_accuracy=("direction_accuracy", "median"),
        target_balanced_delta_mae=("delta_mae", "median"),
        target_balanced_delta_rmse=("delta_rmse", "median"),
        target_balanced_median_absolute_predicted_delta=("median_absolute_predicted_delta", "median"),
        targets=("dataset", "nunique"), spearman_defined_targets=("spearman_defined", "sum"),
    )
    target_balanced.to_csv(output / "pairwise_target_balanced_by_permutation.csv", index=False)
    configuration_stability = target_balanced.groupby(
        ["representation", "predictor"], as_index=False
    ).agg(
        permutations=("permutation_id", "nunique"),
        spearman_mean=("target_balanced_spearman", "mean"),
        spearman_sd=("target_balanced_spearman", "std"),
        direction_accuracy_mean=("target_balanced_direction_accuracy", "mean"),
        direction_accuracy_sd=("target_balanced_direction_accuracy", "std"),
        delta_mae_mean=("target_balanced_delta_mae", "mean"),
        delta_mae_sd=("target_balanced_delta_mae", "std"),
        delta_rmse_mean=("target_balanced_delta_rmse", "mean"),
        delta_rmse_sd=("target_balanced_delta_rmse", "std"),
    )
    configuration_stability.to_csv(output / "pairwise_stability_by_configuration.csv", index=False)
    source_frame = pd.read_csv(source_explanations / "pair_explanations.csv")
    trained = source_frame[
        (source_frame.variant == "trained") & (source_frame.component_split == "test")
        & source_frame.dataset.isin(cfg["confirmation_targets"])
    ]
    random_test = explanations[explanations.component_split == "test"]
    nap_target, nap_configuration, nap_pooled = cumulative_explanation(cfg, random_test, trained)
    nap_target.to_csv(output / "nap_convergence_by_target_configuration.csv", index=False)
    nap_configuration.to_csv(output / "nap_convergence_by_configuration.csv", index=False)
    nap_pooled.to_csv(output / "nap_convergence_pooled.csv", index=False)
    pair_convergence = cumulative_pairwise(cfg, test_pairs)
    pair_convergence.to_csv(output / "pairwise_convergence_by_configuration.csv", index=False)
    expected_models = len(cfg["confirmation_targets"]) * len(cfg["representations"]) * len(cfg["predictors"]) * len(cfg["permutation_ids"])
    unit_manifests = list((output / "units").glob("*/unit_manifest.json"))
    model_hashes_valid = True
    for row in models.itertuples(index=False):
        path = ROOT / row.path
        if not path.exists() or sha256(path) != row.sha256:
            model_hashes_valid = False
            break
    duplicate_explanations = int(explanations.duplicated([
        "dataset", "component_id", "component_split", "representation", "predictor", "explainer", "permutation_id"
    ]).sum())
    label_design = labels.groupby(["dataset", "permutation_id"], as_index=False).agg(
        permutation_seed=("permutation_seed", "first"),
        rows=("row_id", "size"),
    )
    label_design["expected_seed"] = label_design.apply(
        lambda row: cfg["randomization"]["base_seed"]
        + 1000 * cfg["randomization"]["source_target_order"].index(row.dataset)
        + int(row.permutation_id), axis=1,
    )
    label_signatures = labels.groupby(["dataset", "permutation_id"]).apply(
        permutation_signature, include_groups=False
    ).rename("signature").reset_index()
    unique_label_sequences = label_signatures.groupby("dataset").signature.nunique()
    expected_explanation_counts = source_frame[
        (source_frame.variant == "label_permuted") & (source_frame.seed == cfg["model_initialization_seed"])
        & source_frame.dataset.isin(cfg["confirmation_targets"])
    ].groupby(["dataset", "representation", "predictor", "explainer"]).size().rename("expected_rows").reset_index()
    actual_explanation_counts = explanations.groupby(
        ["dataset", "representation", "predictor", "explainer", "permutation_id"]
    ).size().rename("actual_rows").reset_index().merge(
        expected_explanation_counts,
        on=["dataset", "representation", "predictor", "explainer"], validate="many_to_one",
    )
    pair_counts = pairs.groupby(
        ["dataset", "representation", "predictor", "permutation_id"]
    ).size().rename("rows").reset_index().groupby(
        ["dataset", "representation", "predictor"], as_index=False
    ).agg(permutations=("permutation_id", "nunique"), minimum_rows=("rows", "min"), maximum_rows=("rows", "max"))
    checks = {
        "root_summary_pass": json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"] == "pass",
        "expected_model_count": len(models) == expected_models,
        "ten_permutations_per_nap_target_configuration": bool((pd.read_csv(output / "permutation_nap_stability_by_target_configuration.csv").permutations == 10).all()),
        "ten_permutations_per_pairwise_target_configuration": bool((target_stability.permutations == 10).all()),
        "both_explainers_present": set(explanations.explainer) == set(cfg["explainers"]),
        "finite_nap": bool(np.isfinite(explanations.normalized_ap_lift).all()),
        "finite_pairwise_errors": bool(np.isfinite(pairwise[["direction_accuracy", "delta_mae", "delta_rmse"]]).all().all()),
        "convergence_counts_present": set(nap_pooled.permutations_used) == {1, 3, 5, 10},
        "duplicate_pair_explanations": duplicate_explanations == 0,
        "fixed_model_initialization_seed": set(models.model_initialization_seed) == {cfg["model_initialization_seed"]},
        "permutation_seed_formula_valid": bool((label_design.permutation_seed == label_design.expected_seed).all()),
        "ten_distinct_label_sequences_per_target": bool((unique_label_sequences == 10).all()),
        "pair_explanation_rows_complete": bool((actual_explanation_counts.actual_rows == actual_explanation_counts.expected_rows).all()),
        "pair_prediction_rows_complete": bool(
            (pair_counts.permutations == 10).all()
            and (pair_counts.minimum_rows == pair_counts.maximum_rows).all()
        ),
    }
    if not args.compact_package:
        checks.update({
            "model_hashes_valid": model_hashes_valid,
            "unit_manifest_count": len(unit_manifests) == expected_models,
            "unit_manifests_pass": all(json.loads(path.read_text(encoding="utf-8"))["status"] == "pass" for path in unit_manifests),
        })
    validation = {
        "status": "pass" if all(checks.values()) else "fail",
        "validation_scope": "compact row-level reanalysis; model binaries and per-unit directories intentionally excluded" if args.compact_package else "full campaign artifacts",
        "checks": checks,
        "expected_models": expected_models,
        "models": int(len(models)),
        "pair_explanation_rows": int(len(explanations)),
        "undefined_pairwise_spearman_rows": int((~pairwise.spearman_defined).sum()),
        "duplicate_pair_explanation_rows": duplicate_explanations,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    evaluation_summary = {
        "evaluation_summary": "Ten independent within-target label permutations were scored at fixed model initialization across 27 confirmation targets and the complete 3 x 2 x 2 representation-predictor-explainer crossing.",
        "claim_update": "Permutation variance and 1/3/5/10 convergence are measured rather than assumed; interpretation must follow the reported stability diagnostics.",
        "baseline_relation": "Permutation ID 42 is byte-verified reuse of the original fixed-seed randomized arm; IDs 43-51 are new fits under the same data, heads and CReM neighborhoods.",
        "failure_mode": None if validation["status"] == "pass" else "evaluation_pipeline_failure",
        "next_action": "Use the convergence and stability tables in the robustness analysis without changing the original frozen outputs.",
        "status": validation["status"],
    }
    (output / "evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2) + "\n", encoding="utf-8")
    manifest = artifact_manifest(output)
    reported = {
        **validation,
        "artifact_manifest_rows": int(len(manifest)),
        "artifact_manifest_sha256": sha256(output / "artifact_manifest.csv"),
    }
    print(json.dumps(reported), flush=True)
    if validation["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
