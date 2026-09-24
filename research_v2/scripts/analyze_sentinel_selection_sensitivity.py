import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def expected_analysis_counts(spec):
    legacy = {
        "targets": 6,
        "alternative_eligible_components": 139,
        "paired_comparison_components": 138,
        "unpaired_alternative_components": 1,
        "alternative_component_configuration_rows": 1668,
        "paired_component_configuration_rows": 1656,
        "factorial_configurations": 12,
    }
    return spec.get("expected_counts", {}).get("analysis", legacy)


def component_metrics(frame, targets, representations, predictors, explainers, label):
    frame = frame[
        frame.dataset.isin(targets)
        & (frame.component_split == "test")
        & (frame.seed == 42)
        & frame.representation.isin(representations)
        & frame.predictor.isin(predictors)
        & frame.explainer.isin(explainers)
        & frame.variant.isin(["trained", "label_permuted"])
    ].copy()
    key = ["dataset", "component_id", "left_index", "right_index", "representation", "predictor", "explainer", "seed", "variant"]
    if frame.duplicated(key).any():
        raise ValueError(f"{label} has duplicate factorial keys")
    base = ["dataset", "component_id", "left_index", "right_index", "representation", "predictor", "explainer", "seed"]
    trained = frame[frame.variant == "trained"].drop(columns="variant").rename(columns={
        "average_precision": "trained_ap", "normalized_ap_lift": "trained_nap", "prevalence": "trained_prevalence"
    })
    permuted = frame[frame.variant == "label_permuted"].drop(columns="variant").rename(columns={
        "average_precision": "permuted_ap", "normalized_ap_lift": "permuted_nap", "prevalence": "permuted_prevalence"
    })
    keep = base + ["trained_ap", "trained_nap", "trained_prevalence"]
    other = base + ["permuted_ap", "permuted_nap", "permuted_prevalence"]
    result = trained[keep].merge(permuted[other], on=base, validate="one_to_one")
    if not np.allclose(result.trained_prevalence, result.permuted_prevalence, atol=1e-12):
        raise ValueError(f"{label} prevalence differs by model variant")
    result["prevalence"] = result.trained_prevalence
    result["delta_nap"] = result.trained_nap - result.permuted_nap
    result["delta_ap"] = result.trained_ap - result.permuted_ap
    return result.drop(columns=["trained_prevalence", "permuted_prevalence"])


def two_stage_bootstrap(frame, column, draws, seed):
    grouped = {
        target: [part[column].to_numpy(float) for _, part in target_frame.groupby("component_id", sort=True)]
        for target, target_frame in frame.groupby("dataset", sort=True)
    }
    targets = sorted(grouped)
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=float)
    for draw in range(draws):
        target_estimates = []
        for target_index in rng.integers(0, len(targets), len(targets)):
            components = grouped[targets[target_index]]
            sampled = [components[index] for index in rng.integers(0, len(components), len(components))]
            target_estimates.append(np.median(np.concatenate(sampled)))
        output[draw] = np.median(target_estimates)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["sentinel_selection_sensitivity"]
    inputs = {key: resolve(value) for key, value in spec["inputs"].items()}
    experiment = resolve(spec["experiment_root"])
    output = resolve(spec["analysis_output"])
    output.mkdir(parents=True, exist_ok=True)
    targets = spec["representative_targets"]
    representations = list(cfg["representations"])
    predictors = spec["predictors"]
    explainers = cfg["explainers"]
    selected = pd.read_csv(experiment / "prepared/selected_pairs_all.csv")
    eligible = selected[selected.valid_normalized_ap_prevalence.astype(bool)].copy()
    alternate_source = experiment / "explanations/pair_explanations.csv"
    original_source = inputs["original_pair_explanations"]
    alternate = component_metrics(
        pd.read_csv(alternate_source), targets, representations, predictors, explainers, "alternate"
    )
    original_all = component_metrics(
        pd.read_csv(original_source), targets, representations, predictors, explainers, "original"
    )
    valid_components = eligible[["dataset", "component_id"]].drop_duplicates()
    expected_components = len(valid_components)
    factorial_size = len(representations) * len(predictors) * len(explainers)
    expected_alternative_rows = expected_components * factorial_size
    if len(alternate) != expected_alternative_rows:
        raise ValueError("alternate component/configuration crossing is incomplete")
    original_components = original_all[["dataset", "component_id"]].drop_duplicates()
    paired_components = valid_components.merge(original_components, on=["dataset", "component_id"], validate="one_to_one")
    original = original_all.merge(paired_components, on=["dataset", "component_id"], validate="many_to_one")
    expected_paired_rows = len(paired_components) * factorial_size
    if len(original) != expected_paired_rows:
        raise ValueError("original corresponding-subset crossing is incomplete")

    selection_fields = eligible[[
        "dataset", "component_id", "pair_index", "original_pair_index", "potency_difference",
        "original_potency_difference", "component_median_potency_difference", "component_pair_count",
        "same_as_original",
    ]]
    alternate = alternate.merge(selection_fields, on=["dataset", "component_id"], validate="many_to_one")
    alternate.to_csv(output / "alternative_component_configuration_metrics.csv", index=False)
    original.to_csv(output / "original_component_configuration_metrics.csv", index=False)
    config_key = ["dataset", "component_id", "representation", "predictor", "explainer"]
    comparison = alternate.merge(
        original[config_key + ["trained_nap", "permuted_nap", "delta_nap"]],
        on=config_key,
        suffixes=("_alternative", "_original"),
        validate="one_to_one",
    )
    comparison["selection_change_in_delta_nap"] = comparison.delta_nap_alternative - comparison.delta_nap_original
    comparison.to_csv(output / "paired_selection_comparison.csv", index=False)
    unpaired = valid_components.merge(paired_components, on=["dataset", "component_id"], how="left", indicator=True)
    unpaired = unpaired[unpaired._merge == "left_only"].drop(columns="_merge").merge(
        selection_fields, on=["dataset", "component_id"], validate="one_to_one"
    )
    unpaired["reason"] = "original_maximum-difference sentinel has undefined normalized AP"
    unpaired.to_csv(output / "unpaired_alternative_components.csv", index=False)

    original_sentinels = pd.read_csv(inputs["original_sentinel_pairs"])
    original_sentinels = original_sentinels[
        original_sentinels.dataset.isin(targets) & (original_sentinels.component_split == "test")
    ].set_index(["dataset", "component_id"])
    rationale_rows = []
    for row in eligible.itertuples(index=False):
        original_row = original_sentinels.loc[(row.dataset, int(row.component_id))]
        alternative_sets = {
            int(row.left_index): set(json.loads(row.rationale_left)),
            int(row.right_index): set(json.loads(row.rationale_right)),
        }
        original_sets = {
            int(original_row.left_index): set(json.loads(original_row.rationale_left)),
            int(original_row.right_index): set(json.loads(original_row.rationale_right)),
        }
        shared = sorted(set(alternative_sets) & set(original_sets))
        for molecule_index in shared:
            union = alternative_sets[molecule_index] | original_sets[molecule_index]
            intersection = alternative_sets[molecule_index] & original_sets[molecule_index]
            rationale_rows.append({
                "dataset": row.dataset,
                "component_id": int(row.component_id),
                "same_pair": bool(row.same_as_original),
                "molecule_index": molecule_index,
                "alternative_positive_atoms": len(alternative_sets[molecule_index]),
                "original_positive_atoms": len(original_sets[molecule_index]),
                "intersection_atoms": len(intersection),
                "union_atoms": len(union),
                "jaccard": 1.0 if not union else len(intersection) / len(union),
                "exact": alternative_sets[molecule_index] == original_sets[molecule_index],
            })
    rationale_overlap = pd.DataFrame(rationale_rows)
    rationale_overlap.to_csv(output / "rationale_overlap.csv", index=False)

    alternative_target = alternate.groupby("dataset", as_index=False).agg(
        alternative_full_components=("component_id", "nunique"),
        alternative_full_median_delta_nap=("delta_nap", "median"),
    )
    target = comparison.groupby("dataset", as_index=False).agg(
        components=("component_id", "nunique"),
        alternative_paired_median_delta_nap=("delta_nap_alternative", "median"),
        original_paired_median_delta_nap=("delta_nap_original", "median"),
        paired_median_selection_change=("selection_change_in_delta_nap", "median"),
        selected_median_potency_difference=("potency_difference", "median"),
        original_median_potency_difference=("original_potency_difference", "median"),
    ).merge(alternative_target, on="dataset", validate="one_to_one")
    target["paired_primary_estimate_change"] = (
        target.alternative_paired_median_delta_nap - target.original_paired_median_delta_nap
    )
    target.to_csv(output / "target_estimates.csv", index=False)
    target_config = comparison.groupby(["dataset", "representation", "predictor", "explainer"], as_index=False).agg(
        components=("component_id", "nunique"),
        alternative_paired_median_delta_nap=("delta_nap_alternative", "median"),
        original_paired_median_delta_nap=("delta_nap_original", "median"),
        paired_median_selection_change=("selection_change_in_delta_nap", "median"),
    )
    configuration = target_config.groupby(["representation", "predictor", "explainer"], as_index=False).agg(
        target_balanced_alternative_paired_delta_nap=("alternative_paired_median_delta_nap", "median"),
        target_balanced_original_paired_delta_nap=("original_paired_median_delta_nap", "median"),
        target_balanced_paired_selection_change=("paired_median_selection_change", "median"),
        targets=("dataset", "nunique"),
    )
    alternative_target_config = alternate.groupby(
        ["dataset", "representation", "predictor", "explainer"], as_index=False
    ).agg(alternative_full_median_delta_nap=("delta_nap", "median"))
    alternative_configuration = alternative_target_config.groupby(
        ["representation", "predictor", "explainer"], as_index=False
    ).agg(target_balanced_alternative_full_delta_nap=("alternative_full_median_delta_nap", "median"))
    configuration = configuration.merge(
        alternative_configuration,
        on=["representation", "predictor", "explainer"],
        validate="one_to_one",
    )
    configuration["paired_primary_estimate_change"] = (
        configuration.target_balanced_alternative_paired_delta_nap
        - configuration.target_balanced_original_paired_delta_nap
    )
    configuration.to_csv(output / "configuration_estimates.csv", index=False)

    draws = int(spec["bootstrap"]["draws"])
    seed = int(spec["bootstrap"]["seed"])
    alternative_full_bootstrap = two_stage_bootstrap(alternate, "delta_nap", draws, seed)
    alternative_paired_bootstrap = two_stage_bootstrap(comparison, "delta_nap_alternative", draws, seed)
    original_paired_bootstrap = two_stage_bootstrap(comparison, "delta_nap_original", draws, seed)
    change_bootstrap = two_stage_bootstrap(comparison, "selection_change_in_delta_nap", draws, seed)
    primary_change_bootstrap = alternative_paired_bootstrap - original_paired_bootstrap
    bootstrap = pd.DataFrame({
        "draw": np.arange(draws),
        "alternative_full_delta_nap": alternative_full_bootstrap,
        "alternative_paired_delta_nap": alternative_paired_bootstrap,
        "original_paired_delta_nap": original_paired_bootstrap,
        "paired_primary_estimate_change": primary_change_bootstrap,
        "paired_selection_change": change_bootstrap,
    })
    bootstrap.to_csv(output / "bootstrap.csv", index=False)

    required_outputs = [
        "alternative_component_configuration_metrics.csv",
        "original_component_configuration_metrics.csv",
        "paired_selection_comparison.csv",
        "target_estimates.csv",
        "configuration_estimates.csv",
        "bootstrap.csv",
        "unpaired_alternative_components.csv",
        "rationale_overlap.csv",
    ]
    finite_columns = [
        "delta_nap_alternative", "delta_nap_original", "selection_change_in_delta_nap"
    ]
    observed_counts = {
        "targets": len(target),
        "alternative_eligible_components": expected_components,
        "paired_comparison_components": len(paired_components),
        "unpaired_alternative_components": len(unpaired),
        "alternative_component_configuration_rows": len(alternate),
        "paired_component_configuration_rows": len(comparison),
        "factorial_configurations": len(configuration),
    }
    expected_counts = expected_analysis_counts(spec)
    if set(expected_counts) != set(observed_counts):
        raise ValueError("analysis expected-count keys do not match the validation contract")
    count_checks = {key: observed_counts[key] == int(value) for key, value in expected_counts.items()}
    status = "pass" if (
        all(count_checks.values())
        and len(comparison) == expected_paired_rows
        and not comparison.duplicated(config_key).any()
        and np.isfinite(comparison[finite_columns].to_numpy()).all()
        and len(target) == len(targets)
        and len(configuration) == len(representations) * len(predictors) * len(explainers)
    ) else "fail"
    summary = {
        "run_id": cfg["run_id"],
        "status": status,
        "alternative_eligible_components": int(expected_components),
        "paired_comparison_components": int(len(paired_components)),
        "unpaired_alternative_components": int(len(unpaired)),
        "alternative_component_configuration_rows": int(len(alternate)),
        "paired_component_configuration_rows": int(len(comparison)),
        "expected_paired_component_configuration_rows": int(expected_paired_rows),
        "factorial_configurations": int(len(configuration)),
        "target_balanced_alternative_full_delta_nap": float(target.alternative_full_median_delta_nap.median()),
        "target_balanced_alternative_paired_delta_nap": float(target.alternative_paired_median_delta_nap.median()),
        "target_balanced_original_paired_delta_nap": float(target.original_paired_median_delta_nap.median()),
        "paired_primary_estimate_change": float(
            target.alternative_paired_median_delta_nap.median()
            - target.original_paired_median_delta_nap.median()
        ),
        "target_balanced_paired_selection_change": float(target.paired_median_selection_change.median()),
        "alternative_full_delta_nap_ci95": interval(alternative_full_bootstrap),
        "alternative_paired_delta_nap_ci95": interval(alternative_paired_bootstrap),
        "original_paired_delta_nap_ci95": interval(original_paired_bootstrap),
        "paired_primary_estimate_change_ci95": interval(primary_change_bootstrap),
        "paired_selection_change_ci95": interval(change_bootstrap),
        "targets_with_positive_alternative_full_delta": int((target.alternative_full_median_delta_nap > 0).sum()),
        "targets_with_positive_original_paired_delta": int((target.original_paired_median_delta_nap > 0).sum()),
        "configurations_with_positive_alternative_full_delta": int((configuration.target_balanced_alternative_full_delta_nap > 0).sum()),
        "exact_pair_overlap": int(selected.same_as_original.sum()),
        "changed_pairs": int((~selected.same_as_original).sum()),
        "shared_endpoints": int(len(rationale_overlap)),
        "same_pair_rationale_exact_endpoints": int(
            rationale_overlap[rationale_overlap.same_pair].exact.sum()
        ),
        "same_pair_rationale_endpoints": int(rationale_overlap.same_pair.sum()),
        "changed_pair_shared_endpoint_rationale_jaccard_median": float(
            rationale_overlap.loc[~rationale_overlap.same_pair, "jaccard"].median()
        ),
        "changed_pair_shared_endpoints": int((~rationale_overlap.same_pair).sum()),
        "count_validation": {
            "expected": {key: int(value) for key, value in expected_counts.items()},
            "observed": {key: int(value) for key, value in observed_counts.items()},
            "checks": count_checks,
        },
        "input_hashes": {
            "config": sha256(args.config),
            "selected_pairs": sha256(experiment / "prepared/selected_pairs_all.csv"),
            "alternative_pair_explanations": sha256(alternate_source),
            "original_pair_explanations": sha256(original_source),
        },
        "output_hashes": {name: sha256(output / name) for name in required_outputs},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
