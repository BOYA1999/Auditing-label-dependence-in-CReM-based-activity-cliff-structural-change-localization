import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values):
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def factor_design(frame):
    columns = {"intercept": np.ones(len(frame))}
    for representation in ["molformer", "unimol"]:
        columns[f"representation[{representation}]"] = (frame.representation == representation).astype(float)
    columns["predictor[mlp]"] = (frame.predictor == "mlp").astype(float)
    columns["explainer[crem_lime]"] = (frame.explainer == "crem_lime").astype(float)
    for representation in ["molformer", "unimol"]:
        r = columns[f"representation[{representation}]"]
        columns[f"representation[{representation}]:predictor[mlp]"] = r * columns["predictor[mlp]"]
        columns[f"representation[{representation}]:explainer[crem_lime]"] = r * columns["explainer[crem_lime]"]
    columns["predictor[mlp]:explainer[crem_lime]"] = columns["predictor[mlp]"] * columns["explainer[crem_lime]"]
    for representation in ["molformer", "unimol"]:
        columns[f"representation[{representation}]:predictor[mlp]:explainer[crem_lime]"] = (
            columns[f"representation[{representation}]"]
            * columns["predictor[mlp]"] * columns["explainer[crem_lime]"]
        )
    factor_names = list(columns)
    target_dummies = pd.get_dummies(frame.dataset, drop_first=True, dtype=float)
    x = np.column_stack([*columns.values(), target_dummies.to_numpy()])
    names = factor_names + [f"target[{value}]" for value in target_dummies.columns]
    return x, names, factor_names


def clustered_factor_model(frame):
    x, names, factor_names = factor_design(frame)
    y = frame.delta_sanity.to_numpy(dtype=float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for target in frame.dataset.unique():
        mask = (frame.dataset == target).to_numpy()
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    groups = frame.dataset.nunique()
    correction = groups / (groups - 1) * (len(frame) - 1) / (len(frame) - x.shape[1])
    covariance = correction * xtx_inv @ meat @ xtx_inv
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    rows = []
    for index, name in enumerate(factor_names):
        statistic = beta[index] / standard_error[index] if standard_error[index] > 0 else np.nan
        rows.append({
            "term": name, "estimate": float(beta[index]), "cluster_se": float(standard_error[index]),
            "t": float(statistic), "df": groups - 1,
            "p": float(2 * stats.t.sf(abs(statistic), groups - 1)) if np.isfinite(statistic) else np.nan,
        })
    tests = {
        "representation_predictor": [
            "representation[molformer]:predictor[mlp]", "representation[unimol]:predictor[mlp]",
        ],
        "representation_explainer": [
            "representation[molformer]:explainer[crem_lime]", "representation[unimol]:explainer[crem_lime]",
        ],
        "predictor_explainer": ["predictor[mlp]:explainer[crem_lime]"],
        "three_way": [
            "representation[molformer]:predictor[mlp]:explainer[crem_lime]",
            "representation[unimol]:predictor[mlp]:explainer[crem_lime]",
        ],
    }
    test_rows = []
    for label, terms in tests.items():
        indices = [names.index(term) for term in terms]
        values = beta[indices]
        subcov = covariance[np.ix_(indices, indices)]
        statistic = float(values @ np.linalg.pinv(subcov) @ values / len(indices))
        test_rows.append({
            "test": label, "df_numerator": len(indices), "df_denominator": groups - 1,
            "f": statistic, "p": float(stats.f.sf(statistic, len(indices), groups - 1)),
        })
    tests_frame = pd.DataFrame(test_rows)
    order = np.argsort(tests_frame.p.to_numpy())
    adjusted = np.empty(len(tests_frame))
    ranked = tests_frame.p.to_numpy()[order] * len(tests_frame) / np.arange(1, len(tests_frame) + 1)
    adjusted[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    tests_frame["q_bh"] = np.minimum(adjusted, 1.0)
    return pd.DataFrame(rows), tests_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    margin = float(cfg["inference"]["equivalence_margin"])
    draws = int(cfg["inference"]["bootstrap_draws"])
    rng = np.random.default_rng(cfg["inference"]["bootstrap_seed"])
    confirmation = set(cfg["confirmation_targets"])
    pairs = pd.read_csv(args.scores / "pair_explanations.csv")
    pairs = pairs[(pairs.dataset.isin(confirmation)) & (pairs.component_split == "test")].copy()
    keys = ["dataset", "component_id", "representation", "predictor", "explainer"]
    random_mean = pairs[pairs.variant == "label_permuted"].groupby(keys, as_index=False).normalized_ap_lift.mean().rename(
        columns={"normalized_ap_lift": "randomized_mean_nap"}
    )
    trained = pairs[pairs.variant == "trained"].merge(random_mean, on=keys, validate="many_to_one")
    trained["delta_sanity"] = trained.normalized_ap_lift - trained.randomized_mean_nap
    trained.to_csv(args.output / "trained_seed_deltas.csv", index=False)
    component = pairs.groupby(keys + ["variant"], as_index=False).normalized_ap_lift.mean().pivot(
        index=keys, columns="variant", values="normalized_ap_lift"
    ).reset_index()
    component["delta_sanity"] = component.trained - component.label_permuted
    component.to_csv(args.output / "component_deltas.csv", index=False)

    target_rows = trained.groupby("dataset", as_index=False).agg(
        median_delta=("delta_sanity", "median"), mean_delta=("delta_sanity", "mean"),
        components=("component_id", "nunique"), rows=("delta_sanity", "size"),
    )
    target_rows.to_csv(args.output / "target_estimates.csv", index=False)
    target_arrays = {}
    for target, frame in trained.groupby("dataset"):
        target_arrays[target] = [group.delta_sanity.to_numpy() for _, group in frame.groupby("component_id")]
    target_names = sorted(target_arrays)
    bootstrap = np.empty(draws)
    for draw in range(draws):
        medians = []
        for target_index in rng.integers(0, len(target_names), len(target_names)):
            values = target_arrays[target_names[target_index]]
            sampled = [values[index] for index in rng.integers(0, len(values), len(values))]
            medians.append(np.median(np.concatenate(sampled)))
        bootstrap[draw] = np.median(medians)
    pd.DataFrame({"draw": np.arange(draws), "estimate": bootstrap}).to_csv(args.output / "primary_bootstrap.csv", index=False)
    primary_point = float(np.median(target_rows.median_delta))
    primary_ci = interval(bootstrap)
    primary_equivalent = primary_ci[0] >= -margin and primary_ci[1] <= margin

    cell_rows = []
    for key, frame in component.groupby(["dataset", "representation", "predictor", "explainer"]):
        values = frame.delta_sanity.to_numpy()
        sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
        distribution = np.median(sampled, axis=1)
        ci = interval(distribution)
        cell_rows.append({
            "dataset": key[0], "representation": key[1], "predictor": key[2], "explainer": key[3],
            "median_delta": float(np.median(values)), "ci_low": ci[0], "ci_high": ci[1],
            "components": len(values), "equivalent": ci[0] >= -margin and ci[1] <= margin,
        })
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(args.output / "cell_equivalence.csv", index=False)
    equivalent_fraction = float(cells.equivalent.mean())
    secondary_pass = equivalent_fraction >= cfg["inference"]["minimum_equivalent_target_configuration_fraction"]

    coefficients, factor_tests = clustered_factor_model(component)
    coefficients.to_csv(args.output / "factor_coefficients.csv", index=False)
    factor_tests.to_csv(args.output / "factor_interaction_tests.csv", index=False)

    similarity = pd.read_csv(args.scores / "atom_rank_similarity.csv")
    similarity = similarity[(similarity.dataset.isin(confirmation)) & (similarity.comparison == "random_mean")]
    similarity_targets = similarity.groupby("dataset", as_index=False).agg(
        median_supported_spearman=("spearman_supported", "median"),
        median_all_atom_spearman=("spearman_all_atoms", "median"),
        finite_supported=("spearman_supported", "count"),
    )
    similarity_targets.to_csv(args.output / "atom_similarity_target_summary.csv", index=False)
    finite_similarity = similarity_targets.median_supported_spearman.dropna().to_numpy()
    similarity_bootstrap = np.median(
        finite_similarity[
            rng.integers(0, len(finite_similarity), size=(draws, len(finite_similarity)))
        ], axis=1,
    )
    similarity_summary = {
        "target_balanced_median_supported_spearman": float(np.median(finite_similarity)),
        "target_bootstrap_ci": interval(similarity_bootstrap),
        "target_balanced_median_all_atom_spearman": float(np.median(similarity_targets.median_all_atom_spearman)),
        "targets_with_finite_supported_spearman": int(len(finite_similarity)),
    }

    metrics = pd.read_csv(args.heads / "prediction_metrics.csv")
    metrics = metrics[
        (metrics.dataset.isin(confirmation)) & (metrics.variant == "trained") & (metrics.split == "test")
    ].groupby(["dataset", "representation", "predictor"], as_index=False).rmse_against_observed.mean()
    cell_target = component.groupby(["dataset", "representation", "predictor", "explainer"], as_index=False).delta_sanity.median()
    association = cell_target.merge(metrics, on=["dataset", "representation", "predictor"], validate="many_to_one")
    association["rmse_within_target"] = association.rmse_against_observed - association.groupby("dataset").rmse_against_observed.transform("mean")
    association["delta_within_target"] = association.delta_sanity - association.groupby("dataset").delta_sanity.transform("mean")
    correlation = stats.spearmanr(association.rmse_within_target, association.delta_within_target)
    association.to_csv(args.output / "prediction_sensitivity_association.csv", index=False)
    association_groups = [
        frame[["rmse_within_target", "delta_within_target"]].to_numpy()
        for _, frame in association.groupby("dataset")
    ]
    association_rng = np.random.default_rng(cfg["inference"]["bootstrap_seed"] + 1)
    association_bootstrap = np.empty(draws)
    for draw in range(draws):
        sampled = np.concatenate([
            association_groups[index]
            for index in association_rng.integers(0, len(association_groups), len(association_groups))
        ])
        association_bootstrap[draw] = stats.spearmanr(sampled[:, 0], sampled[:, 1]).statistic
    target_correlations = association.groupby("dataset").apply(
        lambda frame: stats.spearmanr(frame.rmse_against_observed, frame.delta_sanity).statistic,
        include_groups=False,
    )
    target_correlations.rename("spearman").to_csv(args.output / "prediction_sensitivity_target_correlations.csv")
    prediction_association = {
        "target_demeaned_spearman": float(correlation.statistic),
        "target_cluster_bootstrap_ci": interval(association_bootstrap),
        "median_within_target_spearman": float(np.median(target_correlations)),
        "naive_cell_p": float(correlation.pvalue), "cells": int(len(association)),
    }

    fidelity = pd.read_csv(args.scores / "local_fidelity.csv")
    fidelity = fidelity[(fidelity.dataset.isin(confirmation)) & (fidelity.component_split == "test")]
    fidelity_summary = fidelity.groupby(["variant", "representation", "predictor"], as_index=False).agg(
        median_local_weighted_r2=("local_weighted_r2", "median"),
        fifth_percentile_r2=("local_weighted_r2", lambda values: np.quantile(values, 0.05)),
        molecules=("row_id", "size"),
    )
    fidelity_summary.to_csv(args.output / "local_fidelity_summary.csv", index=False)

    if primary_equivalent and secondary_pass:
        decision = "CONFIRMED_BENCHMARK_SURVIVAL"
    elif primary_ci[0] > margin:
        decision = "SANITY_TEST_PASSED"
    else:
        decision = "HETEROGENEOUS_OR_INCONCLUSIVE"
    summary = {
        "run_id": cfg["run_id"], "status": "success", "decision": decision,
        "confirmation_targets": len(target_names),
        "primary_target_balanced_median_delta": primary_point,
        "primary_bootstrap_ci": primary_ci,
        "equivalence_margin": [-margin, margin],
        "primary_equivalent": bool(primary_equivalent),
        "equivalent_target_configuration_cells": int(cells.equivalent.sum()),
        "target_configuration_cells": int(len(cells)),
        "equivalent_cell_fraction": equivalent_fraction,
        "secondary_two_thirds_pass": bool(secondary_pass),
        "atom_similarity": similarity_summary,
        "prediction_association": prediction_association,
        "input_hashes": {
            "config": sha256(args.config), "pair_explanations": sha256(args.scores / "pair_explanations.csv"),
            "atom_rank_similarity": sha256(args.scores / "atom_rank_similarity.csv"),
            "prediction_metrics": sha256(args.heads / "prediction_metrics.csv"),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
