import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from rdkit import Chem, rdBase
from scipy import stats


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


def interval(values, level=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
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


def grouped_average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives <= 0 or positives >= len(labels):
        raise ValueError("average precision requires both classes")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    cumulative = np.cumsum(sorted_labels)[ends]
    group_positive = np.diff(np.r_[0, cumulative])
    precision = cumulative / (ends + 1)
    return float(np.sum((group_positive / positives) * precision))


def tie_aware_top_k(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    k = int(labels.sum())
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cutoff = sorted_scores[k - 1]
    above = sorted_scores > cutoff
    tied = sorted_scores == cutoff
    need = k - int(above.sum())
    expected_positive = float(sorted_labels[above].sum())
    if need > 0:
        expected_positive += need * float(sorted_labels[tied].mean())
    precision = expected_positive / k
    recall = expected_positive / k
    prevalence = float(labels.mean())
    return precision, recall, precision / prevalence, int(tied.sum())


def empirical_null_metrics(labels, scores, replicates, seed, label):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    total = len(labels)
    prevalence = positives / total
    observed_ap = grouped_average_precision(labels, scores)
    observed_nap = (observed_ap - prevalence) / (1.0 - prevalence)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    rng = rng_for(seed, label)
    random_values = rng.random((replicates, total))
    selected = np.argpartition(random_values, positives - 1, axis=1)[:, :positives]
    random_labels = np.zeros((replicates, total), dtype=np.int8)
    random_labels[np.arange(replicates)[:, None], selected] = 1
    sorted_random = random_labels[:, order]
    cumulative = np.cumsum(sorted_random, axis=1)[:, ends]
    group_positive = np.diff(np.c_[np.zeros(replicates, dtype=int), cumulative], axis=1)
    precision = cumulative / (ends + 1)
    null_ap = np.sum((group_positive / positives) * precision, axis=1)
    null_nap = (null_ap - prevalence) / (1.0 - prevalence)
    null_mean = float(null_nap.mean())
    null_sd = float(null_nap.std(ddof=1))
    top_precision, top_recall, enrichment, cutoff_ties = tie_aware_top_k(labels, scores)
    return {
        "atoms": total,
        "positive_atoms": positives,
        "prevalence": prevalence,
        "score_unique_values": int(np.unique(scores).size),
        "score_zero_fraction": float(np.mean(scores == 0.0)),
        "observed_ap": observed_ap,
        "observed_nap": observed_nap,
        "null_mean_ap": float(null_ap.mean()),
        "null_sd_ap": float(null_ap.std(ddof=1)),
        "null_mean_nap": null_mean,
        "null_sd_nap": null_sd,
        "null_adjusted_nap": observed_nap - null_mean,
        "null_z": (observed_nap - null_mean) / null_sd if null_sd > 0 else np.nan,
        "empirical_upper_p": float((1 + np.sum(null_nap >= observed_nap)) / (replicates + 1)),
        "precision_at_changed_count": top_precision,
        "recall_at_changed_count": top_recall,
        "changed_atom_enrichment_at_changed_count": enrichment,
        "cutoff_tie_atoms": cutoff_ties,
    }


def parse_json_atoms(value):
    return [int(item) for item in json.loads(value)]


def pair_label_arrays(pair, atom_counts, rationale):
    left_atoms = atom_counts[pair.left_id]
    right_atoms = atom_counts[pair.right_id]
    if rationale == "boundary_extended":
        left_positive = parse_json_atoms(pair.rationale_left)
        right_positive = parse_json_atoms(pair.rationale_right)
    elif rationale == "strict_unmatched":
        match = json.loads(pair.match)
        matched_left = {int(left) for left, _ in match}
        matched_right = {int(right) for _, right in match}
        left_positive = sorted(set(range(left_atoms)) - matched_left)
        right_positive = sorted(set(range(right_atoms)) - matched_right)
    else:
        raise ValueError(rationale)
    left = np.isin(np.arange(left_atoms), left_positive)
    right = np.isin(np.arange(right_atoms), right_positive)
    labels = np.concatenate([left, right])
    return labels, bool(0 < labels.sum() < len(labels))


def load_atom_scores(path, targets, explainer):
    columns = [
        "dataset", "representation", "predictor", "explainer", "seed", "variant",
        "row_id", "component_split", "atom_index", "atom_score", "supported",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250000):
        keep = chunk[
            chunk.dataset.isin(targets)
            & (chunk.component_split == "test")
            & (chunk.explainer == explainer)
        ]
        if len(keep):
            parts.append(keep.drop(columns=["explainer", "component_split"]))
    frame = pd.concat(parts, ignore_index=True)
    key = ["dataset", "representation", "predictor", "seed", "variant", "row_id", "atom_index"]
    if frame.duplicated(key).any():
        raise ValueError("atom score key is not unique")
    return frame


def atom_array_lookup(frame):
    lookup = {}
    key = ["dataset", "representation", "predictor", "seed", "variant", "row_id"]
    for values, group in frame.groupby(key, sort=False):
        ordered = group.sort_values("atom_index")
        expected = np.arange(len(ordered))
        if not np.array_equal(ordered.atom_index.to_numpy(int), expected):
            raise ValueError(f"non-contiguous atom index: {values}")
        lookup[values] = ordered.atom_score.to_numpy(float)
    return lookup


def summarize_null(frame, source, config_columns):
    metrics = [
        "observed_nap", "null_mean_nap", "null_adjusted_nap", "null_z",
        "precision_at_changed_count", "recall_at_changed_count", "changed_atom_enrichment_at_changed_count",
    ]
    rows = []
    group_columns = ["rationale"] + config_columns
    for key, group in frame.groupby(group_columns, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        target = group.groupby("dataset", as_index=False)[metrics].median(numeric_only=True)
        row = {"source": source, **dict(zip(group_columns, key)), "targets": int(target.dataset.nunique()), "pairs": int(group[["dataset", "component_id"]].drop_duplicates().shape[0])}
        for metric in metrics:
            row[f"target_balanced_median_{metric}"] = float(target[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def primary_from_components(frame, value, draws, seed, label):
    target_values = frame.groupby("dataset", as_index=False)[value].median()
    distribution = two_stage_bootstrap(frame, value, draws, seed, label)
    ci95 = interval(distribution, 0.95)
    ci90 = interval(distribution, 0.90)
    return {
        "estimate": float(target_values[value].median()),
        "ci90": list(ci90),
        "ci95": list(ci95),
        "targets": int(frame.dataset.nunique()),
        "components": int(frame[["dataset", "component_id"]].drop_duplicates().shape[0]),
        "configurations": int(frame[["representation", "predictor"]].drop_duplicates().shape[0]),
    }, distribution


def variance_components(cell, value, draws, seed, label, margin):
    work = cell.copy()
    work["configuration"] = work.representation.astype(str) + "/" + work.predictor.astype(str)
    matrix = work.pivot(index="dataset", columns="configuration", values=value).sort_index().sort_index(axis=1)
    if matrix.isna().any().any():
        raise ValueError("heterogeneity matrix is incomplete")
    values = matrix.to_numpy(float)

    def components(array):
        n_target, n_config = array.shape
        grand = float(array.mean())
        target_mean = array.mean(axis=1)
        config_mean = array.mean(axis=0)
        residual = array - target_mean[:, None] - config_mean[None, :] + grand
        ms_target = n_config * float(np.sum(np.square(target_mean - grand))) / (n_target - 1)
        ms_config = n_target * float(np.sum(np.square(config_mean - grand))) / (n_config - 1)
        ms_residual = float(np.sum(np.square(residual))) / ((n_target - 1) * (n_config - 1))
        target_var = max(0.0, (ms_target - ms_residual) / n_config)
        config_var = max(0.0, (ms_config - ms_residual) / n_target)
        residual_var = max(0.0, ms_residual)
        return grand, target_var, config_var, residual_var

    estimate = components(values)
    rng = rng_for(seed, label)
    boot = np.empty((draws, 4), dtype=float)
    for draw in range(draws):
        target_index = rng.integers(0, values.shape[0], values.shape[0])
        config_index = rng.integers(0, values.shape[1], values.shape[1])
        boot[draw] = components(values[np.ix_(target_index, config_index)])
    grand, target_var, config_var, residual_var = estimate
    total_sd = math.sqrt(target_var + config_var + residual_var)
    prediction_interval = [grand - 1.96 * total_sd, grand + 1.96 * total_sd]
    if total_sd > 0:
        within = stats.norm.cdf((margin - grand) / total_sd) - stats.norm.cdf((-margin - grand) / total_sd)
        exceed = float(1.0 - within)
    else:
        exceed = float(abs(grand) > margin)
    result = {
        "model": "balanced two-way random-intercept method-of-moments on target-configuration cell medians; residual includes target-by-configuration interaction and cell sampling error",
        "grand_mean": grand,
        "target_variance": target_var,
        "configuration_variance": config_var,
        "residual_interaction_sampling_variance": residual_var,
        "new_target_configuration_prediction_interval_approx95": prediction_interval,
        "normal_model_probability_absolute_effect_exceeds_margin": exceed,
        "margin": margin,
        "empirical_cell_quantiles_2_5_50_97_5": [float(np.quantile(values, q)) for q in [0.025, 0.5, 0.975]],
        "empirical_fraction_absolute_effect_exceeds_margin": float(np.mean(np.abs(values) > margin)),
        "targets": int(values.shape[0]),
        "configurations": int(values.shape[1]),
        "bootstrap_ci95": {
            "grand_mean": list(interval(boot[:, 0])),
            "target_variance": list(interval(boot[:, 1])),
            "configuration_variance": list(interval(boot[:, 2])),
            "residual_interaction_sampling_variance": list(interval(boot[:, 3])),
        },
    }
    return result


def prediction_metrics(frame):
    true = frame.true_delta.to_numpy(float)
    predicted = frame.predicted_delta.to_numpy(float)
    defined = len(predicted) > 1 and np.ptp(predicted) > 0 and np.ptp(true) > 0
    return pd.Series({
        "pairs": len(frame),
        "spearman": float(stats.spearmanr(true, predicted).statistic) if defined else np.nan,
        "direction_accuracy": float(np.mean(np.sign(true) == np.sign(predicted))),
        "predicted_tie_fraction": float(np.mean(np.abs(predicted) <= 1e-12)),
    })


def clustered_slope(frame, x, y, draws, seed, label):
    work = frame[["dataset", x, y]].dropna().copy()
    work["x_centered"] = work[x] - work.groupby("dataset")[x].transform("mean")
    work["y_centered"] = work[y] - work.groupby("dataset")[y].transform("mean")

    def fit(data):
        xv = data.x_centered.to_numpy(float)
        yv = data.y_centered.to_numpy(float)
        denom = float(np.dot(xv, xv))
        return float(np.dot(xv, yv) / denom) if denom > 0 else np.nan

    slope = fit(work)
    correlation = float(stats.spearmanr(work.x_centered, work.y_centered).statistic)
    target_groups = {name: part for name, part in work.groupby("dataset", sort=True)}
    names = sorted(target_groups)
    rng = rng_for(seed, label)
    boot = []
    for _ in range(draws):
        sampled = []
        for copy_index, index in enumerate(rng.integers(0, len(names), len(names))):
            part = target_groups[names[index]].copy()
            part["dataset"] = f"bootstrap_{copy_index}"
            sampled.append(part)
        value = fit(pd.concat(sampled, ignore_index=True))
        if np.isfinite(value):
            boot.append(value)
    return {
        "quality_metric": x,
        "localization_metric": y,
        "within_target_linear_slope": slope,
        "cluster_bootstrap_ci95": list(interval(boot)),
        "within_target_spearman": correlation,
        "target_configuration_cells": int(len(work)),
        "targets": int(work.dataset.nunique()),
    }


def enumerate_mapping_sets(mol, matched_atoms, maximum):
    smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=sorted(matched_atoms), isomericSmarts=True)
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        return [frozenset(matched_atoms)], False
    matches = mol.GetSubstructMatches(query, uniquify=True, maxMatches=maximum + 1)
    truncated = len(matches) > maximum
    sets = sorted({frozenset(map(int, item)) for item in matches[:maximum]}, key=lambda item: tuple(sorted(item)))
    chosen = frozenset(matched_atoms)
    if chosen not in sets:
        sets.append(chosen)
    return sets, truncated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "research_v2/configs/round2_revision_v1.json")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    source_cfg = json.loads(resolve(cfg["source_config"]).read_text(encoding="utf-8"))
    targets = list(source_cfg["confirmation_targets"])
    representations = list(source_cfg["representations"])
    predictors = ["xgboost", "mlp"]
    seeds = list(source_cfg["seeds"])
    paths = {name: resolve(value) for name, value in cfg["inputs"].items()}
    output = resolve(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)

    molecules = pd.read_csv(paths["model_molecules"])
    molecule_lookup = molecules.set_index("row_id")
    atom_counts = {
        row.row_id: Chem.MolFromSmiles(row.canonical_smiles).GetNumAtoms()
        for row in molecules[molecules.dataset.isin(targets)].itertuples(index=False)
    }
    pairs = pd.read_csv(paths["sentinel_pairs"])
    pairs = pairs[pairs.dataset.isin(targets) & (pairs.component_split == "test")].copy()
    pairs["left_id"] = pairs.dataset + ":" + pairs.left_index.astype(int).astype(str)
    pairs["right_id"] = pairs.dataset + ":" + pairs.right_index.astype(int).astype(str)
    valid_boundary = []
    for pair in pairs.itertuples(index=False):
        _, valid = pair_label_arrays(pair, atom_counts, "boundary_extended")
        valid_boundary.append(valid)
    pairs = pairs[np.asarray(valid_boundary)].sort_values(["dataset", "component_id"]).reset_index(drop=True)
    if len(pairs) != 617:
        raise ValueError(f"expected 617 confirmation test localization pairs, found {len(pairs)}")

    atom_frame = load_atom_scores(paths["atom_scores"], targets, cfg["explainer"])
    eligible_endpoint_ids = set(pairs.left_id) | set(pairs.right_id)
    atom_frame = atom_frame[atom_frame.row_id.isin(eligible_endpoint_ids)].copy()
    expected_atom_keys = len(pairs) * 2 * len(representations) * len(predictors) * len(seeds) * 2
    actual_atom_keys = atom_frame[["dataset", "representation", "predictor", "seed", "variant", "row_id"]].drop_duplicates().shape[0]
    if actual_atom_keys != expected_atom_keys:
        raise ValueError(f"incomplete atom arrays: {actual_atom_keys} != {expected_atom_keys}")
    arrays = atom_array_lookup(atom_frame)

    null_rows = []
    null_reps = int(cfg["empirical_null_replicates"])
    for pair in pairs.itertuples(index=False):
        for rationale in cfg["rationales"]:
            labels, valid = pair_label_arrays(pair, atom_counts, rationale)
            if not valid:
                continue
            for representation in representations:
                for predictor in predictors:
                    for seed in seeds:
                        for variant in ["trained", "label_permuted"]:
                            left = arrays[(pair.dataset, representation, predictor, seed, variant, pair.left_id)]
                            right = arrays[(pair.dataset, representation, predictor, seed, variant, pair.right_id)]
                            scores = np.concatenate([left, right])
                            if len(scores) != len(labels):
                                raise ValueError("atom-score/rationale length mismatch")
                            metrics = empirical_null_metrics(
                                labels, scores, null_reps, int(cfg["bootstrap_seed"]),
                                f"model:{pair.dataset}:{pair.component_id}:{rationale}:{representation}:{predictor}:{seed}:{variant}",
                            )
                            null_rows.append({
                                "source": "model", "dataset": pair.dataset, "component_id": int(pair.component_id),
                                "rationale": rationale, "representation": representation, "predictor": predictor,
                                "seed": int(seed), "variant": variant, "baseline": "", **metrics,
                            })
    model_null = pd.DataFrame(null_rows)
    model_null.to_csv(output / "model_empirical_null_pair_metrics.csv", index=False)

    generator_atoms = pd.read_csv(paths["generator_atom_geometry"])
    generator_atoms = generator_atoms[generator_atoms.dataset.isin(targets)].copy()
    generator_rows = []
    for pair in pairs.itertuples(index=False):
        pair_atoms = generator_atoms[
            (generator_atoms.dataset == pair.dataset) & (generator_atoms.component_id == int(pair.component_id))
        ]
        for rationale in cfg["rationales"]:
            labels, valid = pair_label_arrays(pair, atom_counts, rationale)
            if not valid:
                continue
            ordered = pd.concat([
                pair_atoms[pair_atoms.row_id == pair.left_id].sort_values("atom_index"),
                pair_atoms[pair_atoms.row_id == pair.right_id].sort_values("atom_index"),
            ], ignore_index=True)
            if len(ordered) != len(labels):
                raise ValueError("generator/rationale length mismatch")
            for baseline in cfg["generator_baselines"]:
                scores = ordered[baseline].fillna(0.0).to_numpy(float)
                metrics = empirical_null_metrics(
                    labels, scores, null_reps, int(cfg["bootstrap_seed"]),
                    f"generator:{pair.dataset}:{pair.component_id}:{rationale}:{baseline}",
                )
                generator_rows.append({
                    "source": "generator", "dataset": pair.dataset, "component_id": int(pair.component_id),
                    "rationale": rationale, "representation": "", "predictor": "", "seed": np.nan,
                    "variant": "generator_only", "baseline": baseline, **metrics,
                })
    generator_null = pd.DataFrame(generator_rows)
    generator_null.to_csv(output / "generator_empirical_null_pair_metrics.csv", index=False)

    summary = pd.concat([
        summarize_null(model_null, "model", ["representation", "predictor", "variant"]),
        summarize_null(generator_null, "generator", ["baseline", "variant"]),
    ], ignore_index=True, sort=False)
    summary.to_csv(output / "empirical_null_summary.csv", index=False)

    metric_columns = [
        "observed_nap", "null_mean_nap", "null_adjusted_nap", "null_z",
        "precision_at_changed_count", "recall_at_changed_count", "changed_atom_enrichment_at_changed_count",
    ]
    component_keys = ["dataset", "component_id", "rationale", "representation", "predictor"]
    variant_mean = model_null.groupby(component_keys + ["variant"], as_index=False)[metric_columns + ["prevalence"]].mean(numeric_only=True)
    trained = variant_mean[variant_mean.variant == "trained"].drop(columns="variant")
    permuted = variant_mean[variant_mean.variant == "label_permuted"].drop(columns="variant")
    component = trained.merge(permuted, on=component_keys, suffixes=("_trained", "_permuted"), validate="one_to_one")
    for metric in metric_columns:
        component[f"delta_{metric}"] = component[f"{metric}_trained"] - component[f"{metric}_permuted"]
    component["prevalence"] = component.prevalence_trained
    component.to_csv(output / "crem_mean_empirical_null_component_deltas.csv", index=False)

    original_component = pd.read_csv(paths["symmetric_component_metrics"])
    original_component = original_component[
        original_component.dataset.isin(targets) & (original_component.explainer == cfg["explainer"])
    ].drop(columns="explainer")
    original_component.to_csv(output / "crem_mean_unadjusted_component_deltas.csv", index=False)
    draws = int(cfg["bootstrap_draws"])
    boot_seed = int(cfg["bootstrap_seed"])
    primary = {}
    unadjusted_result, unadjusted_boot = primary_from_components(original_component, "delta_nap", draws, boot_seed, "round2-unadjusted")
    primary["unadjusted_boundary_extended"] = unadjusted_result
    adjusted_boundary = component[component.rationale == "boundary_extended"]
    adjusted_result, adjusted_boot = primary_from_components(adjusted_boundary, "delta_null_adjusted_nap", draws, boot_seed, "round2-null-adjusted")
    primary["empirical_null_adjusted_boundary_extended"] = adjusted_result
    strict = component[component.rationale == "strict_unmatched"]
    strict_result, strict_boot = primary_from_components(strict, "delta_null_adjusted_nap", draws, boot_seed, "round2-strict-null-adjusted")
    primary["empirical_null_adjusted_strict_unmatched"] = strict_result
    prevalence_filtered = adjusted_boundary[adjusted_boundary.prevalence <= float(cfg["prevalence_exclusion_threshold"])]
    prevalence_result, prevalence_boot = primary_from_components(prevalence_filtered, "delta_null_adjusted_nap", draws, boot_seed, "round2-prevalence-filter")
    primary["empirical_null_adjusted_boundary_prevalence_le_0_8"] = prevalence_result
    pd.DataFrame({
        "draw": np.arange(draws), "unadjusted_boundary_extended": unadjusted_boot,
        "null_adjusted_boundary_extended": adjusted_boot, "null_adjusted_strict_unmatched": strict_boot,
        "null_adjusted_boundary_prevalence_le_0_8": prevalence_boot,
    }).to_csv(output / "crem_mean_primary_bootstrap.csv", index=False)

    unadjusted_cells = original_component.groupby(["dataset", "representation", "predictor"], as_index=False).agg(
        components=("component_id", "nunique"), median_delta_nap=("delta_nap", "median")
    )
    adjusted_cells = adjusted_boundary.groupby(["dataset", "representation", "predictor"], as_index=False).agg(
        components=("component_id", "nunique"), median_delta_null_adjusted_nap=("delta_null_adjusted_nap", "median")
    )
    cell = unadjusted_cells.merge(adjusted_cells, on=["dataset", "representation", "predictor", "components"], validate="one_to_one")
    cell.to_csv(output / "crem_mean_target_configuration_effects.csv", index=False)

    configuration_rows = []
    for representation, predictor in sorted(
        cell[["representation", "predictor"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        adjusted_part = adjusted_boundary[
            (adjusted_boundary.representation == representation)
            & (adjusted_boundary.predictor == predictor)
        ]
        strict_part = strict[
            (strict.representation == representation)
            & (strict.predictor == predictor)
        ]
        target_medians = adjusted_part.groupby("dataset", as_index=False).median(numeric_only=True)
        strict_medians = strict_part.groupby("dataset", as_index=False).median(numeric_only=True)
        selected_cell = cell[
            (cell.representation == representation)
            & (cell.predictor == predictor)
        ]
        configuration_rows.append({
            "representation": representation,
            "predictor": predictor,
            "targets": int(selected_cell.dataset.nunique()),
            "boundary_components": int(adjusted_part[["dataset", "component_id"]].drop_duplicates().shape[0]),
            "target_balanced_median_observed_nap_trained": float(target_medians.observed_nap_trained.median()),
            "target_balanced_median_observed_nap_permuted": float(target_medians.observed_nap_permuted.median()),
            "target_balanced_median_null_adjusted_nap_trained": float(target_medians.null_adjusted_nap_trained.median()),
            "target_balanced_median_null_adjusted_nap_permuted": float(target_medians.null_adjusted_nap_permuted.median()),
            "unadjusted_delta_nap": float(selected_cell.median_delta_nap.median()),
            "null_adjusted_delta_nap": float(selected_cell.median_delta_null_adjusted_nap.median()),
            "strict_unmatched_components": int(strict_part[["dataset", "component_id"]].drop_duplicates().shape[0]),
            "strict_unmatched_null_adjusted_delta_nap": float(strict_medians.delta_null_adjusted_nap.median()),
        })
    pd.DataFrame(configuration_rows).to_csv(output / "crem_mean_configuration_summary.csv", index=False)

    heterogeneity = {
        "unadjusted": variance_components(cell, "median_delta_nap", int(cfg["heterogeneity_bootstrap_draws"]), boot_seed, "heterogeneity-unadjusted", float(cfg["equivalence_margin"])),
        "empirical_null_adjusted": variance_components(cell, "median_delta_null_adjusted_nap", int(cfg["heterogeneity_bootstrap_draws"]), boot_seed, "heterogeneity-adjusted", float(cfg["equivalence_margin"])),
    }
    (output / "hierarchical_heterogeneity.json").write_text(json.dumps(heterogeneity, indent=2) + "\n", encoding="utf-8")

    pair_predictions = pd.read_csv(paths["pair_predictions"])
    pair_predictions = pair_predictions[pair_predictions.dataset.isin(targets)].copy()
    calibration = pair_predictions[(pair_predictions.component_split == "calibration") & (pair_predictions.variant == "trained")]
    calibration_seed = calibration.groupby(["dataset", "representation", "predictor", "seed"], sort=True).apply(prediction_metrics, include_groups=False).reset_index()
    calibration_quality = calibration_seed.groupby(["dataset", "representation", "predictor"], as_index=False).agg(
        calibration_pairs=("pairs", "max"), calibration_spearman=("spearman", "median"),
        calibration_direction_accuracy=("direction_accuracy", "median"), calibration_tie_fraction=("predicted_tie_fraction", "median"),
    )
    gate = cfg["calibration_gate"]
    calibration_quality["eligible"] = (
        (calibration_quality.calibration_pairs >= int(gate["minimum_pairs"]))
        & (calibration_quality.calibration_spearman >= float(gate["minimum_spearman"]))
        & (calibration_quality.calibration_direction_accuracy >= float(gate["minimum_direction_accuracy"]))
    )
    calibration_quality.to_csv(output / "calibration_prediction_quality.csv", index=False)
    competence = cell.merge(calibration_quality, on=["dataset", "representation", "predictor"], validate="one_to_one")
    competence.to_csv(output / "prediction_competence_localization_cells.csv", index=False)
    eligible_keys = competence[competence.eligible][["dataset", "representation", "predictor"]]
    eligible_components = adjusted_boundary.merge(eligible_keys, on=["dataset", "representation", "predictor"], validate="many_to_one")
    eligible_result, eligible_boot = primary_from_components(eligible_components, "delta_null_adjusted_nap", draws, boot_seed, "calibration-qualified")
    associations = [
        clustered_slope(competence, metric, "median_delta_null_adjusted_nap", draws, boot_seed, f"competence:{metric}")
        for metric in ["calibration_spearman", "calibration_direction_accuracy"]
    ]
    competence_summary = {
        "gate": gate,
        "eligible_target_configuration_cells": int(competence.eligible.sum()),
        "total_target_configuration_cells": int(len(competence)),
        "eligible_targets": int(competence.loc[competence.eligible, "dataset"].nunique()),
        "eligible_primary": eligible_result,
        "continuous_within_target_associations": associations,
        "interpretation": "Post-confirmation diagnostic; calibration outcomes define eligibility without using test localization outcomes, but the thresholds were not prospectively preregistered.",
    }
    (output / "prediction_competence_summary.json").write_text(json.dumps(competence_summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"draw": np.arange(draws), "calibration_qualified_estimate": eligible_boot}).to_csv(output / "prediction_competence_bootstrap.csv", index=False)

    test_permuted = pair_predictions[(pair_predictions.component_split == "test") & (pair_predictions.variant == "label_permuted")].copy()
    tie_rows = []
    for key, group in test_permuted.groupby(["dataset", "representation", "predictor", "seed"], sort=True):
        true = group.true_delta.to_numpy(float)
        predicted = group.predicted_delta.to_numpy(float)
        tied = np.abs(predicted) <= 1e-12
        non_tied = ~tied
        tie_rows.append({
            "dataset": key[0], "representation": key[1], "predictor": key[2], "seed": int(key[3]),
            "pairs": len(group), "exact_zero_ties": int(tied.sum()), "tie_fraction": float(tied.mean()),
            "strict_direction_accuracy": float(np.mean(np.sign(true) == np.sign(predicted))),
            "non_tie_direction_accuracy": float(np.mean(np.sign(true[non_tied]) == np.sign(predicted[non_tied]))) if non_tied.any() else np.nan,
            "sign_flipped_strict_accuracy": float(np.mean(np.sign(true) == np.sign(-predicted))),
            "median_absolute_predicted_delta": float(np.median(np.abs(predicted))),
        })
    tie_frame = pd.DataFrame(tie_rows)
    tie_frame.to_csv(output / "permuted_direction_tie_audit_by_model.csv", index=False)
    tie_target = tie_frame.groupby(["dataset", "representation", "predictor"], as_index=False).median(numeric_only=True)
    tie_summary = tie_target.groupby(["representation", "predictor"], as_index=False).agg(
        target_balanced_median_strict_accuracy=("strict_direction_accuracy", "median"),
        target_balanced_median_non_tie_accuracy=("non_tie_direction_accuracy", "median"),
        target_balanced_median_tie_fraction=("tie_fraction", "median"),
        target_balanced_median_sign_flipped_accuracy=("sign_flipped_strict_accuracy", "median"),
        target_balanced_median_absolute_predicted_delta=("median_absolute_predicted_delta", "median"),
        targets=("dataset", "nunique"),
    )
    tie_summary.to_csv(output / "permuted_direction_tie_summary.csv", index=False)

    mapping_rows = []
    mapping_ap_rows = []
    seed_mean = atom_frame.groupby(["dataset", "representation", "predictor", "variant", "row_id", "atom_index"], as_index=False).atom_score.mean()
    seed_mean_arrays = {}
    for key, group in seed_mean.groupby(["dataset", "representation", "predictor", "variant", "row_id"], sort=False):
        seed_mean_arrays[key] = group.sort_values("atom_index").atom_score.to_numpy(float)
    max_matches = int(cfg["mcs_mapping_max_matches_per_endpoint"])
    max_combinations = int(cfg["mcs_mapping_max_combinations"])
    for pair in pairs.itertuples(index=False):
        left_mol = Chem.MolFromSmiles(str(molecule_lookup.loc[pair.left_id, "canonical_smiles"]))
        right_mol = Chem.MolFromSmiles(str(molecule_lookup.loc[pair.right_id, "canonical_smiles"]))
        stored = json.loads(pair.match)
        chosen_left = frozenset(int(left) for left, _ in stored)
        chosen_right = frozenset(int(right) for _, right in stored)
        smarts = Chem.MolFragmentToSmarts(left_mol, atomsToUse=sorted(chosen_left), isomericSmarts=True)
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            left_sets, right_sets, left_truncated, right_truncated = [chosen_left], [chosen_right], False, False
        else:
            left_matches = left_mol.GetSubstructMatches(query, uniquify=True, maxMatches=max_matches + 1)
            right_matches = right_mol.GetSubstructMatches(query, uniquify=True, maxMatches=max_matches + 1)
            left_truncated = len(left_matches) > max_matches
            right_truncated = len(right_matches) > max_matches
            left_sets = sorted({frozenset(map(int, value)) for value in left_matches[:max_matches]}, key=lambda value: tuple(sorted(value)))
            right_sets = sorted({frozenset(map(int, value)) for value in right_matches[:max_matches]}, key=lambda value: tuple(sorted(value)))
            if chosen_left not in left_sets:
                left_sets.append(chosen_left)
            if chosen_right not in right_sets:
                right_sets.append(chosen_right)
        combinations = [(left, right) for left in left_sets for right in right_sets]
        combination_truncated = len(combinations) > max_combinations
        if combination_truncated:
            combinations = combinations[:max_combinations]
            if (chosen_left, chosen_right) not in combinations:
                combinations[-1] = (chosen_left, chosen_right)
        positive_counts = [
            (left_mol.GetNumAtoms() - len(left)) + (right_mol.GetNumAtoms() - len(right))
            for left, right in combinations
        ]
        mapping_rows.append({
            "dataset": pair.dataset, "component_id": int(pair.component_id),
            "left_unique_matched_atom_sets": len(left_sets), "right_unique_matched_atom_sets": len(right_sets),
            "mapping_combinations_evaluated": len(combinations),
            "left_matches_truncated": left_truncated, "right_matches_truncated": right_truncated,
            "combination_truncated": combination_truncated,
            "chosen_left_recovered": chosen_left in left_sets, "chosen_right_recovered": chosen_right in right_sets,
            "positive_atom_count_min": int(min(positive_counts)), "positive_atom_count_max": int(max(positive_counts)),
            "positive_atom_set_ambiguous": len(left_sets) > 1 or len(right_sets) > 1,
        })
        if len(left_sets) > 1 or len(right_sets) > 1:
            original_labels = np.concatenate([
                ~np.isin(np.arange(left_mol.GetNumAtoms()), list(chosen_left)),
                ~np.isin(np.arange(right_mol.GetNumAtoms()), list(chosen_right)),
            ])
            for representation in representations:
                for predictor in predictors:
                    scores = np.concatenate([
                        seed_mean_arrays[(pair.dataset, representation, predictor, "trained", pair.left_id)],
                        seed_mean_arrays[(pair.dataset, representation, predictor, "trained", pair.right_id)],
                    ])
                    original_nap = (grouped_average_precision(original_labels, scores) - original_labels.mean()) / (1.0 - original_labels.mean())
                    values = []
                    for left_set, right_set in combinations:
                        labels = np.concatenate([
                            ~np.isin(np.arange(left_mol.GetNumAtoms()), list(left_set)),
                            ~np.isin(np.arange(right_mol.GetNumAtoms()), list(right_set)),
                        ])
                        if 0 < labels.sum() < len(labels):
                            prevalence = labels.mean()
                            values.append((grouped_average_precision(labels, scores) - prevalence) / (1.0 - prevalence))
                    if values:
                        mapping_ap_rows.append({
                            "dataset": pair.dataset, "component_id": int(pair.component_id),
                            "representation": representation, "predictor": predictor,
                            "mapping_combinations_evaluated": len(values), "chosen_mapping_nap": original_nap,
                            "minimum_mapping_nap": float(min(values)), "maximum_mapping_nap": float(max(values)),
                            "mapping_nap_range": float(max(values) - min(values)),
                        })
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(output / "mcs_mapping_ambiguity.csv", index=False)
    mapping_ap = pd.DataFrame(mapping_ap_rows)
    mapping_ap.to_csv(output / "mcs_mapping_nap_sensitivity.csv", index=False)
    mapping_summary = {
        "pairs": int(len(mapping)),
        "pairs_with_multiple_valid_matched_atom_sets": int(mapping.positive_atom_set_ambiguous.sum()),
        "fraction_with_multiple_valid_matched_atom_sets": float(mapping.positive_atom_set_ambiguous.mean()),
        "pairs_with_truncated_enumeration": int((mapping.left_matches_truncated | mapping.right_matches_truncated | mapping.combination_truncated).sum()),
        "chosen_mapping_recovered_all": bool(mapping.chosen_left_recovered.all() and mapping.chosen_right_recovered.all()),
        "median_nap_range_when_ambiguous": float(mapping_ap.mapping_nap_range.median()) if len(mapping_ap) else 0.0,
        "q95_nap_range_when_ambiguous": float(mapping_ap.mapping_nap_range.quantile(0.95)) if len(mapping_ap) else 0.0,
        "scope": "Automorphic/substructure alternatives to the stored matched atom subgraph; this does not enumerate chemically distinct maximum common subgraphs absent from the stored match.",
    }
    (output / "mcs_mapping_summary.json").write_text(json.dumps(mapping_summary, indent=2) + "\n", encoding="utf-8")

    primary["interval_scope"] = "All reported bootstrap intervals resample targets and components conditional on the saved model initializations, label permutations, neighborhoods and observed-label-selected cliff set."
    primary["crem_lime_status"] = "excluded from primary inference; retained only as a numerical and held-out-fidelity failure-case sensitivity"
    (output / "crem_mean_primary_summary.json").write_text(json.dumps(primary, indent=2) + "\n", encoding="utf-8")

    output_files = sorted(path for path in output.iterdir() if path.is_file() and path.name not in {"manifest.json", "evaluation_summary.json", "DATA_DICTIONARY.md"})
    manifest = {
        "run_id": cfg["run_id"],
        "status": "pass",
        "config_sha256": sha256(args.config),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "output_sha256": {path.name: sha256(path) for path in output_files},
        "software": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "rdkit": rdBase.rdkitVersion,
        },
        "validation": {
            "confirmation_pairs": len(pairs), "model_pair_null_rows": len(model_null),
            "generator_pair_null_rows": len(generator_null), "target_configuration_cells": len(cell),
            "all_primary_values_finite": bool(np.isfinite(component[["delta_observed_nap", "delta_null_adjusted_nap"]].to_numpy()).all()),
            "all_chosen_mappings_recovered": mapping_summary["chosen_mapping_recovered_all"],
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    evaluation = {
        "status": "pass",
        "question": "Do the estimator, empirical-null, heterogeneity, predictor-competence and structural-proxy audits change the evidence boundary?",
        "comparison_baselines": ["label-permuted CReM mean", "pair-specific tie-preserving empirical null", "generator-only scores"],
        "claim_update": "CReM-LIME is excluded from primary inference; the remaining claim is conditional, post-confirmation and limited to unsigned CReM-mean structural-change localization.",
        "comparability": "Model/data/neighborhood caches are fixed. Rationale and empirical-null variations are explicit sensitivity estimands rather than replacements for the historical analysis.",
        "next_action": "Revise the manuscript and supplement using these outputs; retain any contradictory or wide-interval result.",
    }
    (output / "evaluation_summary.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    (output / "DATA_DICTIONARY.md").write_text(
        "# Round-2 analysis data dictionary\n\n"
        "All files are derived from frozen workspace rows. `model_empirical_null_pair_metrics.csv` and "
        "`generator_empirical_null_pair_metrics.csv` preserve the observed positive count and exact score-tie structure in 500 deterministic label permutations per row. "
        "`crem_mean_empirical_null_component_deltas.csv` averages seed-level metrics within model variant before differencing. "
        "`crem_mean_configuration_summary.csv` reports paired unadjusted, empirical-null-adjusted and strict-unmatched point estimates for the six CReM-mean configurations; inferential intervals are reserved for the prespecified aggregate and sensitivity estimands. "
        "`crem_mean_target_configuration_effects.csv` contains continuous cell effects used in the method-of-moments heterogeneity model. "
        "`mcs_mapping_ambiguity.csv` enumerates atom-set alternatives for the stored matched subgraph; it is not a claim that every chemically distinct maximum common subgraph was recovered.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "output": str(output), **manifest["validation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
