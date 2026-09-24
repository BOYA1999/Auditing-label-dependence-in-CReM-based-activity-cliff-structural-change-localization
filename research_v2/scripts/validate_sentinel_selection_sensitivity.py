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


def expected_counts(spec):
    legacy = {
        "preparation": {
            "targets": 6,
            "source_test_components": 140,
            "selected_components": 139,
            "source_components_outside_frozen_analysis": 1,
            "localization_eligible_components": 139,
            "changed_from_original": 76,
            "unchanged_from_original": 63,
            "evaluation_endpoints": 278,
            "seed42_heads": 72,
        },
        "analysis": {
            "targets": 6,
            "alternative_eligible_components": 139,
            "paired_comparison_components": 138,
            "unpaired_alternative_components": 1,
            "alternative_component_configuration_rows": 1668,
            "paired_component_configuration_rows": 1656,
            "factorial_configurations": 12,
        },
    }
    configured = spec.get("expected_counts", {})
    return configured.get("preparation", legacy["preparation"]), configured.get("analysis", legacy["analysis"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    spec = cfg["sentinel_selection_sensitivity"]
    experiment = resolve(spec["experiment_root"])
    output = resolve(spec["analysis_output"])
    targets = spec["representative_targets"]
    representations = list(cfg["representations"])
    preparation_expected, analysis_expected = expected_counts(spec)
    checks = {}

    selected = pd.read_csv(experiment / "prepared/selected_pairs_all.csv")
    source_pairs = pd.read_csv(resolve(spec["inputs"]["pair_cache"]))
    recomputed_all = []
    candidate = source_pairs[source_pairs.dataset.isin(targets) & (source_pairs.component_split == "test")]
    for (dataset, component_id), frame in candidate.groupby(["dataset", "component_id"], sort=True):
        center = float(frame.potency_difference.median())
        row = frame.assign(distance=(frame.potency_difference - center).abs()).sort_values(
            ["distance", "left_index", "right_index", "pair_index"], kind="mergesort"
        ).iloc[0]
        recomputed_all.append((dataset, int(component_id), int(row.pair_index)))
    observed = list(zip(selected.dataset, selected.component_id.astype(int), selected.pair_index.astype(int)))
    frozen_keys = set(zip(selected.dataset, selected.component_id.astype(int)))
    recomputed = [row for row in recomputed_all if (row[0], row[1]) in frozen_keys]
    checks["selection_rule_exact"] = observed == recomputed
    outside = pd.read_csv(experiment / "prepared/source_components_outside_frozen_analysis.csv")
    eligible = selected[selected.valid_normalized_ap_prevalence.astype(bool)].copy()
    checks["target_count"] = len(targets) == preparation_expected["targets"]
    checks["source_test_component_count"] = len(recomputed_all) == preparation_expected["source_test_components"]
    checks["frozen_population_exclusion_count"] = len(outside) == preparation_expected["source_components_outside_frozen_analysis"]
    checks["frozen_population_exclusion_reasons"] = bool(
        len(outside) == 0 or outside.original_rationale_failure_reason.fillna("").eq("empty_mces").all()
    )
    checks["selected_row_count"] = len(selected) == preparation_expected["selected_components"]
    checks["selected_key_unique"] = not selected.duplicated(["dataset", "component_id"]).any()
    checks["localization_eligible_count"] = len(eligible) == preparation_expected["localization_eligible_components"]
    checks["changed_and_unchanged_counts"] = (
        int(selected.same_as_original.astype(bool).sum()) == preparation_expected["unchanged_from_original"]
        and int((~selected.same_as_original.astype(bool)).sum()) == preparation_expected["changed_from_original"]
    )

    evaluation = pd.read_csv(experiment / "prepared/evaluation_molecules.csv")
    endpoint_ids = {
        f"{row.dataset}:{int(index)}"
        for row in eligible.itertuples(index=False)
        for index in [row.left_index, row.right_index]
    }
    checks["evaluation_row_count"] = len(evaluation) == preparation_expected["evaluation_endpoints"]
    checks["evaluation_key_unique"] = not evaluation.row_id.duplicated().any()
    checks["evaluation_exact_endpoint_set"] = set(evaluation.row_id.astype(str)) == endpoint_ids
    checks["evaluation_all_test"] = set(evaluation.component_split) == {"test"}

    feasibility = json.loads((experiment / "prepared/feasibility.json").read_text(encoding="utf-8"))
    checks["feasibility_pass"] = feasibility["status"] == "pass"
    checks["feasibility_count_validation"] = all(
        feasibility.get("count_validation", {}).get("checks", {"legacy": True}).values()
    )
    checks["heldout_leakage_zero"] = feasibility["library_all_heldout_overlap"] == 0 and feasibility["library_selected_endpoint_overlap"] == 0
    checks["head_crossing_count"] = (
        feasibility["seed42_heads"]
        == feasibility["expected_seed42_heads"]
        == preparation_expected["seed42_heads"]
    )

    counterfactual_summary = json.loads((experiment / "counterfactuals/summary.json").read_text(encoding="utf-8"))
    masks = pd.read_csv(experiment / "counterfactuals/mask_coverage.csv")
    scored = pd.read_csv(experiment / "counterfactuals/scored_mutants_model.csv")
    unique = pd.read_csv(experiment / "counterfactuals/unique_scored_mutants_model.csv")
    checks["counterfactuals_success"] = (
        counterfactual_summary["status"] == "success"
        and counterfactual_summary["unexpected_generation_failures"] == 0
        and counterfactual_summary["evaluation_molecules"] == len(evaluation)
    )
    checks["mask_rows_and_keys"] = len(masks) == counterfactual_summary["candidate_masks"] and not masks.duplicated(["row_id", "mask_id"]).any()
    checks["scored_row_count"] = len(scored) == counterfactual_summary["scored_mutant_rows"]
    checks["scored_parent_subset"] = set(scored.row_id.astype(str)) <= endpoint_ids
    checks["unique_mutant_rows_and_keys"] = (
        len(unique) == counterfactual_summary["unique_scored_mutants"]
        and not unique.row_id.duplicated().any()
        and not unique.canonical_smiles.duplicated().any()
    )
    checks["scored_mutant_lookup_complete"] = set(scored.mutant_smiles.astype(str)) <= set(unique.canonical_smiles.astype(str))

    representation_manifest = json.loads((experiment / "mutant_representations/manifest.json").read_text(encoding="utf-8"))
    checks["representation_row_count"] = representation_manifest["rows"] == len(unique)
    for arm in representations:
        path = experiment / f"mutant_representations/{arm}.npz"
        with np.load(path, allow_pickle=False) as data:
            row_ids = data["row_id"].astype(str)
            values = data["x"]
            checks[f"{arm}_row_order_exact"] = np.array_equal(row_ids, unique.row_id.astype(str).to_numpy())
            checks[f"{arm}_shape_exact"] = list(values.shape) == representation_manifest["arms"][arm]["shape"]
            checks[f"{arm}_finite"] = bool(np.isfinite(values).all())
        checks[f"{arm}_hash_exact"] = sha256(path) == representation_manifest["arms"][arm]["sha256"]
    reuse_summary = json.loads((experiment / "new_mutants/summary.json").read_text(encoding="utf-8"))
    checks["representation_reuse_partition"] = (
        reuse_summary["reused_unique_mutants"] + reuse_summary["new_unique_mutants"] == len(unique)
        and reuse_summary["alternate_unique_mutants"] == len(unique)
    )

    explanation_summary = json.loads((experiment / "explanations/summary.json").read_text(encoding="utf-8"))
    explanations = pd.read_csv(experiment / "explanations/pair_explanations.csv")
    explanation_key = ["dataset", "component_id", "representation", "predictor", "explainer", "seed", "variant"]
    expected_explanation_rows = (
        preparation_expected["localization_eligible_components"]
        * len(representations)
        * len(spec["predictors"])
        * len(cfg["explainers"])
        * len(spec["variants"])
    )
    checks["explanations_pass"] = explanation_summary["status"] == "pass"
    checks["explanation_row_count"] = len(explanations) == explanation_summary["expected_pair_rows"] == expected_explanation_rows
    checks["explanation_key_unique"] = not explanations.duplicated(explanation_key).any()
    checks["explanation_crossing_exact"] = (
        set(explanations.dataset) == set(targets)
        and set(explanations.representation) == set(representations)
        and set(explanations.predictor) == set(spec["predictors"])
        and set(explanations.explainer) == set(cfg["explainers"])
        and set(explanations.seed) == {42}
        and set(explanations.variant) == set(spec["variants"])
    )
    checks["explanation_nap_finite"] = bool(np.isfinite(explanations.normalized_ap_lift).all())
    expected_randomized_models = len(targets) * len(representations) * len(spec["predictors"])
    checks["randomized_models_nonconstant"] = (
        explanation_summary["nonconstant_randomized_local_models"]
        == explanation_summary["randomized_local_models"]
        == expected_randomized_models
    )

    analysis_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    alternative = pd.read_csv(output / "alternative_component_configuration_metrics.csv")
    paired = pd.read_csv(output / "paired_selection_comparison.csv")
    analysis_key = ["dataset", "component_id", "representation", "predictor", "explainer"]
    checks["analysis_pass"] = analysis_summary["status"] == "pass"
    checks["analysis_count_validation"] = all(
        analysis_summary.get("count_validation", {}).get("checks", {"legacy": True}).values()
    )
    checks["alternative_rows_and_keys"] = (
        len(alternative) == analysis_expected["alternative_component_configuration_rows"]
        and not alternative.duplicated(analysis_key).any()
    )
    checks["paired_rows_and_keys"] = (
        len(paired) == analysis_expected["paired_component_configuration_rows"]
        and not paired.duplicated(analysis_key).any()
    )
    checks["analysis_factorial_count"] = analysis_summary["factorial_configurations"] == analysis_expected["factorial_configurations"]
    checks["analysis_values_finite"] = bool(np.isfinite(paired[["delta_nap_alternative", "delta_nap_original", "selection_change_in_delta_nap"]].to_numpy()).all())

    evidence_files = [
        args.config.resolve(),
        experiment / "prepared/feasibility.json",
        experiment / "prepared/selected_pairs_all.csv",
        experiment / "prepared/source_components_outside_frozen_analysis.csv",
        experiment / "counterfactuals/cache_seed_audit.csv",
        experiment / "counterfactuals/summary.json",
        experiment / "new_mutants/summary.json",
        experiment / "mutant_representations/manifest.json",
        experiment / "explanations/summary.json",
        experiment / "explanations/pair_explanations.csv",
        output / "summary.json",
        output / "paired_selection_comparison.csv",
        output / "rationale_overlap.csv",
    ]
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "run_id": cfg["run_id"],
        "status": status,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "evidence_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in evidence_files},
        "script_sha256": {
            path.name: sha256(path)
            for path in [
                ROOT / "research_v2/scripts/prepare_sentinel_selection_sensitivity.py",
                ROOT / "research_v2/scripts/seed_sentinel_counterfactual_cache.py",
                ROOT / "research_v2/scripts/assemble_sentinel_mutant_representations.py",
                ROOT / "research_v2/scripts/analyze_sentinel_selection_sensitivity.py",
                ROOT / "research_v2/scripts/run_sentinel_selection_sensitivity.py",
                ROOT / "research_v2/scripts/validate_sentinel_selection_sensitivity.py",
            ]
        },
    }
    (output / "strict_validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
