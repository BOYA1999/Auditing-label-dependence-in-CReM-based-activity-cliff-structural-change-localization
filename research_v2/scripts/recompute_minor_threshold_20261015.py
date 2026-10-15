import json
import shutil
from fractions import Fraction
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

import analyze_round2_revision as analysis
import analyze_major_revision as major
import make_round2_figures as figures


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_round2_v1"
OUT = ROOT / "research_v2/artifacts/analysis/minor_revision_20261015/threshold"
KEYS = ["dataset", "representation", "predictor"]


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / "research_v2/configs/round2_revision_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    source = json.loads(analysis.resolve(cfg["source_config"]).read_text(encoding="utf-8"))
    prediction_path = analysis.resolve(cfg["inputs"]["pair_predictions"])
    all_predictions = pd.read_csv(prediction_path)
    predictions = all_predictions[all_predictions.dataset.isin(source["confirmation_targets"])]
    calibration = predictions[(predictions.component_split == "calibration") & (predictions.variant == "trained")]
    seed, quality = analysis.calibration_prediction_quality(calibration, cfg["calibration_gate"])
    seed.to_csv(OUT / "calibration_seed_direction_counts.csv", index=False)
    quality.to_csv(OUT / "calibration_prediction_quality.csv", index=False)
    old_quality = pd.read_csv(OLD / "calibration_prediction_quality.csv")
    old_gate = dict(cfg["calibration_gate"], minimum_direction_accuracy=0.6666666667)
    _, old_recomputed = analysis.calibration_prediction_quality(calibration, old_gate)
    pd.testing.assert_frame_equal(old_recomputed, old_quality, check_exact=False, atol=1e-14, rtol=0)
    pd.testing.assert_frame_equal(quality.drop(columns="eligible"), old_quality.drop(columns="eligible"), check_exact=False, atol=1e-14, rtol=0)
    changed = quality.merge(old_quality[KEYS + ["eligible"]], on=KEYS, suffixes=("", "_old"))
    changed = changed[changed.eligible != changed.eligible_old]
    expected = {
        ("CHEMBL218_EC50", "molformer", "mlp"): [(11, 11, 18), (23, 12, 18), (42, 12, 18)],
        ("CHEMBL238_Ki", "morgan", "mlp"): [(11, 12, 18), (23, 11, 18), (42, 12, 18)],
        ("CHEMBL238_Ki", "morgan", "xgboost"): [(11, 12, 18), (23, 12, 18), (42, 13, 18)],
        ("CHEMBL244_Ki", "molformer", "mlp"): [(11, 33, 51), (23, 34, 51), (42, 39, 51)],
    }
    assert set(changed[KEYS].itertuples(index=False, name=None)) == set(expected)
    assert changed.eligible.all() and not changed.eligible_old.any()
    for key, expected_counts in expected.items():
        selected = seed.set_index(KEYS).loc[key].sort_values("seed")
        counts = [(int(r.seed), int(r.correct_direction_pairs), int(r.pairs)) for r in selected.itertuples()]
        assert counts == expected_counts
        assert median(Fraction(correct, total) for _, correct, total in counts) == Fraction(2, 3)
    changed.to_csv(OUT / "boundary_regression_cases.csv", index=False)
    cell = pd.read_csv(OLD / "crem_mean_target_configuration_effects.csv")
    competence = cell.merge(quality, on=KEYS, validate="one_to_one")
    competence.to_csv(OUT / "prediction_competence_localization_cells.csv", index=False)
    component_path = OLD / "crem_mean_empirical_null_component_deltas.csv"
    components = pd.read_csv(component_path)
    boundary = components[components.rationale == "boundary_extended"]
    old_summary = json.loads((OLD / "prediction_competence_summary.json").read_text(encoding="utf-8"))
    results = {}
    for label, eligible in [("old", old_quality), ("corrected", quality)]:
        subset = boundary.merge(eligible[eligible.eligible][KEYS], on=KEYS, validate="many_to_one")
        result, boot = analysis.primary_from_components(subset, "delta_null_adjusted_nap", cfg["bootstrap_draws"], cfg["bootstrap_seed"], "calibration-qualified")
        results[label] = result
        if label == "old":
            np.testing.assert_allclose([result["estimate"], *result["ci95"], *result["ci90"]], [old_summary["eligible_primary"]["estimate"], *old_summary["eligible_primary"]["ci95"], *old_summary["eligible_primary"]["ci90"]], atol=2e-15, rtol=0)
            np.testing.assert_allclose(boot, pd.read_csv(OLD / "prediction_competence_bootstrap.csv").calibration_qualified_estimate, atol=2e-15, rtol=0)
        else:
            subset.to_csv(OUT / "calibration_qualified_component_deltas.csv", index=False)
            pd.DataFrame({"draw": np.arange(len(boot)), "calibration_qualified_estimate": boot}).to_csv(OUT / "prediction_competence_bootstrap.csv", index=False)
    summary = dict(old_summary, gate=cfg["calibration_gate"], eligible_target_configuration_cells=int(quality.eligible.sum()), eligible_targets=int(quality.loc[quality.eligible, "dataset"].nunique()), eligible_primary=results["corrected"])
    write_json("prediction_competence_summary.json", summary)
    rep = {"molformer": "MoLFormer", "morgan": "Morgan", "unimol": "Uni-Mol"}
    pred = {"mlp": "MLP", "xgboost": "XGBoost"}
    rows = []
    for (r, p), group in quality.groupby(["representation", "predictor"], sort=True):
        rows.append([rep[r], pred[p], f"{group.calibration_pairs.median():.1f}", f"{group.calibration_spearman.median():.3f}", f"{group.calibration_direction_accuracy.median():.3f}", f"{group.calibration_tie_fraction.median():.3f}", f"{group.eligible.sum()}/27"])
    pd.DataFrame(rows, columns=["Representation", "Predictor", "Pairs", "Spearman", "Direction", "Tie fraction", "Eligible targets"]).to_csv(OUT / "table_s07.csv", index=False)
    eligible = results["corrected"]
    s8 = pd.read_csv(ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_supplement_tables_v1/table_s08.csv", dtype=str)
    s8.loc[0, "Estimate/slope"] = f"{eligible['estimate']:.6f}"
    s8.loc[0, "95% CI"] = f"[{eligible['ci95'][0]:.6f}, {eligible['ci95'][1]:.6f}]"
    s8.to_csv(OUT / "table_s08.csv", index=False)
    experiment = ROOT / "research_v2/artifacts/experiment"
    audit_paths = [p for p in experiment.glob("xai_sanity_cliffs*/**/pair_predictions.csv") if "units" not in p.parts and "smoke" not in str(p)]
    audit_rows, near_rows = [], []
    for path in sorted(audit_paths):
        frame = pd.read_csv(path)
        x = frame.predicted_delta.to_numpy(float)
        near = (np.abs(x) > 0) & (np.abs(x) <= 1e-12)
        audit_rows.append({"source": path.relative_to(ROOT).as_posix(), "rows": len(frame), "exact_zero_ties": int((x == 0).sum()), "nonzero_within_1e_12": int(near.sum()), "minimum_absolute_nonzero_delta": float(np.abs(x[x != 0]).min()), "sha256": analysis.sha256(path)})
        if near.any():
            near_rows.append(frame.loc[near].assign(source=path.relative_to(ROOT).as_posix()))
    pd.DataFrame(audit_rows).to_csv(OUT / "near_zero_tie_audit_by_source.csv", index=False)
    pd.concat(near_rows, ignore_index=True).to_csv(OUT / "nonzero_tolerance_tie_records.csv", index=False) if near_rows else pd.DataFrame(columns=["source", *all_predictions.columns]).to_csv(OUT / "nonzero_tolerance_tie_records.csv", index=False)
    assert sum(row["nonzero_within_1e_12"] for row in audit_rows) == 0
    tie_rows = []
    for key, group in predictions[(predictions.component_split == "test") & (predictions.variant == "label_permuted")].groupby(KEYS + ["seed"], sort=True):
        true, predicted = group.true_delta.to_numpy(float), group.predicted_delta.to_numpy(float)
        tied = predicted == 0.0
        tie_rows.append(dict(zip(KEYS + ["seed"], key)) | {"pairs": len(group), "exact_zero_ties": int(tied.sum()), "tie_fraction": float(tied.mean()), "strict_direction_accuracy": float(np.mean(np.sign(true) == np.sign(predicted))), "non_tie_direction_accuracy": float(np.mean(np.sign(true[~tied]) == np.sign(predicted[~tied]))) if (~tied).any() else np.nan, "sign_flipped_strict_accuracy": float(np.mean(np.sign(true) == np.sign(-predicted))), "median_absolute_predicted_delta": float(np.median(np.abs(predicted)))})
    ties = pd.DataFrame(tie_rows)
    pd.testing.assert_frame_equal(ties, pd.read_csv(OLD / "permuted_direction_tie_audit_by_model.csv"), check_exact=False, atol=1e-14, rtol=0)
    ties.to_csv(OUT / "permuted_direction_tie_audit_by_model.csv", index=False)
    test = predictions[predictions.component_split == "test"]
    model_metrics = test.groupby(KEYS + ["seed", "variant"], sort=True).apply(major.prediction_metrics, include_groups=False).reset_index()
    saved_model = pd.read_csv(ROOT / "research_v2/artifacts/analysis/xai_sanity_cliffs_major_revision_v1/prediction_model_cell_metrics.csv")
    pd.testing.assert_frame_equal(model_metrics, saved_model, check_exact=False, atol=1e-12, rtol=0)
    slope = model_metrics.groupby(["dataset", "variant"], as_index=False).calibration_slope.median().groupby("variant").calibration_slope.median()
    write_json("slope_definition_audit.json", {"formula": "predicted_delta = intercept + slope * observed_delta + error", "implementation": "numpy.polyfit(true_delta, predicted_delta, 1)", "aggregation": "median over model configurations and seeds within each dataset, then median over datasets", "target_balanced_median_slope": slope.to_dict(), "saved_model_metrics_reproduced": True})
    for name in ["crem_mean_target_configuration_effects.csv", "hierarchical_heterogeneity.json"]:
        shutil.copy2(OLD / name, OUT / name)
    figures.ROUND2 = OUT
    figures.FIGURES = OUT / "figures"
    figures.LATEX_FIGURES = OUT / "latex_figures"
    figures.SOURCE = OUT / "figure_source_data"
    figures.SOURCE.mkdir(exist_ok=True)
    figures.figure4()
    write_json("verification_summary.json", {"status": "PASS", "old_result_reproduced": True, "old_bootstrap_draws_reproduced": 5000, "corrected_gate": cfg["calibration_gate"], "boundary_cases": len(changed), "eligible_cells_old": int(old_quality.eligible.sum()), "eligible_cells_corrected": int(quality.eligible.sum()), "total_cells": len(quality), "results": results, "seed_aggregation": "Median of exact correct-direction count / pair count across the same three seeds; Spearman median and maximum pair count unchanged", "near_zero_source_files": len(audit_rows), "near_zero_rows_checked": sum(row["rows"] for row in audit_rows), "nonzero_within_1e_12": sum(row["nonzero_within_1e_12"] for row in audit_rows), "unchanged_checks": ["all calibration metrics apart from eligibility", "all 162 continuous-association inputs", "permuted direction tie diagnostics", "all model-cell predictive metrics"], "analysis_boundary": "Frozen-prediction and saved component-score statistical replay; no model retraining or neighborhood regeneration"})
    inputs = [cfg_path, Path(__file__), Path(analysis.__file__), Path(major.__file__), Path(figures.__file__), component_path, OLD / "prediction_competence_summary.json", OLD / "prediction_competence_bootstrap.csv", *audit_paths]
    write_json("manifest.json", {"inputs": {p.relative_to(ROOT).as_posix(): analysis.sha256(p) for p in inputs}, "outputs": {p.relative_to(OUT).as_posix(): analysis.sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file() and "backups" not in p.parts and p.name != "manifest.json"}})
    print(json.dumps({"eligible_cells": int(quality.eligible.sum()), "result": eligible, "near_zero_sources": len(audit_rows), "near_zero_count": 0}, indent=2))


if __name__ == "__main__":
    main()
