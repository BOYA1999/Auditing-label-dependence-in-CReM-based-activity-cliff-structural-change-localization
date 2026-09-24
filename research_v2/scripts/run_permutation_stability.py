import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import Chem
from sklearn.metrics import mean_squared_error

from score_factorial_explanations import (
    add_pair_rows,
    build_lime_neighborhoods,
    lime_arrays,
    load_npz,
    mean_replacement_arrays,
    predict as score_predict,
)
from train_common_heads import load_representation, model_from_config
from train_mlp_heads import fit as fit_mlp, predict as predict_mlp


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(path, message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def permutation_seed(cfg, target, permutation_id):
    target_index = cfg["randomization"]["source_target_order"].index(target)
    return int(cfg["randomization"]["base_seed"] + 1000 * target_index + permutation_id)


def add_design_fields(frame, cfg, permutation_id, seed):
    frame = frame.copy()
    frame["seed"] = int(cfg["model_initialization_seed"])
    frame["model_initialization_seed"] = int(cfg["model_initialization_seed"])
    frame["permutation_id"] = int(permutation_id)
    frame["permutation_seed"] = int(seed)
    frame["variant"] = "label_permuted"
    return frame


def make_labels(cfg, molecules, target, permutation_id, labels_dir):
    seed = permutation_seed(cfg, target, permutation_id)
    path = labels_dir / f"{target}--perm{permutation_id}.csv"
    target_mask = (molecules.dataset == target).to_numpy()
    eligible = target_mask & molecules.component_split.isin(["train", "isolated", "calibration"]).to_numpy()
    indices = np.flatnonzero(eligible)
    observed = molecules.y.to_numpy(dtype=np.float32)
    permuted = np.random.default_rng(seed).permutation(observed[indices])
    frame = pd.DataFrame({
        "row_id": molecules.row_id.iloc[indices].astype(str).to_numpy(),
        "dataset": target,
        "permutation_id": int(permutation_id),
        "permutation_seed": seed,
        "model_initialization_seed": int(cfg["model_initialization_seed"]),
        "observed_y": observed[indices].astype(float),
        "permuted_y": permuted.astype(float),
    })
    frame.to_csv(path, index=False)
    return path, indices, permuted


def valid_rationale_pairs(evaluation, pairs):
    atom_counts = {
        row.row_id: Chem.MolFromSmiles(row.canonical_smiles).GetNumAtoms()
        for row in evaluation.itertuples(index=False)
    }
    keep = []
    audit = []
    for pair in pairs.itertuples(index=False):
        left_id = f"{pair.dataset}:{int(pair.left_index)}"
        right_id = f"{pair.dataset}:{int(pair.right_index)}"
        positives = len(json.loads(pair.rationale_left)) + len(json.loads(pair.rationale_right))
        total = atom_counts[left_id] + atom_counts[right_id]
        valid = 0 < positives < total
        keep.append(valid)
        audit.append({
            "dataset": pair.dataset,
            "component_id": int(pair.component_id),
            "component_split": pair.component_split,
            "positive_atoms": positives,
            "total_atoms": total,
            "valid_normalized_ap_prevalence": valid,
        })
    return pairs[np.asarray(keep)].copy(), pd.DataFrame(audit)


def unit_key(target, representation, predictor, permutation_id):
    return f"{target}--{representation}--{predictor}--perm{permutation_id}"


def unit_complete(unit_dir, config_hash):
    manifest_path = unit_dir / "unit_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "pass" or manifest.get("config_sha256") != config_hash:
            return False
        for name, digest in manifest["output_hashes"].items():
            path = unit_dir / name
            if not path.exists() or sha256(path) != digest:
                return False
        model_path = ROOT / manifest["model"]["path"]
        return model_path.exists() and sha256(model_path) == manifest["model"]["sha256"]
    except (KeyError, json.JSONDecodeError, OSError):
        return False


def write_unit(unit_dir, tables, model_row, config_hash, source):
    unit_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name, frame in tables.items():
        path = unit_dir / name
        frame.to_csv(path, index=False)
        output_hashes[name] = sha256(path)
    manifest = {
        "status": "pass",
        "source": source,
        "config_sha256": config_hash,
        "model": model_row,
        "output_hashes": output_hashes,
    }
    (unit_dir / "unit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def reuse_source_unit(cfg, paths, target, representation, predictor, permutation_id, seed, unit_dir, config_hash):
    filters = {
        "dataset": target,
        "representation": representation,
        "predictor": predictor,
        "seed": int(cfg["model_initialization_seed"]),
        "variant": "label_permuted",
    }

    def subset(path):
        frame = pd.read_csv(path)
        mask = np.ones(len(frame), dtype=bool)
        for column, value in filters.items():
            if column in frame.columns:
                mask &= frame[column].to_numpy() == value
        result = frame[mask].copy()
        if result.empty:
            raise ValueError(f"missing reusable source rows: {path} {filters}")
        return add_design_fields(result, cfg, permutation_id, seed)

    source_heads = paths["source_heads"]
    source_explanations = paths["source_explanations"]
    manifest = subset(source_heads / "model_manifest.csv")
    if len(manifest) != 1:
        raise ValueError(f"expected one reusable model, found {len(manifest)}")
    source_model = manifest.iloc[0]
    if int(source_model.permutation_seed) != seed:
        raise ValueError("reused source permutation seed does not match the frozen formula")
    model_path = ROOT / source_model.path
    if sha256(model_path) != source_model.sha256:
        raise ValueError(f"reused model hash mismatch: {model_path}")
    model_row = source_model.to_dict()
    model_row.update({
        "seed": int(cfg["model_initialization_seed"]),
        "model_initialization_seed": int(cfg["model_initialization_seed"]),
        "permutation_id": int(permutation_id),
        "permutation_seed": seed,
        "source": "reused_existing",
    })
    tables = {
        "predictions.csv": subset(source_heads / "predictions.csv"),
        "prediction_metrics.csv": subset(source_heads / "prediction_metrics.csv"),
        "pair_predictions.csv": subset(source_heads / "pair_predictions.csv"),
        "pair_explanations.csv": subset(source_explanations / "pair_explanations.csv"),
        "local_fidelity.csv": subset(source_explanations / "local_fidelity.csv"),
        "model_output_audit.csv": subset(source_explanations / "model_output_audit.csv"),
    }
    write_unit(unit_dir, tables, model_row, config_hash, "reused_existing")


def train_and_score_unit(
    cfg, molecules, pairs, valid_pairs, unique, original_ids, original_position,
    original_morgan, mutant_morgan, values, mutant_values, target_geometry,
    representation, target, predictor, permutation_id, seed, label_indices,
    permuted_labels, unit_dir, output, config_hash, device,
):
    fit_y = molecules.y.to_numpy(dtype=np.float32).copy()
    fit_y[label_indices] = permuted_labels
    target_mask = (molecules.dataset == target).to_numpy()
    train_mask = target_mask & molecules.component_split.isin(["train", "isolated"]).to_numpy()
    calibration_mask = target_mask & (molecules.component_split == "calibration").to_numpy()
    target_indices = np.flatnonzero(target_mask)
    init_seed = int(cfg["model_initialization_seed"])
    if predictor == "xgboost":
        model = model_from_config(cfg["common_head"], init_seed)
        model.fit(
            values[train_mask], fit_y[train_mask],
            eval_set=[(values[calibration_mask], fit_y[calibration_mask])], verbose=False,
        )
        bundle = None
        best_step = int(model.best_iteration)
    else:
        model, bundle, best_step, _ = fit_mlp(
            values, fit_y, train_mask, calibration_mask, cfg["mlp_head"], init_seed, device,
        )
    target_predictions = model.predict(values[target_indices]) if predictor == "xgboost" else predict_mlp(
        model, bundle, values[target_indices], device,
    )
    if not np.isfinite(target_predictions).all():
        raise ValueError("non-finite target predictions")
    prediction_map = {}
    prediction_rows = []
    observed_y = molecules.y.to_numpy(dtype=np.float32)
    for index, value in zip(target_indices, target_predictions):
        molecule_index = int(molecules.molecule_index.iloc[index])
        prediction_map[molecule_index] = float(value)
        prediction_rows.append({
            "row_id": original_ids[index], "dataset": target,
            "component_split": molecules.component_split.iloc[index],
            "representation": representation, "predictor": predictor,
            "seed": init_seed, "model_initialization_seed": init_seed,
            "permutation_id": permutation_id, "permutation_seed": seed,
            "variant": "label_permuted", "observed_y": float(observed_y[index]),
            "prediction": float(value),
        })
    metric_rows = []
    for split in ["calibration", "test"]:
        split_mask = target_mask & (molecules.component_split == split).to_numpy()
        split_predictions = model.predict(values[split_mask]) if predictor == "xgboost" else predict_mlp(
            model, bundle, values[split_mask], device,
        )
        metric_rows.append({
            "dataset": target, "representation": representation, "predictor": predictor,
            "seed": init_seed, "model_initialization_seed": init_seed,
            "permutation_id": permutation_id, "permutation_seed": seed,
            "variant": "label_permuted", "split": split,
            "rmse_against_observed": float(mean_squared_error(observed_y[split_mask], split_predictions) ** 0.5),
            "prediction_std": float(np.std(split_predictions)),
            "prediction_range": float(np.ptp(split_predictions)), "n": int(split_mask.sum()),
        })
    observed_map = molecules[molecules.dataset == target].set_index("molecule_index").y.to_dict()
    pair_rows = []
    for pair in pairs[pairs.dataset == target].itertuples(index=False):
        true_delta = float(observed_map[pair.left_index] - observed_map[pair.right_index])
        predicted_delta = prediction_map[int(pair.left_index)] - prediction_map[int(pair.right_index)]
        pair_rows.append({
            "dataset": target, "component_id": int(pair.component_id),
            "component_split": pair.component_split, "left_index": int(pair.left_index),
            "right_index": int(pair.right_index), "representation": representation,
            "predictor": predictor, "seed": init_seed, "model_initialization_seed": init_seed,
            "permutation_id": permutation_id, "permutation_seed": seed,
            "variant": "label_permuted", "true_delta": true_delta,
            "predicted_delta": predicted_delta,
            "absolute_signed_delta_error": abs(predicted_delta - true_delta),
        })
    pair_frame = pd.DataFrame(pair_rows)
    model_suffix = "json" if predictor == "xgboost" else "pt"
    model_path = output / "models" / f"label_permuted--{predictor}--{target}--{representation}--init{init_seed}--perm{permutation_id}.{model_suffix}"
    if predictor == "xgboost":
        model.save_model(model_path)
        loaded = model
    else:
        torch.save(bundle, model_path)
        loaded = (model, bundle)
    model_row = {
        "dataset": target, "representation": representation, "predictor": predictor,
        "seed": init_seed, "model_initialization_seed": init_seed,
        "permutation_id": permutation_id, "permutation_seed": seed,
        "variant": "label_permuted", "best_step": int(best_step),
        "source": "new_fit", "path": str(model_path.relative_to(ROOT)), "sha256": sha256(model_path),
    }
    target_eval = target_geometry["evaluation"]
    target_scored = target_geometry["scored"]
    target_mutant_positions = target_geometry["mutant_positions"]
    parent_positions = target_geometry["parent_positions"]
    mutant_predictions = np.full(len(unique), np.nan, dtype=float)
    mutant_predictions[target_mutant_positions] = score_predict(
        predictor, loaded, mutant_values[target_mutant_positions], device,
    )
    parent_values = score_predict(predictor, loaded, values[parent_positions], device)
    parent_predictions = dict(zip(target_eval.row_id, parent_values))
    mean_scores, mean_support = mean_replacement_arrays(
        target_eval, target_scored, mutant_predictions, parent_predictions, cfg["mask_size_exponent"],
    )
    lime_scores, lime_support, fidelity = lime_arrays(
        target_eval, target_geometry["neighborhoods"], original_morgan, mutant_morgan, original_position,
        mutant_predictions, parent_predictions, cfg["crem_lime"],
    )
    pair_metrics = pair_frame.copy()
    pair_metrics["gap_score"] = -pair_metrics.absolute_signed_delta_error
    explanation_rows = []
    for explainer, arrays, supports in [
        ("crem_mean", mean_scores, mean_support), ("crem_lime", lime_scores, lime_support),
    ]:
        metadata = {
            "dataset": target, "representation": representation, "predictor": predictor,
            "explainer": explainer, "seed": init_seed, "model_initialization_seed": init_seed,
            "permutation_id": permutation_id, "permutation_seed": seed, "variant": "label_permuted",
        }
        add_pair_rows(explanation_rows, valid_pairs, arrays, supports, metadata, pair_metrics)
    fidelity_frame = add_design_fields(pd.DataFrame(fidelity), cfg, permutation_id, seed)
    fidelity_frame["representation"] = representation
    fidelity_frame["predictor"] = predictor
    local_mutant_predictions = mutant_predictions[target_mutant_positions]
    combined = np.concatenate([parent_values, local_mutant_predictions])
    output_audit = pd.DataFrame([{
        "dataset": target, "representation": representation, "predictor": predictor,
        "seed": init_seed, "model_initialization_seed": init_seed,
        "permutation_id": permutation_id, "permutation_seed": seed,
        "variant": "label_permuted", "parent_prediction_range": float(np.ptp(parent_values)),
        "mutant_prediction_range": float(np.ptp(local_mutant_predictions)),
        "local_domain_prediction_range": float(np.ptp(combined)),
        "finite_local_domain": bool(np.isfinite(combined).all()),
        "parent_rows": len(parent_values), "mutant_rows": len(local_mutant_predictions),
    }])
    tables = {
        "predictions.csv": pd.DataFrame(prediction_rows),
        "prediction_metrics.csv": pd.DataFrame(metric_rows),
        "pair_predictions.csv": pair_frame,
        "pair_explanations.csv": pd.DataFrame(explanation_rows),
        "local_fidelity.csv": fidelity_frame,
        "model_output_audit.csv": output_audit,
    }
    if not np.isfinite(tables["pair_explanations.csv"].normalized_ap_lift).all():
        raise ValueError("non-finite normalized AP lift")
    if not bool(output_audit.finite_local_domain.iloc[0]) or output_audit.local_domain_prediction_range.iloc[0] <= 1e-8:
        raise ValueError("randomized model is invalid on the explanation domain")
    write_unit(unit_dir, tables, model_row, config_hash, "new_fit")


def consolidate(output, unit_dirs, label_paths, cfg, selected, config_path, input_hashes, smoke):
    table_names = [
        "predictions.csv", "prediction_metrics.csv", "pair_predictions.csv",
        "pair_explanations.csv", "local_fidelity.csv", "model_output_audit.csv",
    ]
    tables = {}
    model_rows = []
    for name in table_names:
        tables[name] = pd.concat([pd.read_csv(path / name) for path in unit_dirs], ignore_index=True)
        tables[name].to_csv(output / name, index=False)
    for path in unit_dirs:
        model_rows.append(json.loads((path / "unit_manifest.json").read_text(encoding="utf-8"))["model"])
    models = pd.DataFrame(model_rows)
    models.to_csv(output / "model_manifest.csv", index=False)
    labels = pd.concat([pd.read_csv(path) for path in label_paths], ignore_index=True).drop_duplicates(
        ["row_id", "dataset", "permutation_id"]
    )
    labels.to_csv(output / "permuted_labels.csv", index=False)
    pair_frame = tables["pair_explanations.csv"]
    test = pair_frame[pair_frame.component_split == "test"]
    by_target = test.groupby(
        ["dataset", "representation", "predictor", "explainer", "permutation_id", "permutation_seed"],
        as_index=False,
    ).agg(
        median_normalized_ap_lift=("normalized_ap_lift", "median"),
        mean_normalized_ap_lift=("normalized_ap_lift", "mean"),
        test_pairs=("component_id", "size"),
    )
    by_target.to_csv(output / "permutation_nap_by_target_configuration.csv", index=False)
    target_stability = by_target.groupby(
        ["dataset", "representation", "predictor", "explainer"], as_index=False,
    ).agg(
        permutations=("permutation_id", "nunique"),
        mean_of_permutation_medians=("median_normalized_ap_lift", "mean"),
        sd_of_permutation_medians=("median_normalized_ap_lift", "std"),
        median_of_permutation_medians=("median_normalized_ap_lift", "median"),
        q25_of_permutation_medians=("median_normalized_ap_lift", lambda x: x.quantile(0.25)),
        q75_of_permutation_medians=("median_normalized_ap_lift", lambda x: x.quantile(0.75)),
        minimum_permutation_median=("median_normalized_ap_lift", "min"),
        maximum_permutation_median=("median_normalized_ap_lift", "max"),
    )
    target_stability.to_csv(output / "permutation_nap_stability_by_target_configuration.csv", index=False)
    target_equal = by_target.groupby(
        ["representation", "predictor", "explainer", "permutation_id"], as_index=False,
    ).agg(
        target_balanced_median_normalized_ap_lift=("median_normalized_ap_lift", "median"),
        targets=("dataset", "nunique"),
    )
    target_equal.to_csv(output / "permutation_nap_target_balanced.csv", index=False)
    configuration_stability = target_equal.groupby(
        ["representation", "predictor", "explainer"], as_index=False,
    ).agg(
        permutations=("permutation_id", "nunique"),
        mean_target_balanced_nap=("target_balanced_median_normalized_ap_lift", "mean"),
        sd_target_balanced_nap=("target_balanced_median_normalized_ap_lift", "std"),
        median_target_balanced_nap=("target_balanced_median_normalized_ap_lift", "median"),
        minimum_target_balanced_nap=("target_balanced_median_normalized_ap_lift", "min"),
        maximum_target_balanced_nap=("target_balanced_median_normalized_ap_lift", "max"),
    )
    configuration_stability.to_csv(output / "permutation_nap_stability_by_configuration.csv", index=False)
    expected_models = len(selected["targets"]) * len(selected["representations"]) * len(selected["predictors"]) * len(selected["permutation_ids"])
    finite = bool(np.isfinite(pair_frame.normalized_ap_lift).all())
    local_gate = bool(tables["model_output_audit.csv"].finite_local_domain.all()) and bool(
        (tables["model_output_audit.csv"].local_domain_prediction_range > 1e-8).all()
    )
    complete_crossing = len(models) == expected_models and bool(
        (target_stability.permutations == len(selected["permutation_ids"])).all()
    )
    status = "pass" if finite and local_gate and complete_crossing else "fail"
    summary = {
        "run_id": cfg["run_id"], "analysis_id": cfg["analysis_id"], "smoke": smoke,
        "status": status, "models": int(len(models)), "expected_models": expected_models,
        "reused_models": int((models.source == "reused_existing").sum()),
        "new_models": int((models.source == "new_fit").sum()),
        "targets": len(selected["targets"]), "representations": len(selected["representations"]),
        "predictors": len(selected["predictors"]), "permutations": len(selected["permutation_ids"]),
        "pair_explanation_rows": int(len(pair_frame)), "finite_normalized_ap_lift": finite,
        "nonconstant_local_domain_gate": local_gate, "complete_crossing": complete_crossing,
        "minimum_configuration_permutation_sd": float(configuration_stability.sd_target_balanced_nap.min()),
        "median_configuration_permutation_sd": float(configuration_stability.sd_target_balanced_nap.median()),
        "maximum_configuration_permutation_sd": float(configuration_stability.sd_target_balanced_nap.max()),
        "input_hashes": input_hashes,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["python", "research_v2/scripts/run_permutation_stability.py", *sys.argv[1:]],
        "config": config_path.resolve().relative_to(ROOT).as_posix(), "config_sha256": sha256(config_path),
        "selected_design": selected, "summary_sha256": sha256(output / "summary.json"),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "rdkit": rdkit.__version__, "scikit_learn": sklearn.__version__,
            "torch": torch.__version__, "xgboost": xgboost.__version__, "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    if status != "pass":
        raise SystemExit(2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--representations", nargs="+")
    parser.add_argument("--predictors", nargs="+")
    parser.add_argument("--permutation-ids", nargs="+", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {key: ROOT / value for key, value in cfg["inputs"].items()}
    output = args.output.resolve() if args.output else ROOT / cfg["output"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "units").mkdir(exist_ok=True)
    labels_dir = output / "labels"
    labels_dir.mkdir(exist_ok=True)
    run_log = output / "run.log"
    config_hash = sha256(args.config)
    if args.smoke:
        smoke = cfg["smoke_contract"]
        selected = {
            "targets": [smoke["target"]], "representations": [smoke["representation"]],
            "predictors": [smoke["predictor"]], "permutation_ids": smoke["permutation_ids"],
        }
    else:
        selected = {
            "targets": args.targets or cfg["confirmation_targets"],
            "representations": args.representations or cfg["representations"],
            "predictors": args.predictors or cfg["predictors"],
            "permutation_ids": args.permutation_ids or cfg["permutation_ids"],
        }
    for key, allowed in [
        ("targets", cfg["confirmation_targets"]), ("representations", cfg["representations"]),
        ("predictors", cfg["predictors"]), ("permutation_ids", cfg["permutation_ids"]),
    ]:
        if not set(selected[key]).issubset(set(allowed)):
            raise ValueError(f"selection outside frozen contract: {key}")
    molecules = pd.read_csv(paths["prepared"] / "model_molecules.csv")
    pairs = pd.read_csv(paths["prepared"] / "sentinel_pairs.csv")
    evaluation = pd.read_csv(paths["prepared"] / "evaluation_molecules.csv")
    scored = pd.read_csv(paths["counterfactuals"] / "scored_mutants_model.csv")
    unique = pd.read_csv(paths["counterfactuals"] / "unique_scored_mutants_model.csv")
    mutant_position = {smiles: index for index, smiles in enumerate(unique.canonical_smiles)}
    scored["mutant_position"] = scored.mutant_smiles.map(mutant_position)
    if scored.mutant_position.isna().any():
        raise ValueError("counterfactual representation mapping mismatch")
    scored["mutant_position"] = scored.mutant_position.astype(int)
    valid_pairs, rationale_audit = valid_rationale_pairs(evaluation, pairs)
    rationale_audit.to_csv(output / "rationale_prevalence_audit.csv", index=False)
    original_ids = molecules.row_id.astype(str).to_numpy()
    original_position = {row_id: index for index, row_id in enumerate(original_ids)}
    mutant_ids = unique.row_id.astype(str).to_numpy()
    original_morgan = load_npz(paths["representations"] / "morgan.npz", original_ids)
    mutant_morgan = load_npz(paths["mutant_representations"] / "morgan.npz", mutant_ids)
    source_labels = pd.read_csv(paths["source_heads"] / "permuted_labels.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_hashes = {
        "config": config_hash,
        "model_molecules": sha256(paths["prepared"] / "model_molecules.csv"),
        "sentinel_pairs": sha256(paths["prepared"] / "sentinel_pairs.csv"),
        "evaluation_molecules": sha256(paths["prepared"] / "evaluation_molecules.csv"),
        "scored_mutants_model": sha256(paths["counterfactuals"] / "scored_mutants_model.csv"),
        "source_model_manifest": sha256(paths["source_heads"] / "model_manifest.csv"),
        "source_pair_explanations": sha256(paths["source_explanations"] / "pair_explanations.csv"),
    }
    unit_dirs = []
    label_paths = []
    label_cache = {}
    total = len(selected["targets"]) * len(selected["representations"]) * len(selected["predictors"]) * len(selected["permutation_ids"])
    completed = 0
    log(run_log, f"start smoke={args.smoke} units={total} device={device}")
    geometry = {}
    for target in selected["targets"]:
        target_eval = evaluation[evaluation.dataset == target]
        target_scored = scored[scored.dataset == target]
        geometry[target] = {
            "evaluation": target_eval,
            "scored": target_scored,
            "mutant_positions": np.sort(target_scored.mutant_position.unique()),
            "parent_positions": [original_position[row_id] for row_id in target_eval.row_id],
            "neighborhoods": build_lime_neighborhoods(
                target_eval, target_scored, original_morgan, mutant_morgan,
                original_position, cfg["crem_lime"],
            ),
        }
    for target in selected["targets"]:
        for permutation_id in selected["permutation_ids"]:
            label_path, label_indices, permuted_labels = make_labels(cfg, molecules, target, permutation_id, labels_dir)
            label_paths.append(label_path)
            label_cache[(target, permutation_id)] = (label_indices, permuted_labels)
            if permutation_id == cfg["randomization"]["reused_permutation_id"]:
                existing = source_labels[(source_labels.dataset == target) & (source_labels.replicate == permutation_id)].sort_values("row_id")
                generated = pd.read_csv(label_path).sort_values("row_id")
                if len(existing) != len(generated) or not np.allclose(existing.permuted_y, generated.permuted_y, atol=0, rtol=0):
                    raise ValueError(f"reused label permutation mismatch: {target}")
    for representation in selected["representations"]:
        values = load_representation(paths["representations"] / f"{representation}.npz", original_ids)
        mutant_values = load_npz(paths["mutant_representations"] / f"{representation}.npz", mutant_ids)
        for target in selected["targets"]:
            for predictor in selected["predictors"]:
                for permutation_id in selected["permutation_ids"]:
                    seed = permutation_seed(cfg, target, permutation_id)
                    key = unit_key(target, representation, predictor, permutation_id)
                    unit_dir = output / "units" / key
                    unit_dirs.append(unit_dir)
                    if unit_complete(unit_dir, config_hash):
                        completed += 1
                        log(run_log, f"resume-skip {completed}/{total} {key}")
                        continue
                    if permutation_id == cfg["randomization"]["reused_permutation_id"]:
                        reuse_source_unit(
                            cfg, paths, target, representation, predictor, permutation_id, seed, unit_dir, config_hash,
                        )
                    else:
                        label_indices, permuted_labels = label_cache[(target, permutation_id)]
                        train_and_score_unit(
                            cfg, molecules, pairs, valid_pairs, unique, original_ids, original_position,
                            original_morgan, mutant_morgan, values, mutant_values, geometry[target],
                            representation, target, predictor, permutation_id, seed, label_indices,
                            permuted_labels, unit_dir, output, config_hash, device,
                        )
                    completed += 1
                    log(run_log, f"complete {completed}/{total} {key}")
    summary = consolidate(output, unit_dirs, label_paths, cfg, selected, args.config, input_hashes, args.smoke)
    log(run_log, f"finish status={summary['status']} models={summary['models']}")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
