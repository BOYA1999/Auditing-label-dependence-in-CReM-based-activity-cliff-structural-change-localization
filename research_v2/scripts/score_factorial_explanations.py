import argparse
import hashlib
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from torch import nn
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research_v2/configs/factorial_v1.json"
BASE = ROOT / "research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1"
PREPARED = BASE / "prepared"
REPRESENTATIONS = BASE / "full/representations"
MUTANT_REPRESENTATIONS = BASE / "full/counterfactual_representations"
COUNTERFACTUALS = BASE / "full/counterfactuals"
XGB_HEADS = BASE / "full/common_heads"
MLP_HEADS = BASE / "full/mlp_heads"
REFERENCE = BASE / "full/explanations_v2"
DEFAULT_OUT = BASE / "full/factorial_explanations"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers = []
        current = input_dim
        for width in hidden:
            layers.extend([nn.Linear(current, width), nn.ReLU()])
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values).squeeze(-1)


def load_npz(path, expected_ids):
    data = np.load(path, allow_pickle=False)
    row_ids = data["row_id"].astype(str)
    if not np.array_equal(row_ids, expected_ids):
        raise ValueError(f"row order mismatch: {path}")
    values = data["x"].astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite representation: {path}")
    return values


def normalized_ap(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    prevalence = float(labels.mean())
    if prevalence <= 0 or prevalence >= 1:
        return None
    ap = float(average_precision_score(labels, scores))
    return ap, prevalence, float((ap - prevalence) / (1 - prevalence))


def control_labels(mol):
    atoms = list(mol.GetAtoms())
    return {
        "halogen_count": np.array([atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms], dtype=bool),
        "heteroatom_count": np.array([atom.GetAtomicNum() in {7, 8, 15, 16} for atom in atoms], dtype=bool),
        "aromatic_atom_count": np.array([atom.GetIsAromatic() for atom in atoms], dtype=bool),
    }


def atom_environment(mol, center, radius):
    if radius == 0:
        return {int(center)}
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(center))
    atoms = {int(center)}
    for bond_index in bonds:
        bond = mol.GetBondWithIdx(int(bond_index))
        atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return atoms


def build_lime_neighborhoods(evaluation, scored, original_morgan, mutant_morgan, original_position, cfg):
    grouped = {row_id: frame for row_id, frame in scored.groupby("row_id", sort=False)}
    neighborhoods = {}
    for record in evaluation.itertuples(index=False):
        group = grouped.get(record.row_id)
        if group is None:
            mutant_positions = np.array([], dtype=int)
            similarities = np.array([1.0])
        else:
            group = group.drop_duplicates("mutant_position")
            mutant_positions = group.mutant_position.to_numpy(dtype=int)
            similarities = np.concatenate([[1.0], group.parent_mutant_tanimoto.to_numpy(dtype=float)])
        parent = original_morgan[original_position[record.row_id]]
        matrix = np.vstack([parent, mutant_morgan[mutant_positions]])
        variable_bits = np.flatnonzero(np.any(matrix != matrix[0], axis=0))
        weights = np.exp(-np.square((1.0 - similarities) / cfg["kernel_width"]))
        mol = Chem.MolFromSmiles(record.canonical_smiles)
        additional = rdFingerprintGenerator.AdditionalOutput()
        additional.AllocateBitInfoMap()
        rdFingerprintGenerator.GetMorganGenerator(
            radius=cfg["fingerprint_radius"], fpSize=cfg["fingerprint_bits"],
        ).GetFingerprint(mol, additionalOutput=additional)
        bit_info = additional.GetBitInfoMap()
        bit_to_column = {int(bit): idx for idx, bit in enumerate(variable_bits)}
        atom_columns = [[] for _ in range(mol.GetNumAtoms())]
        for bit, occurrences in bit_info.items():
            if int(bit) not in bit_to_column:
                continue
            column = bit_to_column[int(bit)]
            atoms = set()
            for center, radius in occurrences:
                atoms.update(atom_environment(mol, center, radius))
            for atom_index in atoms:
                atom_columns[atom_index].append(column)
        neighborhoods[record.row_id] = {
            "mutant_positions": mutant_positions,
            "variable_bits": variable_bits,
            "weights": weights,
            "atom_columns": atom_columns,
        }
    return neighborhoods


def weighted_r2(observed, predicted, weights):
    mean = float(np.average(observed, weights=weights))
    denominator = float(np.sum(weights * np.square(observed - mean)))
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - np.sum(weights * np.square(observed - predicted)) / denominator)


def mean_replacement_arrays(molecules, scored, mutant_predictions, parent_predictions, exponent):
    frame = scored.copy()
    frame["parent_prediction"] = frame.row_id.map(parent_predictions)
    frame["mutant_prediction"] = mutant_predictions[frame.mutant_position.to_numpy(dtype=int)]
    frame["effect"] = (frame.mutant_prediction - frame.parent_prediction).abs()
    masks = frame.groupby(["row_id", "mask_id", "mask_atoms", "mask_size"], as_index=False).effect.mean()
    masks["effect"] = masks.effect / np.power(masks.mask_size, exponent)
    grouped = {row_id: group for row_id, group in masks.groupby("row_id", sort=False)}
    arrays = {}
    supports = {}
    for record in molecules.itertuples(index=False):
        atoms = Chem.MolFromSmiles(record.canonical_smiles).GetNumAtoms()
        scores = np.zeros(atoms, dtype=float)
        counts = np.zeros(atoms, dtype=int)
        for mask in grouped.get(record.row_id, pd.DataFrame()).itertuples(index=False):
            for atom_index in json.loads(mask.mask_atoms):
                scores[atom_index] += float(mask.effect)
                counts[atom_index] += 1
        arrays[record.row_id] = np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)
        supports[record.row_id] = counts > 0
    return arrays, supports


def lime_arrays(molecules, neighborhoods, original_morgan, mutant_morgan, original_position, mutant_predictions, parent_predictions, cfg):
    arrays = {}
    supports = {}
    fidelity_rows = []
    for record in molecules.itertuples(index=False):
        neighborhood = neighborhoods[record.row_id]
        parent_fp = original_morgan[original_position[record.row_id]]
        positions = neighborhood["mutant_positions"]
        bits = neighborhood["variable_bits"]
        matrix = np.vstack([parent_fp, mutant_morgan[positions]])[:, bits]
        observed = np.concatenate([[parent_predictions[record.row_id]], mutant_predictions[positions]])
        if len(bits):
            model = Ridge(alpha=cfg["ridge_alpha"], solver="lsqr")
            model.fit(matrix, observed, sample_weight=neighborhood["weights"])
            fitted = model.predict(matrix)
            coefficients = np.abs(model.coef_)
        else:
            fitted = np.repeat(observed.mean(), len(observed))
            coefficients = np.array([], dtype=float)
        scores = np.zeros(len(neighborhood["atom_columns"]), dtype=float)
        counts = np.zeros(len(scores), dtype=int)
        for atom_index, columns in enumerate(neighborhood["atom_columns"]):
            if columns:
                scores[atom_index] = float(np.mean(coefficients[columns]))
                counts[atom_index] = len(columns)
        arrays[record.row_id] = scores
        supports[record.row_id] = counts > 0
        fidelity_rows.append({
            "row_id": record.row_id, "dataset": record.dataset,
            "component_split": record.component_split,
            "local_weighted_r2": weighted_r2(observed, fitted, neighborhood["weights"]),
            "local_samples": int(len(observed)), "variable_bits": int(len(bits)),
            "supported_atom_fraction": float((counts > 0).mean()),
        })
    return arrays, supports, fidelity_rows


def load_predictor(predictor, entry, device):
    path = ROOT / entry.path
    if sha256(path) != entry.sha256:
        raise ValueError(f"model hash mismatch: {path}")
    if predictor == "xgboost":
        model = XGBRegressor()
        model.load_model(path)
        return model
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = MLP(bundle["input_dimension"], bundle["hidden_dimensions"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle


def predict(predictor, loaded, values, device):
    if predictor == "xgboost":
        return loaded.predict(values)
    model, bundle = loaded
    mean = bundle["feature_mean"].numpy()
    std = bundle["feature_std"].numpy()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(values), 2048):
            batch = torch.from_numpy((values[start:start + 2048] - mean) / std).to(device)
            rows.append(model(batch).float().cpu().numpy())
    return np.concatenate(rows) * bundle["target_std"] + bundle["target_mean"]


def add_atom_rows(rows, molecules, arrays, supports, metadata):
    for record in molecules.itertuples(index=False):
        for atom_index, (score, supported) in enumerate(zip(arrays[record.row_id], supports[record.row_id])):
            rows.append({
                **metadata, "row_id": record.row_id, "dataset": record.dataset,
                "component_split": record.component_split, "atom_index": atom_index,
                "atom_score": float(score), "supported": bool(supported),
            })


def add_pair_rows(rows, pairs, arrays, supports, metadata, pair_metrics):
    lookup = pair_metrics.set_index(["component_id", "component_split"])
    target = metadata["dataset"]
    for pair in pairs[pairs.dataset == target].itertuples(index=False):
        left_id = f"{target}:{int(pair.left_index)}"
        right_id = f"{target}:{int(pair.right_index)}"
        labels = np.concatenate([
            np.isin(np.arange(len(arrays[left_id])), json.loads(pair.rationale_left)),
            np.isin(np.arange(len(arrays[right_id])), json.loads(pair.rationale_right)),
        ])
        scores = np.concatenate([arrays[left_id], arrays[right_id]])
        result = normalized_ap(labels, scores)
        if result is None:
            raise ValueError(f"invalid rationale prevalence: {target} component={pair.component_id}")
        ap, prevalence, lift = result
        metrics = lookup.loc[(int(pair.component_id), pair.component_split)]
        rows.append({
            **metadata, "component_id": int(pair.component_id),
            "component_split": pair.component_split,
            "left_index": int(pair.left_index), "right_index": int(pair.right_index),
            "average_precision": ap, "prevalence": prevalence,
            "normalized_ap_lift": lift,
            "supported_atom_fraction": float(np.concatenate([supports[left_id], supports[right_id]]).mean()),
            "true_delta": float(metrics.true_delta), "predicted_delta": float(metrics.predicted_delta),
            "absolute_signed_delta_error": float(metrics.absolute_signed_delta_error),
            "gap_score": float(metrics.gap_score),
        })


def add_control_rows(rows, molecules, arrays, supports, metadata, predictive_r2):
    control = metadata["control"]
    for record in molecules[molecules.component_split == "test"].itertuples(index=False):
        labels = control_labels(Chem.MolFromSmiles(record.canonical_smiles))[control]
        result = normalized_ap(labels, arrays[record.row_id])
        if result is None:
            continue
        ap, prevalence, lift = result
        rows.append({
            **metadata, "row_id": record.row_id, "dataset": record.dataset,
            "predictive_test_r2": predictive_r2,
            "average_precision": ap, "prevalence": prevalence,
            "normalized_ap_lift": lift,
            "supported_atom_fraction": float(supports[record.row_id].mean()),
            "rationale_supported_fraction": float(supports[record.row_id][labels].mean()),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mask-size-exponent", type=float)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.warning")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mask_size_exponent = cfg["mask_size_exponent"] if args.mask_size_exponent is None else args.mask_size_exponent
    molecules = pd.read_csv(PREPARED / "molecules.csv")
    pairs = pd.read_csv(PREPARED / "sentinel_pairs.csv")
    scored = pd.read_csv(COUNTERFACTUALS / "scored_mutants.csv")
    unique = pd.read_csv(COUNTERFACTUALS / "unique_scored_mutants.csv")
    original_ids = molecules.row_id.astype(str).to_numpy()
    mutant_ids = unique.row_id.astype(str).to_numpy()
    mutant_position = {smiles: idx for idx, smiles in enumerate(unique.canonical_smiles)}
    scored["mutant_position"] = scored.mutant_smiles.map(mutant_position)
    if scored.mutant_position.isna().any():
        raise ValueError("scored mutant missing from representation index")
    evaluation_ids = set(scored.row_id)
    for pair in pairs.itertuples(index=False):
        evaluation_ids.update([f"{pair.dataset}:{int(pair.left_index)}", f"{pair.dataset}:{int(pair.right_index)}"])
    evaluation = molecules[molecules.row_id.isin(evaluation_ids)].copy()
    original_position = {row_id: idx for idx, row_id in enumerate(original_ids)}
    original_morgan = load_npz(REPRESENTATIONS / "morgan.npz", original_ids)
    mutant_morgan = load_npz(MUTANT_REPRESENTATIONS / "morgan.npz", mutant_ids)
    neighborhoods = build_lime_neighborhoods(
        evaluation, scored, original_morgan, mutant_morgan, original_position, cfg["crem_lime"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = {
        "xgboost": pd.read_csv(XGB_HEADS / "model_manifest.csv"),
        "mlp": pd.read_csv(MLP_HEADS / "model_manifest.csv"),
    }
    predictions = {
        "xgboost": pd.read_csv(XGB_HEADS / "predictions.csv"),
        "mlp": pd.read_csv(MLP_HEADS / "predictions.csv"),
    }
    pair_predictions = {
        "xgboost": pd.read_csv(XGB_HEADS / "pair_predictions.csv"),
        "mlp": pd.read_csv(MLP_HEADS / "pair_predictions.csv"),
    }
    probes = {
        "xgboost": pd.read_csv(XGB_HEADS / "probe_metrics.csv"),
        "mlp": pd.read_csv(MLP_HEADS / "probe_metrics.csv"),
    }
    pair_rows = []
    control_rows = []
    fidelity_rows = []
    atom_rows = []
    reload_deltas = []
    active_representations = cfg["representations"][:1] if args.smoke else cfg["representations"]
    active_predictors = cfg["predictors"][:1] if args.smoke else cfg["predictors"]
    active_targets = cfg["development_targets"][:1] if args.smoke else cfg["development_targets"]
    active_seeds = [42] if args.smoke else cfg["seeds"]
    for representation in active_representations:
        original_values = load_npz(REPRESENTATIONS / f"{representation}.npz", original_ids)
        mutant_values = load_npz(MUTANT_REPRESENTATIONS / f"{representation}.npz", mutant_ids)
        for predictor in active_predictors:
            manifest = manifests[predictor]
            for target in active_targets:
                target_molecules = evaluation[evaluation.dataset == target]
                target_scored = scored[scored.dataset == target]
                for seed in active_seeds:
                    entry = manifest[
                        (manifest.kind == "bioactivity") & (manifest.dataset == target)
                        & (manifest.representation == representation) & (manifest.seed == seed)
                    ].iloc[0]
                    loaded = load_predictor(predictor, entry, device)
                    mutant_predictions = predict(predictor, loaded, mutant_values, device)
                    positions = [original_position[row_id] for row_id in target_molecules.row_id]
                    parent_values = predict(predictor, loaded, original_values[positions], device)
                    parent_predictions = dict(zip(target_molecules.row_id, parent_values))
                    expected = predictions[predictor][
                        (predictions[predictor].dataset == target)
                        & (predictions[predictor].representation == representation)
                        & (predictions[predictor].seed == seed)
                        & (predictions[predictor].row_id.isin(target_molecules.row_id))
                    ].set_index("row_id").prediction
                    reload_delta = max(abs(float(parent_predictions[row_id]) - float(expected.loc[row_id])) for row_id in target_molecules.row_id)
                    if reload_delta > 1e-5:
                        raise ValueError(f"prediction reload mismatch: {predictor} {target} {representation} seed={seed}")
                    reload_deltas.append(reload_delta)
                    mean_arrays, mean_supports = mean_replacement_arrays(
                        target_molecules, target_scored, mutant_predictions, parent_predictions, mask_size_exponent,
                    )
                    lime_score_arrays, lime_supports, local_fidelity = lime_arrays(
                        target_molecules, neighborhoods, original_morgan, mutant_morgan, original_position,
                        mutant_predictions, parent_predictions, cfg["crem_lime"],
                    )
                    pair_metric = pair_predictions[predictor][
                        (pair_predictions[predictor].dataset == target)
                        & (pair_predictions[predictor].representation == representation)
                        & (pair_predictions[predictor].seed == seed)
                    ]
                    for explainer, arrays, supports in [
                        ("crem_mean", mean_arrays, mean_supports),
                        ("crem_lime", lime_score_arrays, lime_supports),
                    ]:
                        metadata = {"dataset": target, "representation": representation, "predictor": predictor, "explainer": explainer, "seed": int(seed)}
                        add_pair_rows(pair_rows, pairs, arrays, supports, metadata, pair_metric)
                        add_atom_rows(atom_rows, target_molecules, arrays, supports, metadata)
                    for row in local_fidelity:
                        row.update({"score_kind": "bioactivity", "model_name": target, "representation": representation, "predictor": predictor, "seed": int(seed)})
                        fidelity_rows.append(row)
                    print(f"bioactivity {target} {representation} {predictor} seed={seed}", flush=True)
            qualified = probes[predictor][
                (probes[predictor].representation == representation)
                & (probes[predictor].split == "test")
                & (probes[predictor].r2 >= cfg["positive_controls"]["minimum_predictive_r2"])
            ]
            for control in qualified.control:
                entry = manifest[
                    (manifest.kind == "probe") & (manifest.representation == representation)
                    & (manifest.control == control)
                ].iloc[0]
                loaded = load_predictor(predictor, entry, device)
                mutant_predictions = predict(predictor, loaded, mutant_values, device)
                positions = [original_position[row_id] for row_id in evaluation.row_id]
                parent_values = predict(predictor, loaded, original_values[positions], device)
                parent_predictions = dict(zip(evaluation.row_id, parent_values))
                mean_arrays, mean_supports = mean_replacement_arrays(
                    evaluation, scored, mutant_predictions, parent_predictions, mask_size_exponent,
                )
                lime_score_arrays, lime_supports, local_fidelity = lime_arrays(
                    evaluation, neighborhoods, original_morgan, mutant_morgan, original_position,
                    mutant_predictions, parent_predictions, cfg["crem_lime"],
                )
                predictive_r2 = float(qualified.loc[qualified.control == control, "r2"].iloc[0])
                for explainer, arrays, supports in [
                    ("crem_mean", mean_arrays, mean_supports),
                    ("crem_lime", lime_score_arrays, lime_supports),
                ]:
                    metadata = {"representation": representation, "predictor": predictor, "explainer": explainer, "control": control, "seed": 42}
                    add_control_rows(control_rows, evaluation, arrays, supports, metadata, predictive_r2)
                for row in local_fidelity:
                    row.update({"score_kind": "positive_control", "model_name": control, "representation": representation, "predictor": predictor, "seed": 42})
                    fidelity_rows.append(row)
                print(f"control {control} {representation} {predictor}", flush=True)
    pair_frame = pd.DataFrame(pair_rows)
    control_frame = pd.DataFrame(control_rows)
    fidelity_frame = pd.DataFrame(fidelity_rows)
    atom_frame = pd.DataFrame(atom_rows)
    pair_frame.to_csv(args.output / "pair_explanations.csv", index=False)
    control_frame.to_csv(args.output / "positive_controls.csv", index=False)
    fidelity_frame.to_csv(args.output / "local_fidelity.csv", index=False)
    atom_frame.to_csv(args.output / "atom_scores.csv", index=False)
    control_summary = control_frame.groupby(["representation", "predictor", "explainer", "control"], as_index=False).agg(
        median_normalized_ap_lift=("normalized_ap_lift", "median"),
        mean_normalized_ap_lift=("normalized_ap_lift", "mean"), molecules=("row_id", "size"),
    )
    control_summary["passes"] = control_summary.median_normalized_ap_lift > cfg["positive_controls"]["minimum_median_normalized_ap_lift"]
    control_summary.to_csv(args.output / "positive_control_summary.csv", index=False)
    fidelity_summary = fidelity_frame.groupby(["score_kind", "model_name", "representation", "predictor"], as_index=False).agg(
        median_local_weighted_r2=("local_weighted_r2", "median"),
        mean_local_weighted_r2=("local_weighted_r2", "mean"),
        median_supported_atom_fraction=("supported_atom_fraction", "median"),
        molecules=("row_id", "size"),
    )
    fidelity_summary.to_csv(args.output / "local_fidelity_summary.csv", index=False)
    reference = pd.read_csv(REFERENCE / "pair_explanations.csv")
    reproduced = pair_frame[(pair_frame.predictor == "xgboost") & (pair_frame.explainer == "crem_mean")].merge(
        reference[["dataset", "component_id", "component_split", "representation", "seed", "normalized_ap_lift"]],
        on=["dataset", "component_id", "component_split", "representation", "seed"], suffixes=("_factorial", "_reference"), validate="one_to_one",
    )
    reproduction_delta = float((reproduced.normalized_ap_lift_factorial - reproduced.normalized_ap_lift_reference).abs().max())
    reference_reproduction_applicable = abs(mask_size_exponent - 0.5) <= 1e-12
    summary = {
        "run_id": cfg["run_id"], "smoke": args.smoke, "mask_size_exponent": mask_size_exponent,
        "status": "success" if np.isfinite(pair_frame.normalized_ap_lift).all() and np.isfinite(control_frame.normalized_ap_lift).all() and np.isfinite(fidelity_frame.local_weighted_r2).all() and (not reference_reproduction_applicable or reproduction_delta <= 1e-12) else "failure",
        "pair_rows": int(len(pair_frame)), "test_pair_rows": int((pair_frame.component_split == "test").sum()),
        "atom_rows": int(len(atom_frame)), "positive_control_rows": int(len(control_frame)),
        "positive_control_tasks": int(len(control_summary)), "positive_control_tasks_passing": int(control_summary.passes.sum()),
        "crem_lime_positive_control_tasks": int((control_summary.explainer == "crem_lime").sum()),
        "crem_lime_positive_control_tasks_passing": int(control_summary.loc[control_summary.explainer == "crem_lime", "passes"].sum()),
        "xgboost_crem_mean_reference_reproduction_applicable": reference_reproduction_applicable,
        "xgboost_crem_mean_reference_maximum_delta": reproduction_delta,
        "maximum_reloaded_prediction_delta": float(max(reload_deltas)),
        "input_hashes": {
            "config": sha256(args.config), "scored_mutants": sha256(COUNTERFACTUALS / "scored_mutants.csv"),
            "xgb_manifest": sha256(XGB_HEADS / "model_manifest.csv"), "mlp_manifest": sha256(MLP_HEADS / "model_manifest.csv"),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    if summary["status"] != "success":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
