import argparse
import importlib.util
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdRascalMCES


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def rationale(mol_a, mol_b, cfg):
    options = rdRascalMCES.RascalOptions()
    options.similarityThreshold = cfg["similarity_threshold"]
    options.ringMatchesRingOnly = cfg["ring_matches_ring_only"]
    options.completeAromaticRings = cfg["complete_aromatic_rings"]
    options.completeSmallestRings = cfg["complete_smallest_rings"]
    options.ignoreAtomAromaticity = False
    options.ignoreBondOrders = False
    options.timeout = cfg["timeout_seconds"]
    results = rdRascalMCES.FindMCES(mol_a, mol_b, options)
    if results and results[0].timedOut:
        return None, "timeout"
    if not results:
        return None, "empty_mces"
    matches = results[0].atomMatches()
    if not matches:
        return None, "invalid_match"
    match_a, match_b = zip(*matches)
    matched_a, matched_b = set(match_a), set(match_b)
    changed_a = set(range(mol_a.GetNumAtoms())) - matched_a
    changed_b = set(range(mol_b.GetNumAtoms())) - matched_b
    bond_matches = results[0].bondMatches()
    matched_bonds_a = {a for a, _ in bond_matches}
    matched_bonds_b = {b for _, b in bond_matches}
    for bond in mol_a.GetBonds():
        if bond.GetIdx() not in matched_bonds_a:
            changed_a.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    for bond in mol_b.GetBonds():
        if bond.GetIdx() not in matched_bonds_b:
            changed_b.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    if cfg["include_boundary_atoms"]:
        changed_a |= {
            neighbor.GetIdx()
            for idx in list(changed_a)
            for neighbor in mol_a.GetAtomWithIdx(idx).GetNeighbors()
            if neighbor.GetIdx() in matched_a
        }
        changed_b |= {
            neighbor.GetIdx()
            for idx in list(changed_b)
            for neighbor in mol_b.GetAtomWithIdx(idx).GetNeighbors()
            if neighbor.GetIdx() in matched_b
        }
    if not changed_a and not changed_b:
        return None, "no_changed_atoms"
    return {
        "mcs_atoms": len(matches),
        "match": list(zip(match_a, match_b)),
        "rationale_a": sorted(changed_a),
        "rationale_b": sorted(changed_b),
    }, ""


def refresh_rationales(root, cfg, reuse_path, out, target, scope):
    source_pairs = pd.read_csv(reuse_path / "pairs.csv")
    molecule_splits = pd.read_csv(reuse_path / "molecule_splits.csv")
    datasets = sorted(source_pairs.dataset.unique())
    if target:
        datasets = [dataset for dataset in datasets if dataset == target]
    data_dir = root / "external" / "MoleculeACE" / "MoleculeACE" / "Data" / "benchmark_data"
    rows, summaries = [], []
    for dataset in datasets:
        df = pd.read_csv(data_dir / f"{dataset}.csv")
        mols = [Chem.MolFromSmiles(s) for s in df.smiles]
        target_pairs = source_pairs[source_pairs.dataset == dataset].copy()
        failures = 0
        for row in target_pairs.itertuples(index=False):
            evaluate = bool(row.is_sentinel) or scope == "all"
            proxy, reason = rationale(mols[int(row.left_index)], mols[int(row.right_index)], cfg["mcs"]) if evaluate else (None, "not_evaluated")
            failures += int(evaluate and proxy is None)
            item = row._asdict()
            item.update(
                {
                    "mcs_atoms": proxy["mcs_atoms"] if proxy else np.nan,
                    "match": json.dumps(proxy["match"]) if proxy else "",
                    "rationale_left": json.dumps(proxy["rationale_a"]) if proxy else "",
                    "rationale_right": json.dumps(proxy["rationale_b"]) if proxy else "",
                    "rationale_valid": proxy is not None,
                    "rationale_failure_reason": reason,
                }
            )
            rows.append(item)
        evaluated = int(target_pairs.is_sentinel.sum()) if scope == "sentinel" else len(target_pairs)
        summaries.append(
            {
                "dataset": dataset,
                "pairs": len(target_pairs),
                "components": int(target_pairs.component_id.nunique()),
                "sentinels": int(target_pairs.is_sentinel.sum()),
                "rationales_evaluated": evaluated,
                "rationale_failures": failures,
            }
        )
        print(f"{dataset}: {len(target_pairs)} pairs, {failures}/{evaluated} proxy failures", flush=True)
    pair_output = pd.DataFrame(rows)
    molecule_output = molecule_splits[molecule_splits.dataset.isin(datasets)]
    pair_output.to_csv(out / "pairs.csv", index=False)
    molecule_output.to_csv(out / "molecule_splits.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "pair_preparation_summary.csv", index=False)
    totals = {
        "datasets": len(summaries),
        "pairs": int(summary.pairs.sum()),
        "components": int(summary.components.sum()),
        "sentinels": int(summary.sentinels.sum()),
        "rationales_evaluated": int(summary.rationales_evaluated.sum()),
        "rationale_failures": int(summary.rationale_failures.sum()),
        "rationale_scope": scope,
        "pair_graph_reused_from": str(reuse_path),
    }
    (out / "pair_preparation_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target")
    parser.add_argument("--rationale-scope", choices=["sentinel", "all"], default="sentinel")
    parser.add_argument("--reuse-pairs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    source = root / "external" / "MoleculeACE"
    spec = importlib.util.spec_from_file_location(
        "moleculeace_cliffs", source / "MoleculeACE" / "benchmark" / "cliffs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.warning")
    if args.reuse_pairs:
        refresh_rationales(
            root,
            cfg,
            root / args.reuse_pairs,
            out,
            args.target,
            args.rationale_scope,
        )
        return
    paths = sorted((source / "MoleculeACE" / "Data" / "benchmark_data").glob("CHEMBL*.csv"))
    if args.target:
        paths = [p for p in paths if p.stem == args.target]
    pair_rows, molecule_rows, summaries = [], [], []
    for path in paths:
        dataset = path.stem
        df = pd.read_csv(path)
        mols = [Chem.MolFromSmiles(s) for s in df.smiles]
        matrix = module.ActivityCliffs(
            df.smiles.tolist(), df["exp_mean [nM]"].tolist()
        ).find_cliffs(
            similarity=cfg["activity_cliff"]["similarity"],
            potency_fold=cfg["activity_cliff"]["potency_fold"],
        )
        left, right = np.where(np.triu(matrix, 1) == 1)
        uf = UnionFind(len(df))
        for a, b in zip(left, right):
            uf.union(int(a), int(b))
        members = sorted(set(left.tolist() + right.tolist()))
        raw_components = {}
        for idx in members:
            raw_components.setdefault(uf.find(idx), []).append(idx)
        component_members = sorted(raw_components.values(), key=min)
        component_for = {
            idx: component_id
            for component_id, indices in enumerate(component_members)
            for idx in indices
        }
        component_ids = np.arange(len(component_members))
        rng = np.random.default_rng(cfg.get("split_seed", cfg["seed"]) + zlib.crc32(dataset.encode()))
        rng.shuffle(component_ids)
        n_train = int(np.floor(len(component_ids) * cfg["component_split"]["train"]))
        n_cal = int(np.floor(len(component_ids) * cfg["component_split"]["calibration"]))
        split_for = {
            int(c): "train" if i < n_train else "calibration" if i < n_train + n_cal else "test"
            for i, c in enumerate(component_ids)
        }
        component_pairs = {}
        for pair_idx, (a, b) in enumerate(zip(left, right)):
            component_pairs.setdefault(component_for[int(a)], []).append(pair_idx)
        sentinel = {
            max(
                indices,
                key=lambda i: (
                    abs(df.iloc[int(left[i])]["y [pEC50/pKi]"] - df.iloc[int(right[i])]["y [pEC50/pKi]"]),
                    -int(left[i]),
                    -int(right[i]),
                ),
            )
            for indices in component_pairs.values()
        }
        rationale_failures = 0
        for pair_idx, (a, b) in enumerate(zip(left, right)):
            a, b = int(a), int(b)
            is_sentinel = pair_idx in sentinel
            proxy, failure_reason = rationale(mols[a], mols[b], cfg["mcs"]) if is_sentinel or args.rationale_scope == "all" else (None, "not_evaluated")
            if (is_sentinel or args.rationale_scope == "all") and proxy is None:
                rationale_failures += 1
            pair_rows.append(
                {
                    "dataset": dataset,
                    "pair_index": pair_idx,
                    "left_index": a,
                    "right_index": b,
                    "component_id": component_for[a],
                    "component_split": split_for[component_for[a]],
                    "is_sentinel": is_sentinel,
                    "potency_difference": abs(df.iloc[a]["y [pEC50/pKi]"] - df.iloc[b]["y [pEC50/pKi]"]),
                    "mcs_atoms": proxy["mcs_atoms"] if proxy else np.nan,
                    "match": json.dumps(proxy["match"]) if proxy else "",
                    "rationale_left": json.dumps(proxy["rationale_a"]) if proxy else "",
                    "rationale_right": json.dumps(proxy["rationale_b"]) if proxy else "",
                    "rationale_valid": proxy is not None,
                    "rationale_failure_reason": failure_reason,
                }
            )
        for idx in range(len(df)):
            component_id = component_for.get(idx)
            molecule_rows.append(
                {
                    "dataset": dataset,
                    "molecule_index": idx,
                    "component_id": component_id if component_id is not None else -1,
                    "component_split": split_for[component_id] if component_id is not None else "isolated",
                }
            )
        evaluated = len(sentinel) if args.rationale_scope == "sentinel" else len(left)
        summaries.append(
            {
                "dataset": dataset,
                "pairs": len(left),
                "components": len(component_members),
                "sentinels": len(sentinel),
                "rationales_evaluated": evaluated,
                "rationale_failures": rationale_failures,
            }
        )
        print(f"{dataset}: {len(left)} pairs, {len(component_members)} components, {rationale_failures}/{evaluated} proxy failures", flush=True)
    pd.DataFrame(pair_rows).to_csv(out / "pairs.csv", index=False)
    pd.DataFrame(molecule_rows).to_csv(out / "molecule_splits.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "pair_preparation_summary.csv", index=False)
    totals = {
        "datasets": len(summaries),
        "pairs": int(summary.pairs.sum()),
        "components": int(summary.components.sum()),
        "sentinels": int(summary.sentinels.sum()),
        "rationales_evaluated": int(summary.rationales_evaluated.sum()),
        "rationale_failures": int(summary.rationale_failures.sum()),
        "rationale_scope": args.rationale_scope,
    }
    (out / "pair_preparation_summary.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
