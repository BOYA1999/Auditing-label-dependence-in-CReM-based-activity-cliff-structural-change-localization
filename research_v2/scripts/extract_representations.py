import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research_v2/configs/pilot_v1.json"
DEFAULT_INPUT = ROOT / "research_v2/artifacts/experiment/reprxai_cliffs_pilot_v1/prepared/smoke_molecules.csv"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def morgan(smiles, radius, bits):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    out = np.zeros((len(smiles), bits), dtype=np.float32)
    for idx, value in enumerate(smiles):
        fp = generator.GetFingerprint(Chem.MolFromSmiles(value))
        DataStructs.ConvertToNumpyArray(fp, out[idx])
    return out, {}


def molformer(smiles, cfg):
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path.home() / ".cache/huggingface/hub" / f"models--{cfg['model_id'].replace('/', '--')}" / "snapshots" / cfg["revision"]
    if not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"MoLFormer snapshot missing: {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(snapshot, deterministic_eval=True, trust_remote_code=True).eval().cuda()
    max_length = int(model.config.max_position_embeddings)
    raw = tokenizer(smiles, add_special_tokens=True, truncation=False)
    too_long = [idx for idx, ids in enumerate(raw["input_ids"]) if len(ids) > max_length]
    if too_long:
        raise ValueError(f"{len(too_long)} SMILES exceed MoLFormer length {max_length}")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(smiles), cfg["batch_size"]):
            batch = tokenizer(smiles[start:start + cfg["batch_size"]], padding=True, return_tensors="pt")
            batch = {key: value.cuda() for key, value in batch.items()}
            chunks.append(model(**batch).pooler_output.float().cpu().numpy())
    weights = snapshot / "model.safetensors"
    return np.concatenate(chunks).astype(np.float32), {
        "snapshot": str(snapshot),
        "weights_sha256": sha256(weights),
        "max_position_embeddings": max_length,
        "overlength_smiles": 0,
    }


def unimol(smiles, cfg):
    from importlib.metadata import version
    from unimol_tools import UniMolRepr
    import unimol_tools

    initial_log_sizes = {path: path.stat().st_size for path in (ROOT / "logs").glob("unimol_tools_*.log")}
    model = UniMolRepr(
        data_type="molecule",
        remove_hs=cfg["remove_hs"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )
    model.params["multi_process"] = True
    chunks = []
    failure_indices = []
    log_path = None
    log_position = 0
    chunk_size = 1024
    for start in range(0, len(smiles), chunk_size):
        result = model.get_repr(smiles[start:start + chunk_size], return_atomic_reprs=False)
        chunks.append(np.asarray(result, dtype=np.float32))
        logs = sorted((ROOT / "logs").glob("unimol_tools_*.log"), key=lambda path: path.stat().st_mtime)
        if logs:
            current = logs[-1]
            if current != log_path:
                log_path = current
                log_position = initial_log_sizes.get(current, 0)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            new_text = text[log_position:]
            log_position = len(text)
            for match in re.findall(r"Failed 3d conformers indices: \[([^]]*)\]", new_text):
                failure_indices.extend(start + int(value.strip()) for value in match.split(",") if value.strip())
    weights = Path(unimol_tools.__file__).parent / "weights" / cfg["weights"]
    return np.concatenate(chunks), {
        "package_version": version("unimol_tools"),
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "conformer_seed": cfg["seed"],
        "remove_hs": cfg["remove_hs"],
        "conformer_workers": 8,
        "extraction_chunk_size": chunk_size,
        "conformer_failures": len(failure_indices),
        "conformer_failure_indices": failure_indices,
        "conformer_log": str(log_path) if log_path else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--arms", nargs="+", choices=["morgan", "molformer", "unimol"], default=["morgan", "molformer", "unimol"])
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    frame = pd.read_csv(args.input)
    smiles = frame.canonical_smiles.astype(str).tolist()
    row_ids = np.asarray(frame.row_id.astype(str).tolist(), dtype="U")
    args.output.mkdir(parents=True, exist_ok=True)
    extractors = {"morgan": morgan, "molformer": molformer, "unimol": unimol}
    manifest = {"input": str(args.input), "input_sha256": sha256(args.input), "rows": len(frame), "arms": {}}
    for arm in args.arms:
        arm_cfg = cfg["representations"][arm]
        values, provenance = extractors[arm](smiles, **arm_cfg) if arm == "morgan" else extractors[arm](smiles, arm_cfg)
        if values.shape[0] != len(row_ids) or not np.isfinite(values).all():
            raise ValueError(f"invalid {arm} embedding array {values.shape}")
        if arm == "unimol":
            provenance["conformer_failure_row_ids"] = [str(row_ids[idx]) for idx in provenance["conformer_failure_indices"]]
        path = args.output / f"{arm}.npz"
        np.savez_compressed(path, row_id=row_ids, x=values)
        manifest["arms"][arm] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite": True,
            "sha256": sha256(path),
            **provenance,
        }
        print(f"{arm}: {values.shape}", flush=True)
        if arm != "morgan":
            torch.cuda.empty_cache()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
