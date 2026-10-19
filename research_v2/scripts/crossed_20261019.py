import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research_v2/artifacts/experiment/crossed_20261019'
OLD = ROOT / 'research_v2/artifacts/experiment/xai_sanity_cliffs_major_revision_v1'
JOINT = OLD / 'round2_joint_uncertainty_v1'
RUNNER = ROOT / 'research_v2/scripts/run_permutation_stability.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.time()
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'CHECKLIST.md').write_text('# Crossed initialization and permutation\n\nFrozen contract: ../../analysis/extension_20261019/PLAN.md\n\n- Smoke replay and prediction agreement\n- Preserve all historical caches\n- Complete 2430 heads, with 720 newly fitted\n- Verify shared label vectors and unit hashes\n- Export test-endpoint atom scores without retraining\n- Quantify separate observed/fixed-permutation initialization noise\n', encoding='utf-8')
    old = JOINT / 'by_initialization/init11/units/CHEMBL1862_Ki--morgan--mlp--perm42'
    new = OUT / 'smoke/units/CHEMBL1862_Ki--morgan--mlp--perm42'
    smoke = {}
    for name in ['predictions.csv', 'prediction_metrics.csv', 'pair_predictions.csv', 'pair_explanations.csv', 'model_output_audit.csv']:
        a, b = pd.read_csv(old / name), pd.read_csv(new / name)
        if a.shape != b.shape or a.columns.tolist() != b.columns.tolist():
            raise ValueError(f'smoke schema mismatch {name}')
        numeric = a.select_dtypes(include='number').columns
        difference = np.abs(a[numeric].to_numpy(float) - b[numeric].to_numpy(float))
        smoke[name] = {'rows': len(a), 'maximum_absolute_difference': float(np.nanmax(difference)), 'bytes_exact': sha(old / name) == sha(new / name)}
        if not np.allclose(a[numeric], b[numeric], rtol=0, atol=1e-7, equal_nan=True):
            raise ValueError(f'smoke numerical mismatch {name}: {smoke[name]}')
    (OUT / 'smoke_comparison.json').write_text(json.dumps(smoke, indent=2))
    cached = {}
    source_hashes = {}
    for init in [11, 23]:
        origin = JOINT / f'by_initialization/init{init}'
        dest = OUT / f'init{init}'
        dest.mkdir(exist_ok=True)
        config = JOINT / f'derived_configs/init{init}.json'
        source_hashes[str(config.relative_to(ROOT))] = sha(config)
        unit_count = 0
        for manifest_path in sorted((origin / 'units').glob('*/unit_manifest.json')):
            manifest = json.loads(manifest_path.read_text())
            if manifest['config_sha256'] != sha(config):
                raise ValueError('cached config mismatch')
            for name, expected in manifest['output_hashes'].items():
                if sha(manifest_path.parent / name) != expected:
                    raise ValueError('cached output hash mismatch')
            if sha(ROOT / manifest['model']['path']) != manifest['model']['sha256']:
                raise ValueError('cached model hash mismatch')
            shutil.copytree(manifest_path.parent, dest / 'units' / manifest_path.parent.name, dirs_exist_ok=True)
            source_hashes[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
            unit_count += 1
        cached[str(init)] = unit_count
        command = [sys.executable, str(RUNNER), '--config', str(config), '--output', str(dest)]
        with (OUT / f'init{init}_console.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (OUT / 'failure.json').write_text(json.dumps({'initialization': init, 'returncode': result.returncode, 'command': command}, indent=2))
            raise SystemExit(result.returncode)
    sources = [(42, OLD / 'permutation_stability'), (11, OUT / 'init11'), (23, OUT / 'init23')]
    counts = {}
    for table in ['model_manifest.csv', 'pair_explanations.csv', 'pair_predictions.csv', 'prediction_metrics.csv', 'model_output_audit.csv']:
        frames = []
        for init, path in sources:
            frame = pd.read_csv(path / table)
            frame = frame[frame.permutation_id.isin([42, 43, 44, 45, 46])]
            if 'explainer' in frame:
                frame = frame[frame.explainer.eq('crem_mean')]
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        frame.to_csv(OUT / ('consolidated_' + table), index=False)
        counts[table] = len(frame)
    models = pd.read_csv(OUT / 'consolidated_model_manifest.csv')
    keys = ['dataset', 'representation', 'predictor', 'model_initialization_seed', 'permutation_id']
    assert len(models) == 2430 and not models.duplicated(keys).any()
    for row in models.itertuples():
        assert sha(ROOT / row.path) == row.sha256
    labels = []
    for init, path in sources:
        x = pd.read_csv(path / 'permuted_labels.csv').drop(columns=['model_initialization_seed'])
        labels.append(x[x.permutation_id.isin([42, 43, 44, 45, 46])].sort_values(['dataset', 'row_id', 'permutation_id']).reset_index(drop=True))
    assert labels[0].equals(labels[1]) and labels[0].equals(labels[2])
    assert all(sha(ROOT / p) == v for p, v in source_hashes.items())
    manifest = {'status': 'pass', 'elapsed_seconds': time.time() - started, 'new_models': 720, 'cached_additional_initialization_models': cached, 'cached_init42_models': 810, 'counts': counts, 'label_rows_equal': len(labels[0]), 'source_hashes': source_hashes, 'script_sha256': sha(__file__), 'runner_sha256': sha(RUNNER), 'smoke': smoke}
    (OUT / 'training_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
