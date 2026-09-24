# Executed environment and external model identifiers

## Core environment

- operating system: Windows
- Python: 3.12.0
- NumPy: 1.26.4
- pandas: 2.3.3
- SciPy: 1.17.1
- scikit-learn: 1.8.0
- PyTorch: 2.11.0+cu128
- XGBoost: 3.3.0
- RDKit: 2026.3.3
- transformers: 5.10.2
- Matplotlib: 3.10.8
- CReM: 0.2.14
- unimol-tools: 0.1.6

`requirements.txt` records package pins. Install a PyTorch build appropriate for the host CUDA runtime when rerunning GPU-dependent stages.

## Representation identifiers

- Morgan: radius 2, 2,048 binary bits
- MoLFormer: `ibm-research/MoLFormer-XL-both-10pct`
- MoLFormer revision: `361063d0ad524ef77cf39b08469f6be770dc550f`
- Uni-Mol weights: `mol_pre_all_h_220816.pt`
- Uni-Mol representation dimension: 512
- MoLFormer weights SHA-256: `0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6`
- Uni-Mol weights SHA-256: `7f5f14bb28bf479a0b8f38ebab4bb9d7ae8673383f14c4d694d056500fa3446a7`

Third-party pretrained weights are not redistributed.

## Randomization and resampling

- observed-label model seeds: 11, 23 and 42
- revised main bootstrap seed: 20260903; 5,000 draws
- independent permutation IDs: 42 through 51
- permutation seed: `20260902 + 1000 * source_target_index + permutation_id`
- permutation-stability model initialization: fixed at 42
- component-split seeds: 20260903 and 20260904
- Uni-Mol conformer seed: 42
- full 27-target alternative-sentinel bootstrap: seed 20260904; 5,000 draws

## Executed hardware and resource record

- operating system: Microsoft Windows 11 Pro
- CPU: AMD Ryzen 9 9950X, 16 cores / 32 logical processors
- physical memory: 31.1 GiB
- GPU: NVIDIA GeForce RTX 5070, 12,227 MiB reported memory
- NVIDIA driver: 616.64
- full 27-target alternative-sentinel elapsed time after preparation began: approximately 42 minutes with existing model and counterfactual caches
- incremental round-2 sentinel artifacts: 406,096,813 bytes, excluding the already-cached source data, pretrained assets, representations and model heads

This is the environment in which the reported run was executed, not a tested minimum specification. A fresh end-to-end refit requires additional time and storage for the unbundled pretrained weights, representations, model binaries, CReM database and mutant caches; that fresh-install resource envelope was not benchmarked.

GPU kernels and third-party conformer generation may not be bitwise identical across hardware, drivers and library builds. Row-level statistical inputs and exact executed hashes are therefore distributed alongside full-rerun code.
