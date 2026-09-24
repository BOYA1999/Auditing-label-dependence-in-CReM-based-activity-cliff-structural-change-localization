# Independent label-permutation stability

## Run contract

- Scope: the 27 untouched confirmation targets and the complete 3 representation x 2 predictor x 2 explainer crossing.
- Intervention: ten independent within-target label permutations (IDs 42--51) over train, isolated and calibration rows.
- Isolation: model initialization was fixed at seed 42 for every fit; permutation seeds followed `20260902 + 1000 * source_target_index + permutation_id`.
- Reuse: permutation ID 42 was verified row by row against the original run and its 162 matching heads and explanations were reused read-only. IDs 43--51 were newly fitted.
- Endpoint: test-pair normalized AP lift for CReM-mean and CReM-LIME, plus signed pair-delta prediction metrics.

## Completion and validation

- Status: PASS.
- Models: 1,620/1,620 (162 verified reuse; 1,458 new fits).
- Pair-explanation rows: 144,120; all normalized AP lift values were finite.
- Every randomized head was finite and nonconstant on its actual parent-plus-mutant explanation domain.
- Every target/configuration had all ten permutations for both explanation and pairwise prediction summaries.
- Model SHA-256 values, 1,620 unit manifests and their output hashes passed validation; no duplicate pair-explanation key was found.

## Explanation stability

The target-balanced trained-minus-random nAP estimate was 0.000315 with one permutation, -0.015453 with three, -0.018152 with five and -0.018601 with ten. Absolute deviations from the ten-permutation estimate were 0.018916, 0.003148 and 0.000449 for one, three and five permutations, respectively. Thus the single reused permutation was not representative, whereas the pooled estimate was nearly stable by five permutations.

Across the 12 representation-predictor-explainer configurations, the standard deviation of target-balanced randomized nAP across ten permutations ranged from 0.00859 to 0.02318 (median 0.01414). At the more granular 324 target-configuration level, the standard deviation of the ten permutation-specific medians ranged from 0.01036 to 0.24911 (median 0.05112); 168/324 exceeded 0.05. This distinction is important: target balancing damped substantial target-local permutation variability.

At ten permutations, configuration-specific target-balanced trained-minus-random nAP ranged from -0.08644 (Morgan-XGBoost-CReM-LIME) to -0.00288 (Uni-Mol-MLP-CReM-mean). The pooled value therefore must not be used as evidence that every configuration behaves similarly.

## Pairwise prediction stability

Across permutations, target-balanced pair-delta Spearman correlations remained near zero (configuration means -0.0070 to 0.0145), while target-balanced delta MAE means ranged from 1.7691 to 1.7790 and delta RMSE means from 1.8911 to 1.9048. Twenty-four target-configuration-permutation Spearman values were undefined because the test subset or prediction range did not support a rank correlation; these rows remain explicit and are not replaced with zero.

When predicted pair deltas were averaged across all ten permutations, target-balanced direction accuracy ranged from 0.474 to 0.524 across the six representation-predictor configurations and delta MAE ranged from 1.7620 to 1.7672. These results confirm that label randomization disrupted pairwise prediction information even though structural-change localization scores remained nonzero.

## Interpretation boundary

This run isolates label-permutation variability at one fixed model initialization. It does not estimate the joint distribution over label permutations and independent model initializations. The result supports using multiple independent permutations and shows that the ten-permutation pooled estimate is stable, while target/configuration heterogeneity remains too large for a universal configuration-level conclusion.

## Evidence pointers

- `summary.json`: completion and global explanation-stability checks.
- `validation.json`: explicit PASS gates.
- `permutation_nap_stability_by_target_configuration.csv`: ten-permutation nAP variability for all 324 target/configuration cells.
- `nap_convergence_by_configuration.csv` and `nap_convergence_pooled.csv`: 1/3/5/10 convergence.
- `pairwise_stability_by_target_configuration.csv` and `pairwise_convergence_by_configuration.csv`: pair-delta prediction controls.
- `model_manifest.csv`, `units/*/unit_manifest.json` and `artifact_manifest.csv`: model and output hashes.
