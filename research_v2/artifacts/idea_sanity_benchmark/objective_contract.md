# Objective contract: activity-cliff XAI sanity benchmark

- stable idea id: `xai_sanity_cliffs_v1`
- article type: hypothesis-driven Research Article
- direction: B — explanation differences and failure modes
- data: public MoleculeACE with component-disjoint sentinel pairs
- development targets: three already inspected targets
- confirmation targets: all remaining eligible targets, still unopened

## Primary question

Do atom-level activity-cliff localization scores respond to learned bioactivity labels, or do they remain equivalent under matched label randomization because representation and perturbation structure dominate the explanation?

## Primary confirmatory estimand

For every component pair and matched trained/permuted cell:

`Delta_sanity = nAP_trained - mean(nAP_label_permuted)`.

The confirmatory null-of-practical-difference interval is `[-0.05, 0.05]` normalized AP lift. The primary target-balanced bootstrap tests whether the 95% interval for the pooled median `Delta_sanity` lies inside this equivalence region. This margin is frozen before any confirmation-target explanation is computed and represents five percentage points of prevalence-normalized localization.

The pooled point estimator is the median of 27 target-specific medians, where each target median contains all test components, 12 representation/predictor/explainer configurations and three trained seeds. Each trained-seed value subtracts the mean of the three permuted-seed values for the same component and configuration. The two-stage bootstrap resamples targets and then component IDs within each sampled target; all factor and seed rows for a sampled component remain together.

## Factor panel

- representations: Morgan, MoLFormer, Uni-Mol
- predictors: XGBoost, MLP
- explainers: mean absolute CReM replacement effect and CReM-LIME over the same valid chemical neighborhood
- trained seeds: 11, 23, 42
- label-permuted counterparts: matched seeds and optimization contracts
- label permutations: within each target, jointly permute component-train, isolated and calibration labels with seed `20260902 + 1000 * target_index + replicate`; the same replicate is reused across representations and predictors
- parameter-randomized MLP: secondary development and confirmation subset
- deterministic count concepts: independent construct-failure analysis, not a gate for claiming explanation quality

## Hypotheses

- H1: raw changed-region localization remains practically equivalent after label randomization for a substantial fraction of factor cells and targets.
- H2: trained and randomized atom rankings remain positively correlated, indicating persistent structural/algorithmic priors.
- H3: randomization sensitivity depends on representation × predictor × explainer and target, explaining why bundled pipelines appear to disagree.
- H4: prediction quality and randomization sensitivity are weakly associated; accurate prediction does not guarantee label-sensitive explanations.

## Competing hypotheses

- C1: trained localization consistently exceeds randomized localization by more than 0.05, so the benchmark passes the sanity test.
- C2: survival occurs only for one explainer or one representation and is not general.
- C3: randomized models have degenerate outputs, making the comparison uninterpretable.

## Execution gates

1. every matched random model has finite, nonconstant predictions over the local explanation domain (target evaluation parents plus valid scored CReM mutants) and an auditable checkpoint; endpoint-only collapse is retained as a separate failure mode and exclusion sensitivity;
2. component, SMILES, representation row and mutant hashes pass;
3. every confirmation cell has three trained and three label-permuted fits;
4. target-balanced and component bootstrap estimators are frozen before confirmation;
5. both equivalence and non-equivalence outcomes are reportable; no threshold revision is allowed.

## Frozen rank-agreement estimator

For H2, Spearman agreement is computed first on atoms supported by the explainer in either compared map, so shared unsupported zeros cannot create agreement. The all-atom correlation is retained only as a sensitivity analysis. Each trained seed is compared with its matched permuted seed and with the mean score map across all three permuted fits.

## Frozen secondary equivalence estimator

For each target x representation x predictor x explainer cell, trained and permuted nAP are first averaged across their three seeds within component. A 5,000-draw component bootstrap yields the cell interval. A cell is practically equivalent only when its full 95% interval lies inside `[-0.05, 0.05]`; the frozen benchmark-level secondary threshold is at least two thirds of the 27 x 12 confirmation cells. Factor interactions use the component-level seed-averaged differences with target fixed effects and target-clustered standard errors. Prediction-quality association uses trained test RMSE and target-demeaned Spearman correlation.

## Claim boundary

The study audits sensitivity of a structural localization benchmark to learned labels. It does not infer biochemical mechanisms, declare all molecular XAI invalid, or treat randomized equivalence as proof that individual predictions are useless.
