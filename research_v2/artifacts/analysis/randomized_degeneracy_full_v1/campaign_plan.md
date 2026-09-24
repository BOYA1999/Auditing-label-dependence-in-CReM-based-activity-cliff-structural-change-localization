# Randomized endpoint-degeneracy campaign

- campaign id: `randomized_degeneracy_full_v1`
- trigger: all 1,080 checkpoints and finite predictions were produced, but three of 540 label-permuted models had exactly one unique prediction across target test endpoints
- affected cells: confirmation `CHEMBL1871_Ki / Morgan / XGBoost / seed42`, development `CHEMBL234_Ki / Morgan / XGBoost / seed23`, and confirmation `CHEMBL2835_Ki / Uni-Mol / XGBoost / seed23`
- frozen model policy: no retraining, seed replacement, hyperparameter change or checkpoint deletion
- construct clarification: the original nonconstant-output gate did not name an evaluation domain. Before any full explanation score is computed, the decision-relevant domain is fixed as the union of target evaluation parents and their valid scored CReM mutants, because these are exactly the outputs consumed by both explainers
- final gate: every randomized checkpoint must be finite and have local-domain prediction range greater than `1e-8`
- mandatory sensitivity: repeat confirmation inference after excluding every target/configuration cell containing an endpoint-constant randomized replicate, regardless of the local-domain result
- reportable failure mode: endpoint collapse under label permutation and early stopping remains a named result even if local mutant predictions are nonconstant
