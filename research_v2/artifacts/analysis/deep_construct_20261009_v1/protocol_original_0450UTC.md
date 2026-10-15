# Deep construct analysis: frozen protocol

Locked 2026-10-09 04:50 UTC, before calculating the new mechanistic outcomes. User authorizes new model fitting and/or independent-data experiments; first use exact functions because they remove prediction error without adding model-performance ambiguity.

Parent: the rerevised JCheminformatics manuscript and public code commit 30cb2cc2b3e7871218acdd163e58d51a887dce3f. Current finding: CReM structural-proxy localization shows no observed-label advantage; the exact heteroatom-count oracle has weak localization. Existing results remain immutable.

## Question and competing explanations

Can weak exact-count localization be distinguished between (a) concept creation at currently negative atom sites and cancellation during replacement, and (b) diffusion/indistinguishability imposed by the mask-to-atom averaging operator? A secondary diagnostic tests response-location specificity using a conditional response permutation reference with fixed masks and response distribution.

This is a new retrospective mechanism-oriented analysis of the three existing development datasets plus a separately fixed external molecule panel. It does not retrospectively restore prospective status to the manuscript. No biochemical causal mechanism is inferred.

## D1: known-function decomposition on existing development molecules

Use the exact original retained mutants and mask atoms in reprxai_cliffs_pilot_v1/full/counterfactuals/scored_mutants.csv, with molecules from its prepared/molecules.csv. Test molecules eligible for each of the original three concepts are the primary population: halogen (F/Cl/Br/I), heteroatom (N/O/P/S), aromatic atoms. Require both positive and negative current-concept atoms. Keep unscored/unsupported cases explicitly excluded or flagged; do not substitute new mutants.

For each mask and mutant, R is the number of concept-positive original mask atoms; delta is mutant count minus parent count; I = delta + R. For elemental counts, I is an insertion accounting quantity if nonnegative. Negative I flags nonlocal change or inadequate mask bookkeeping and prohibits interpreting I as an inserted count for that row. Aromatic I is accounting only because aromaticity can change outside the mask.

Compare four responses through exactly the same incidence matrix and row-normalized mask-to-atom operator: actual |delta|; removal-only R; insertion-accounting max(I,0); gross turnover R+max(I,0). Removal-only is an explicitly artificial label-aligned mask response, not a chemically valid deletion experiment or biological oracle. It can diagnose averaging losses, but its use of the known labels is intentional and cannot support an improvement claim on bioactivity data.

Primary contrast: molecule-level removal-only minus actual nAP for heteroatom count. Report component-balanced dataset medians and dataset-balanced median, with 5000 component-within-dataset bootstrap draws, seed 20261009. Repeat raw molecule median to reproduce the original oracle summary. Primary mechanistic claim requires a positive 95% interval and point contrast at least 0.10; this is an author-defined decision gate, not a validated effect-size standard. Preserve contrary and inconclusive outcomes.

Report creation (R=0,I>0), destruction (R>0,I=0), cancellation (R=I>0) and other turnover counts, their absolute-response mass, leakage to currently negative atoms, support, mask sizes, number of distinct atom-incidence rows, incidence rank and proportion of positive/negative pairs forced to tie by identical rows. For the latter, AUC <= 1 - 0.5 * sum_g(P_g*N_g)/(P*N) is a loose structural upper bound, not an attained optimum or a bound on biochemical explanations. Reconstruction tests verify that identical rows always receive identical scores.

Conditional response permutation: 256 permutations of mask responses within equal mask size, separately within molecule and concept. Keep mask incidence, atom labels, support and response multiset fixed. Report original nAP minus mean permuted nAP and non-permutable strata. This is an operator reference, not a valid hypothesis-test p value under an established mask-exchangeability model. Use shared deterministic permutations across response variants. Do not call the contrast an additive causal contribution.

## D2: independent molecular panel

Source: dmtsh/wombat-smiles, Hugging Face revision e86ec35687b0c0ebd1ecc151610a08224d8cf851; fixed file validation_datasets/Pattern 1/validation_molecules.parquet (157820200 bytes). Repository reference gmum/wombat commit 40a46639713eba3bc4fa175f5035392c0791fd8d; dataset declares CC0 1.0. This is a molecule-source reuse, not reproduction of WOMBAT's 14 whitebox-GNN benchmark. The source is a September 2026 preprint, not treated as peer-reviewed evidence.

Choose 256 valid, single-fragment molecules with 8-60 heavy atoms and elements C/N/O/F/P/S/Cl/Br/I. Exclude every canonical structure in all 30 MoleculeACE source tables and duplicates. Rank candidates by SHA256('deep-construct-20261009:' + canonical_smiles), without evaluating localization. Preserve the selected list before generation. No replacement of failed or unsupported molecules.

Use the existing frozen development-training CReM library and its original candidate-mask algorithm: radius 1, at most 8 replaced atoms, max 8 generated replacements, min frequency 2, size increments -2..2, parent Morgan similarity >=0.60, retain up to 4 mutants by fixed SHA256 order. Write a fresh external cache; original caches remain unchanged. Reference optimized fragmentation is reused with its prior equality validation. Record generation failures and empty supports.

Apply exactly D1 responses/metrics to the external molecules. Use a 5000-draw molecule bootstrap (seed 20261009). Source selection is motif-conditioned and not a representative sample of all PubChem chemistry. The external panel tests transport of a computational accounting/averaging phenomenon; it supplies no new activity, potency-direction or biological validation.

Independent support gate: primary heteroatom contrast >=0.10 and lower 95% bound >0, at least 50 eligible supported molecules, with no negative elemental insertion-accounting rows. If unmet, do not claim independent confirmation. Halogen and aromatic results remain reported even when eligibility is low or hypotheses fail.

## Resource and stop boundaries

Executed host: 32 GB RAM, NVIDIA RTX 5070 12 GB (about 9.7 GB free at intake), about 586 GB free on D:. Python 3.12 .venv-joc includes RDKit/CReM/NumPy/pandas/SciPy/PyArrow. Initial slices run on CPU, one generator process and up to four numeric threads; no environment package upgrades or unbounded grid search. External input is about 158 MB. Allow up to 45 minutes for first fixed generation stage; if it exceeds the envelope, preserve partial status and report the reduced population without changing the 256-molecule selection.

Stop widening if two consecutive slices fail to clarify a parent claim. Do not rescue-tune on the external panel. New model training remains authorized but is deferred until exact-function results identify a question for which training adds information. No automatic manuscript or remote-repository update from favorable results.

## Evidence and write-back

Outputs: input_provenance.json, selected_external_molecules.csv, generation cache/failure records, molecule-level and mask-level metrics, cohort/concept summaries, bootstrap contrasts, conditional-response diagnostics, structural operator diagnostics, evaluation_summary.json, figure-source CSVs and a Chinese evidence report. Integrity check must reproduce the existing oracle's raw test medians before promoting any new result.

Paper links: E10 exact-function diagnostic and E5 generator controls; possible Methods subsection on mask response/assignment audit; Results on concept creation/cancellation and operator limits; Discussion on proxy semantics. New results are separate post-submission-preparation analyses and do not replace the frozen main estimand. A result becomes claim-carrying only after metric reproduction, accounting validity and the independent gate; otherwise keep it supporting, negative or unresolved.

Literature boundary: WISP (10.1039/D5DD00398A) already supplies model-agnostic atom mutation/MMP evaluation; WOMBAT (arXiv:2610.00713) already uses whitebox models to distinguish model and explainer errors. Proposed increment is the controlled replacement-accounting and fixed-mask response/assignment diagnosis. No first-oracle or first-perturbation-attributor claim is permitted.
