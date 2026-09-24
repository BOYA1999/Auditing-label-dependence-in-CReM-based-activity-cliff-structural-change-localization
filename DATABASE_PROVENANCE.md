# Database and data provenance

## Redistributed benchmark data

The directory `external/MoleculeACE/MoleculeACE/Data/benchmark_data/` contains the 30 assay-level CSV files used by the frozen study configuration. They were obtained from the public MoleculeACE repository:

- Repository: <https://github.com/molML/MoleculeACE>
- Pinned upstream commit: `7e6de0bd2968c56589c580f2a397f01c531ede26`
- Study access date: 1 September 2026
- Upstream licence and README: `external/MoleculeACE/LICENCE` and `external/MoleculeACE/README.md`
- Benchmark collection: the MoleculeACE activity-cliff benchmark tables, with assay identifiers such as `CHEMBL214_Ki` and `CHEMBL235_EC50`

The files are redistributed unchanged from that pinned upstream snapshot. Their rows contain molecular SMILES and assay measurements/split annotations; they do not contain patient-level, clinical, contact, institutional, or other private information.

## Original bioactivity source

MoleculeACE documents ChEMBL as the original public bioactivity source. The relevant database citation is:

Zdrazil B, et al. The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. *Nucleic Acids Research*. 2024;52(D1):D1180-D1192. DOI: <https://doi.org/10.1093/nar/gkad1004>.

This study did not directly query ChEMBL. The fixed MoleculeACE assets do not record the historical ChEMBL release, extraction query, or access snapshot corresponding to each redistributed assay table. Those details cannot be reconstructed from the redistributed tables, and the study does not claim a specific ChEMBL release.

This repository does not claim to be an independent ChEMBL release or to reproduce the entire ChEMBL database. It distributes only the benchmark tables needed by this study and preserves the upstream MoleculeACE licence notice.

## Derived study files

Files under `research_v2/artifacts/` are derived computational products: frozen molecule lists, sentinel pairs, counterfactual audit tables, model-output audits, atom-level explanation rows, and compact analysis summaries. They are generated from the pinned benchmark tables and the protocol in `research_v2/artifacts/idea_sanity_benchmark/objective_contract.md`.

No private data, human participants, patient records, clinical records, or institution-only databases were used. No database credentials or access tokens are required for the compact verification route.

## Third-party model/data dependencies

Full reruns may download public pretrained weights listed in `environment.md` and `research_v2/configs/sanity_full_v1.json`, including MoLFormer and Uni-Mol assets. These weights are not redistributed here; users must obtain them from their respective public sources and comply with their terms. CReM is rebuilt locally from the training partition when a full rerun is attempted.

## Reuse and licensing boundary

The upstream MoleculeACE files remain governed by `external/MoleculeACE/LICENCE`. The study scripts and derived files do not yet carry an author-selected public software licence; `LICENSE_PENDING.txt` is an explicit release gate. Select and add a compatible licence before publishing the repository.
