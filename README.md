# LLM-Assisted Content Analysis Replication Materials

This repository contains computational materials associated with a study of
LLM-assisted content analysis of parliamentary discourse.

## Repository structure

- `code/` – Python scripts used in the computational pipeline.
- `prompts/` – prompts used in the LLM-assisted analytical stages.
- `data/` – documentation and, when appropriate, shareable analytical data.

## Computational pipeline

The public implementation covers the following stages:

1. claim extraction;
2. analytical filtering;
3. weak semantic normalization;
4. multi-resolution semantic clustering.

The final HDBSCAN analysis uses three resolutions:

- **Macro:** `min_cluster_size=40`, `min_samples=15`
- **Meso:** `min_cluster_size=25`, `min_samples=10`
- **Micro:** `min_cluster_size=12`, `min_samples=6`, `cluster_selection_method="leaf"`

Taxonomic interpretation and validation remain researcher-led analytical stages.

## Reproducibility

The Python environment can be recreated using the packages listed in
`requirements.txt`.

API credentials are not stored in this repository. When required, the Gemini
API key must be supplied through the `GEMINI_API_KEY` environment variable.

## Data availability

The final public data package and persistent repository identifier will be added
after verification of the shareable analytical datasets.

## Citation

Citation metadata will be finalized together with the article metadata and the
archived release of the replication package.
