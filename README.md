# LLM-Assisted Content Analysis Replication Materials

Replication materials for an LLM-assisted content analysis of Brazilian parliamentary discourse.

## Repository structure

- `code/` – Python scripts used in the computational pipeline.
- `prompts/` – prompts used in the LLM-assisted analytical stages.
- `data/` – analytical datasets and files used to reproduce the study results.

## Analytical pipeline

The computational workflow consists of:

1. classification of parliamentary speeches according to land and territorial relevance;
2. identification of delegitimizing discourse;
3. identification of environmental and climate-related repertoires;
4. extraction of argumentative claims;
5. claim-level analytical filtering;
6. weak semantic normalization;
7. multi-resolution semantic clustering with HDBSCAN;
8. researcher-led taxonomic interpretation and validation.

## Multi-resolution clustering

The final HDBSCAN analysis uses three levels of semantic granularity:

- **Macro:** `min_cluster_size=40`, `min_samples=15`
- **Meso:** `min_cluster_size=25`, `min_samples=10`
- **Micro:** `min_cluster_size=12`, `min_samples=6`, `cluster_selection_method="leaf"`

## Reproducibility

Python dependencies are listed in `requirements.txt`.

API credentials are not stored in this repository. When required, the Gemini API key must be provided through the `GEMINI_API_KEY` environment variable.

## Data

Analytical datasets and files required to reproduce the reported results are provided in the `data/` directory or through the associated archival repository.
