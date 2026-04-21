# Claim-Aware Propaganda Scanner

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Addressing **SemEval-2020 Task 11**, this project provides a high-performance framework for detecting likely propaganda techniques in news articles. While many projects remain theoretical, this repository includes a functional **Chrome Extension** to bring model inference into real-world browsing.

## Key Features

- **High-Performance Models:** Benchmarked against SemEval-2020 Task 11, our models outperform previous competition attempts in identifying fine-grained propaganda techniques.
- **Real-Time Chrome Extension:** A browser tool that scans the active page, sending text to our models to highlight potential propaganda fragments in situ.
- **Claim-Aware Extraction:** A lightweight claim extraction tool—heavily inspired by *Microsoft's Claimify*—designed to identify core assertions and prepare text for fact-checking.
- **End-to-End EDA:** Extensive Exploratory Data Analysis notebooks that visualize linguistic patterns and technique distributions across the SemEval dataset.

## The Task: SemEval-2020 Task 11

Propaganda uses psychological and rhetorical techniques to influence audiences. Our models address two primary subtasks:
1. **Span Identification (SI):** Highlighting the specific text fragments where propaganda occurs.
2. **Technique Classification (TC):** Labeling fragments with one of 14 techniques (e.g., *Loaded Language, Appeal to Fear, Slogans, etc.*).

## Project Organization


```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
││
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         claim_aware_propaganda_scanner and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
││
└── create-model       <- Source code for use in this project.
    ├── claim-utils.py                <- Lightweight claim extraction logic (Claimify-inspired)
    │
    ├── data-pull.py                  <- Scripts to download and prepare raw data
    │
    ├── gdown-download-models.py      <- Utility to fetch pre-trained weights via Google Drive
    │
    ├── helpers.py                    <- General utilities and model pipeline processing endpoints
    │
    ├── is-subjective.py              <- Feature engineering: specialized subjectivity detection
    │
    ├── run-semeval-create-pipeline.py <- Orchestration script for the end-to-end model pipeline
    │
    ├── semeval-clean.py              <- Data preprocessing and text cleaning scripts
    │
    ├── semeval-create-si-model.py     <- Code for Span Identification (SI) task
    │
    ├── semeval-create-tc-model.py     <- Code for Technique Classification (TC) task
    │
    └── semeval-create-specialist-model.py <- Specialist model to help with SI on types that are frequently missed
```

--------

