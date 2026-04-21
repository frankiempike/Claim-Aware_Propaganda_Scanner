# Claim-Aware Propaganda Scanner

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Addressing **SemEval-2020 Task 11**, this project provides a high-performance framework for detecting likely propaganda techniques in news articles. While many projects remain theoretical, this repository includes a functional **Chrome Extension** to bring model inference into real-world browsing.

### Table of Contents
* [Key Features](#key-features)
* [The Task](#the-task-semeval-2020-task-11)
* [Running the Code](#running-the-code)
* [Data Access](#data-access)
* [Dependency Management](#dependency-management)
* [Project Organization](#project-organization)

## Key Features

- **High-Performance Models:** Benchmarked against SemEval-2020 Task 11, our models outperform previous competition attempts in identifying fine-grained propaganda techniques.
- **Real-Time Chrome Extension:** A browser tool that scans the active page, sending text to our models to highlight potential propaganda fragments in situ.
- **Claim-Aware Extraction:** A lightweight claim extraction tool—heavily inspired by Microsoft's Claimify—designed to identify core assertions and prepare text for fact-checking.
- **End-to-End EDA:** Extensive Exploratory Data Analysis notebooks that visualize linguistic patterns and technique distributions across the SemEval dataset.

## The Task: SemEval-2020 Task 11

Propaganda uses psychological and rhetorical techniques to influence audiences. Our models address two primary subtasks:
1. **Span Identification (SI):** Highlighting the specific text fragments where propaganda occurs.
2. **Technique Classification (TC):** Labeling fragments with one of 14 techniques (e.g., *Loaded Language, Appeal to Fear, Slogans, etc.*).

## Running the Code

#### Prerequisites

Make sure you have [Poetry](https://python-poetry.org/docs/#installation) installed (Python ≥ 3.13 required), then install all dependencies from the lockfile:

```bash
git clone https://github.com/frankiempike/Claim-Aware_Propaganda_Scanner.git
cd Claim-Aware_Propaganda_Scanner
poetry install
```

#### Running the full model pipeline from scratch

Then, populate you need to make the models available to your machine.

You have the option to build the models locally on your machine:

```bash
# Build all models end-to-end (SI model, TC model, specialist model)
poetry run create-pipeline
```

OR download the pre-built model weights from Google Drive (contact Frankie Pike for access to the Drive link):

To do this you need to set these variables in the environment file

```bash
GOOGLE_DRIVE_SI_ZIP_ID = '<GOOGLE DRIVE ID FOR SPAN IDENTIFICATION MODEL>'
GOOGLE_DRIVE_SPEC_ZIP_ID = '<GOOGLE DRIVE ID FOR SPECIALIZATION MODEL>'
GOOGLE_DRIVE_TC_CONTEXT_ZIP_ID = '<GOOGLE DRIVE ID FOR TECHNIQUE CLASSIFICATION MODEL>'
```

We provide the Google Drive ID's we used for our models in our .env.example. If you choose to make your own models and host them on Google Drive, you can replace those values with the Google Drive ID's of your own models.

If you have trouble accessing our models, contact Frankie Pike at frankiep@umich.edu.

```bash
poetry run create-gdown
```

#### Quickstart: scan a string of text for propaganda

Once you have the models available (either built locally or downloaded via gdown), you can run inference directly against the Flask API. Start the server:

```bash
poetry run deploy-flask-app
```

Then in a second terminal, send a text snippet to the `/process` endpoint:

```bash
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{"text": "They are coming for your children. Real patriots know the truth and will not be silenced."}'
```

The API will return a JSON response indicating which spans (if any) were identified as propaganda and, for those spans, which of the 14 SemEval techniques (e.g. *Loaded Language*, *Appeal to Fear*, *Name Calling*, etc.) were detected.

If you have started the Flask server, you can also interact with the Chrome extension directly against a locally running instance once `deploy-flask-app` is active.

#### Running the extension locally

First you need to create a `config.js` from the example:

```javascript
const API_ENDPOINT = "http://localhost:5000/process";

const MAX_TEXT_LENGTH = 5000; // Limit text length to prevent crashes
```

Then you need to open Google Chrome and visit `chrome://extensions`.

Turn on Developer Mode using the toggle in the upper right corner.

Then click "Load Unpacked" and select the `extension` directory.

The Propaganda Scanner Extension should then show up as a panel on the screen.

Make sure the extension is enabled via the toggle in the lower right corner of the panel. The Icon for the Propaganda scanner should show up in the browsere frame, next to the address bar.

To use the extension, visit a page you want scanned and click on the extension and then "Scan Page." Wait for the extension to finish processing. A table of all techniques found should show up on a table in the extension and text flagged as propaganda on the page should be highlighted.

#### Running the Jupyter Notebooks

The `notebooks/` directory contains the exploratory data analysis (EDA) and experimentation notebooks. These are self-contained and can be run independently of the API or model pipeline. To start Jupyter:

```bash
poetry run jupyter notebook
# or, if you prefer JupyterLab:
poetry run jupyter lab
```

The notebooks follow the naming convention `<step>.<substep>-<author-initials>-<description>` and cover:

- **Data loading and inspection** of the SemEval-2020 corpus (raw article text and span/label files)
- **Exploratory Data Analysis (EDA):** class distributions across 14 propaganda techniques, span-length statistics, sentence-level labeling patterns
- **Model experiments:** training runs, evaluation on the development set, F1/precision/recall comparisons across SI and TC subtasks
- **Claim extraction experiments:** using the Claimify-inspired pipeline to identify core factual assertions

> **Note:** Some notebooks require the data to be present under `data/raw/`. See the [Data Access](#data-access) section below for how to obtain it.

## Data Access

This project uses two third-party datasets. **No data is bundled in this repository.** You must obtain the data directly from the original providers under their respective licenses.

### SemEval-2020 Task 11 — Fine-Grained Propaganda Detection

The primary training and evaluation data is the **PTC-SemEval20 corpus**, consisting of annotated news articles with span-level labels across 14 propaganda techniques.

- **Access:** The dataset is publicly available on Zenodo: [https://zenodo.org/records/3952415](https://zenodo.org/records/3952415)
- **License:** The data is released for research purposes by the task organizers (Qatar Computing Research Institute / HBKU). Please refer to the Zenodo record and the [task website](https://propaganda.qcri.org/semeval2020-task11) for the specific terms of use. Academic, non-commercial use is the intended scope.
- **Citation:**
  > Da San Martino, G., Barrón-Cedeño, A., Wachsmuth, H., Petrov, R., & Nakov, P. (2020). SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles. *Proceedings of the 14th International Workshop on Semantic Evaluation (SemEval 2020)*, Barcelona, Spain.

### AVeriTeC — Real-World Claim Verification

The claim-extraction component draws on the **AVeriTeC** dataset, a corpus of 4,568 real-world fact-checked claims annotated with question-answer evidence pairs and verdict justifications.

- **Access:** Available at [https://fever.ai/dataset/averitec.html](https://fever.ai/dataset/averitec.html)
- **License:** CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International). The data and baseline code **may not be used for commercial purposes**. Attribution is required.
- **Citation:**
  > Schlichtkrull, M. S., Guo, Z., & Vlachos, A. (2023). AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web. *NeurIPS 2023 Datasets and Benchmarks Track*.

This repository does not redistribute either dataset. All data use in this project respects the licenses stated above.

## Dependency Management

This repo ships **both** `poetry.lock` and `requirements.txt`. Here is what each is for and when to use which:

#### `poetry.lock` (recommended)

`poetry.lock` is a machine-generated file that records the **exact resolved version of every package and every transitive dependency** at the time dependencies were last resolved. When you run `poetry install`, Poetry reads this file and installs precisely those versions — byte-for-byte reproducible across any machine, operating system, and time.

**Use this** if you want a reproducible environment that exactly matches what the project authors tested with:

```bash
poetry install        # installs everything from the lockfile, no surprises
```

The `pyproject.toml` defines the *acceptable* version ranges (e.g. `torch = "^2.11.0"`); the lockfile records the *exact* version that was actually resolved (e.g. `torch 2.11.0`). This distinction matters because a range like `^2.11.0` allows future patch or minor releases, and without a lockfile a `pip install` done a year from now might pull a subtly different version.

You should commit `poetry.lock` to version control (it is already committed in this repo). You should **not** edit it by hand. If you add or change a dependency in `pyproject.toml`, regenerate it with `poetry lock`.

#### `requirements.txt` (fallback / interoperability)

`requirements.txt` is provided for environments where Poetry is not available — for example, if someone wants to use plain `pip`, or deploy to a platform that only accepts a requirements file. It pins specific versions, but unlike `poetry.lock` it does **not** capture the full dependency graph of transitive dependencies with hashes.

```bash
pip install -r requirements.txt   # use only if you cannot use poetry
```

> **Recommendation:** Always prefer `poetry install` for development and reproducibility. Use `requirements.txt` only if you are integrating with a tool that does not support Poetry.

## Project Organization

```
├── LICENSE            <- License
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         claim_aware_propaganda_scanner and configuration
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── create-model       <- Source code for use in this project
    │
    ├── claim-utils.py                     <- Lightweight claim extraction logic (Claimify-inspired)
    │
    ├── data-pull.py                       <- Scripts to download and prepare raw data
    │
    ├── gdown-download-models.py           <- Utility to fetch pre-trained weights via Google Drive
    │
    ├── helpers.py                         <- General utilities and model pipeline processing endpoints
    │
    ├── is-subjective.py                   <- Feature engineering: specialized subjectivity detection
    │
    ├── run-semeval-create-pipeline.py     <- Orchestration script for the end-to-end model pipeline
    │
    ├── semeval-clean.py                   <- Data preprocessing and text cleaning scripts
    │
    ├── semeval-create-si-model.py         <- Code for Span Identification (SI) task
    │
    ├── semeval-create-tc-model.py         <- Code for Technique Classification (TC) task
    │
    └── semeval-create-specialist-model.py <- Specialist model to help with SI on types that are frequently missed
```

--------

