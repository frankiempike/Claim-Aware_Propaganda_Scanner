# Claim-Aware_Propaganda_Scanner

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Our project detects likely propaganda in news articles and social media posts

## Constructing models

Before you are able to run the api or extension, you have create models

To install libraries, run

```
poetry install
```

You can run the following command to build models from scratch, locally

```
poetry run create-pipeline
```

OR

You can run the following to download the precreated models from google drive (contact Frankie Pike for access)

```
poetry run create-gdown
```

## Starting the API

Start the flask api with the poetry command

```
poetry run deploy-flask-app
```

## Setting up and running the extension

In the extension folder, create a new config.js using `config.js.example` as a guide

Then navigate to `chrome://extensions` and click 'Load unpacked'

Select the `extension` directory.

The extension should now be available on your browser.

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
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
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
│
├── setup.cfg          <- Configuration file for flake8
│
└── claim_aware_propaganda_scanner   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes claim_aware_propaganda_scanner a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

