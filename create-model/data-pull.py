import os
import subprocess
import pandas as pd
import numpy as np
import json
from pathlib import Path

#Define paths
BASE_DIR = Path("..").resolve()
RAW_DIR = BASE_DIR / "data" / "raw"
if not RAW_DIR.exists():
    os.makedirs(RAW_DIR, exist_ok=True)
INTERIM_DIR = BASE_DIR / "data" / "interim"
if not INTERIM_DIR.exists():
    os.makedirs(INTERIM_DIR, exist_ok=True)

#Files of interest
TGZ_PATH = RAW_DIR / "semeval-dataset.tgz"
SEMEVAL_EXTRACT_DIR = RAW_DIR / "semeval2020_data"
INTERIM_SI_FILE = INTERIM_DIR / "semeval_task1_si_merged.csv"
INTERIM_TC_FILE = INTERIM_DIR / "semeval_task2_tc_merged.csv"
INTERIM_AV_FILE = INTERIM_DIR / "averitec_dev_flattened.csv"

#Check if data is already processed
if INTERIM_SI_FILE.exists() and INTERIM_TC_FILE.exists() and INTERIM_AV_FILE.exists():
    print("Interim files found. Loading directly from data/interim...")
    df_si = pd.read_csv(INTERIM_SI_FILE)
    df_tc = pd.read_csv(INTERIM_TC_FILE)
    averitec = pd.read_csv(INTERIM_AV_FILE)

else:
    print("Data not found in interim. Starting full download and extraction...")

    # 1. Clone AVeriTeC
    averitec_dir = RAW_DIR / "averitec"
    if not averitec_dir.exists():
        print("Cloning AVeriTeC...")
        subprocess.run(["git", "clone", "https://github.com/MichSchli/AVeriTeC.git", str(averitec_dir)])

    # 2. Download SemEval
    if not TGZ_PATH.exists():
        print("Downloading SemEval-2020 Task 11 dataset...")
        subprocess.run(["curl", "-L", "-o", str(TGZ_PATH), "https://zenodo.org/record/3952415/files/datasets-v2.tgz?download=1"])

    # 3. Clone HQP
    ###Not using for now but may later if we can figure out Twitter API
    #hqp_dir = RAW_DIR / "hqp"
    #if not hqp_dir.exists():
        #print("Cloning HQP Dataset Repo...")
        #subprocess.run(["git", "clone", "https://github.com/abdumaa/HiQualProp.git", str(hqp_dir)])

    #Extract SemEval data from TGZ
    if not SEMEVAL_EXTRACT_DIR.exists():
        print("Extracting SemEval tarball...")
        os.makedirs(SEMEVAL_EXTRACT_DIR, exist_ok=True)
        subprocess.run(["tar", "-xvzf", str(TGZ_PATH), "-C", str(SEMEVAL_EXTRACT_DIR)])

    #Connect article texts to labels in SemEval rows
    txt_paths = list(SEMEVAL_EXTRACT_DIR.rglob("*.txt"))

    #Create a dictionary mapping: article_id -> full_text
    article_text_map = {}
    for p in txt_paths:
        article_id = p.stem.replace("article", "") # Keeps just the number
        with open(p, 'r', encoding='utf-8') as f:
            article_text_map[article_id] = f.read()

    #Task 1: Span Identification
    si_paths = list(SEMEVAL_EXTRACT_DIR.rglob("*.task1-SI.labels"))
    df_si = pd.concat([
        pd.read_csv(p, sep="\t", header=None, names=["article_id", "start_char", "end_char"]).assign(source_file=p.name)
        for p in si_paths
    ], ignore_index=True)

    #Task 2: Technique Classification
    tc_paths = list(SEMEVAL_EXTRACT_DIR.rglob("*.task2-TC.labels"))
    df_tc = pd.concat([
        pd.read_csv(p, sep="\t", header=None, names=["article_id", "technique", "start_char", "end_char"]).assign(source_file=p.name)
        for p in tc_paths
    ], ignore_index=True)

    #Map the text to the dataframes using the article_id
    df_si['article_id'] = df_si['article_id'].astype(str)
    df_tc['article_id'] = df_tc['article_id'].astype(str)

    df_si['text_content'] = df_si['article_id'].map(article_text_map)
    df_tc['text_content'] = df_tc['article_id'].map(article_text_map)

    #Process AVeriTeC
    with open(RAW_DIR / "averitec/data/dev.json", 'r') as f:
        averitec = pd.json_normalize(json.load(f))

    #Save to interim
    df_si.to_csv(INTERIM_SI_FILE, index=False)
    df_tc.to_csv(INTERIM_TC_FILE, index=False)
    averitec.to_csv(INTERIM_AV_FILE, index=False)
    print(f"All files processed and saved to {INTERIM_DIR}")