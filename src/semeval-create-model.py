import os
import json
from torch import nn, torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support
from datasets import Dataset
import evaluate
import tqdm
from tqdm.auto import tqdm as tqdm_auto
tqdm_auto.pandas()
from accelerate.state import AcceleratorState
from transformers.utils.notebook import NotebookProgressCallback
AcceleratorState._reset_state()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Set up paths
BASE_DIR = Path("..")
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_scanner"

def tokenize_and_align_labels(examples):
    # 1. Tokenize with a sliding window
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 2. Correctly map chunks to original articles
    # 'overflow_to_sample_mapping' is the correct key for Fast Tokenizers
    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    labels = []

    # Iterate through every chunk (i) and find which original article (sample_idx) it belongs to
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        article_spans = examples["propaganda_offsets"][sample_idx]

        doc_labels = []
        for start, end in offsets:
            # -100 for special tokens like [CLS], [SEP], and [PAD]
            if start == end == 0:
                doc_labels.append(-100)
                continue

            # Check if this specific token character-range overlaps with any propaganda span
            is_prop = any(s <= start < e or s < end <= e for s, e in article_spans)

            # Label as 1 (Propaganda) or 0 (Normal)
            doc_labels.append(1 if is_prop else 0)

        labels.append(doc_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

#Check for GPU support
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) for acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU. Training might be slow.")

#Load article-level span identification data
df_si = pd.read_csv(DATA_DIR / "semeval_si_cleaned.csv")

df_si['propaganda_offsets'] = df_si['propaganda_offsets'].apply(json.loads)
print(f"Loaded {len(df_si)} articles.")
df_si.head()

#Initialize the model
model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

#Initialize model - Load from local if exists, else from checkpoint
if (MODEL_DIR / "config.json").exists() and (MODEL_DIR / "pytorch_model.bin").exists() and (MODEL_DIR / "model.safetensors").exists():
    print(f"Loading existing trained model from: {MODEL_DIR}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model_already_trained = True
else:
    print(f"No existing model found. Initializing from: {model_checkpoint}")
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=3)
    model_already_trained = False

model.to(device)
