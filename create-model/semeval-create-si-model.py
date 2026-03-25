import os
import json
import torch
from torch import nn
import pandas as pd
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, EvalPrediction, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback)
from accelerate.state import AcceleratorState
from datasets import Dataset
import numpy as np
from helpers import WeightedTrainer, compute_metrics, get_tokenized_datasets, compute_metrics
AcceleratorState._reset_state()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Set up paths
BASE_DIR = Path("..")
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_scanner"
SPECIALIST_DIR = BASE_DIR / "models" / "semeval_roberta_scanner_specialist"

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

#Initialize the model
model_checkpoint = "roberta-base"
raw_dataset = Dataset.from_pandas(df_si)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=3)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2.5e-05,
    per_device_train_batch_size=8,
    num_train_epochs=13,
    weight_decay=0.15,
    logging_steps=5,
    metric_for_best_model="si_f1",
    greater_is_better=True,
    dataloader_pin_memory=False,
    disable_tqdm=False,
    report_to="none",
    load_best_model_at_end=True
)

tokenized_datasets = get_tokenized_datasets()

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

model.to(device)

trainer.train()
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(os.fspath(MODEL_DIR))
print(f"Model trained and saved to {MODEL_DIR}")
