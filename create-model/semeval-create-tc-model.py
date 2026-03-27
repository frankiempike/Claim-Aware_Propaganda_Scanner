import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from functools import partial
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from safetensors.torch import load_file
from helpers import compute_metrics_tc, preprocess_function, setup_models_from_gdrive

# Define global paths
BASE_DIR = Path("..")
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
TC_MODEL_PATH = MODEL_DIR / "semeval_roberta_classifier"

GOOGLE_DRIVE_TC_CONTEXT_ZIP_ID = os.getenv("GOOGLE_DRIVE_TC_CONTEXT_ZIP_ID")


WINDOW_SIZE = 200

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Tokenizer with special span markers
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
special_tokens_dict = {'additional_special_tokens': ['[SPAN]', '[/SPAN]']}
tokenizer.add_special_tokens(special_tokens_dict)
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print(f"Added {num_added_toks} special tokens to the tokenizer.")

# Download model from Google Drive if not present locally
model_already_trained = setup_models_from_gdrive(GOOGLE_DRIVE_TC_CONTEXT_ZIP_ID, TC_MODEL_PATH)

# Load technique classification data
df_tc = pd.read_csv(DATA_DIR / "semeval_tc_cleaned.csv")

print(f"Initial dataset loaded with {len(df_tc)} spans and {df_tc['article_id'].nunique()} unique articles.")
print(df_tc.head())

# Split by article to avoid data leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df_tc, groups=df_tc['article_id']))
train_df = df_tc.iloc[train_idx].reset_index(drop=True)
test_df = df_tc.iloc[test_idx].reset_index(drop=True)

print(f"Total spans: {len(df_tc)}, total articles: {df_tc['article_id'].nunique()}")
print(f"Train spans: {len(train_df)} ({train_df['article_id'].nunique()} articles)")
print(f"Test spans:  {len(test_df)} ({test_df['article_id'].nunique()} articles)")

overlap = set(train_df['article_id']).intersection(set(test_df['article_id']))
print(f"Number of overlapping articles: {len(overlap)}")

# Identify label columns (exclude metadata and linguistic feature columns)
feature_cols = ['sentiment', 'punct_count', 'lexical_diversity']
non_label_cols = ['article_id', 'text_content', 'span_text', 'start', 'end',
                  'technique', 'technique_list', 'source_file'] + feature_cols
label_cols = [c for c in train_df.columns if c not in non_label_cols]
print(f"Label columns: {label_cols}")

missing = [c for c in label_cols if c not in train_df.columns]
if missing:
    print(f"Warning: Missing columns in dataframe: {missing}")
else:
    print(f"Label columns correctly set. Number of techniques: {len(label_cols)}")

# Cast labels to float lists for BCEWithLogitsLoss
train_df['labels'] = train_df[label_cols].astype(float).values.tolist()
test_df['labels'] = test_df[label_cols].astype(float).values.tolist()

# Tokenize with contextual windowing around each span
preprocess = partial(preprocess_function, tokenizer=tokenizer, window_size=WINDOW_SIZE)
tokenized_train = Dataset.from_pandas(train_df).map(preprocess, batched=True)
tokenized_test = Dataset.from_pandas(test_df).map(preprocess, batched=True)

# Model config with label mappings
config = AutoConfig.from_pretrained(
    "roberta-base",
    num_labels=len(label_cols),
    id2label={i: label for i, label in enumerate(label_cols)},
    label2id={label: i for i, label in enumerate(label_cols)},
    problem_type="multi_label_classification",
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
)

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    config=config
).to(device)
model.resize_token_embeddings(len(tokenizer))

# Load pre-trained weights if available
if model_already_trained:
    safe_path = TC_MODEL_PATH / "model.safetensors"
    bin_path = TC_MODEL_PATH / "pytorch_model.bin"
    state_dict = None

    if safe_path.exists():
        state_dict = load_file(safe_path, device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location=torch.device('cpu'))

    if state_dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(".beta", ".bias").replace(".gamma", ".weight")
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=True)
        print("Weights successfully renamed and injected into the model.")
    else:
        print("Error: No weights found. Check your paths!")


# Custom trainer with standard (unweighted) BCE loss
class TCTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=TC_MODEL_PATH,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2.47e-05,
    per_device_train_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.2,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    load_best_model_at_end=True
)

trainer = TCTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics_tc,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

# Train only if no pre-trained model was loaded
if not model_already_trained:
    print("Starting training process...")
    trainer.train()
    trainer.save_model(TC_MODEL_PATH)
    tokenizer.save_pretrained(os.fspath(TC_MODEL_PATH))
    print(f"Model trained and saved to {TC_MODEL_PATH}")
else:
    print("Model loaded from disk. Skipping training.")

# Evaluate aggregate performance
test_results = trainer.evaluate(eval_dataset=tokenized_test)
print("\n" + "="*30)
print("FINAL MODEL PERFORMANCE")
print(f"Recall:    {test_results['eval_recall']:.4f}")
print(f"Precision: {test_results['eval_precision']:.4f}")
print(f"F1 Score:  {test_results['eval_f1']:.4f}")
print(f"Macro F1 Score:  {test_results['eval_macro_f1']:.4f}")
print("="*30)

# Per-technique evaluation
test_predictions = trainer.predict(tokenized_test)
logits = torch.tensor(test_predictions.predictions)
true_labels = test_predictions.label_ids
probabilities = torch.sigmoid(logits).numpy()

THRESHOLD = 0.5
binary_predictions = (probabilities >= THRESHOLD).astype(int)

print("\n--- Per-Technique Performance ---")
print(classification_report(
    true_labels,
    binary_predictions,
    target_names=label_cols,
    zero_division=0
))
