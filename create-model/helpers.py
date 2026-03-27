import json
import torch
from torch import nn
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, Trainer
from accelerate.state import AcceleratorState
from datasets import Dataset
import numpy as np
import gdown
import shutil
import zipfile
import os

def tokenize_and_align_labels(examples):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_inputs["overflow_to_sample_mapping"]
    offset_mapping = tokenized_inputs["offset_mapping"]
    labels = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        article_spans = examples["propaganda_offsets"][sample_idx]
        if isinstance(article_spans, str):
            article_spans = json.loads(article_spans)
        doc_labels = []
        for start, end in offsets:
            if start == end == 0:
                doc_labels.append(-100)
                continue
            is_prop = any(s <= start < e or s < end <= e for s, e in article_spans)
            doc_labels.append(1 if is_prop else 0)
        labels.append(doc_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics_tc(p):
    """Compute metrics for multi-label technique classification (2D logits)."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    logits, labels = p
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    predictions = (probs >= 0.5).astype(int)
    labels = np.array(labels).astype(int)
    return {
        "precision": precision_score(labels, predictions, average="micro", zero_division=0),
        "recall": recall_score(labels, predictions, average="micro", zero_division=0),
        "f1": f1_score(labels, predictions, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }

def compute_metrics(p):
    BASE_DIR = Path("..")
    DATA_DIR = BASE_DIR / "data" / "processed"
    df_si = pd.read_csv(DATA_DIR / "semeval_si_cleaned.csv")
    raw_dataset = Dataset.from_pandas(df_si)
    
    tokenized_datasets = raw_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=raw_dataset.column_names).train_test_split(test_size=0.2, seed=42)

    logits, labels = p
    predictions = np.argmax(logits, axis=2) # Default 0.5 threshold

    # Safety check: Get correct offsets for current batch
    if len(predictions) == len(tokenized_datasets["test"]):
        eval_offsets = tokenized_datasets["test"]["offset_mapping"]
    else:
        eval_offsets = tokenized_datasets["train"]["offset_mapping"]

    all_p, all_r, all_f1 = [], [], []

    for i in range(len(predictions)):
        pred_spans, gold_spans = [], []
        curr_p, curr_g = None, None

        for j, (pred, label) in enumerate(zip(predictions[i], labels[i])):
            if label == -100: continue
            start, end = eval_offsets[i][j]

            if pred == 1: # Predicted Propaganda
                if curr_p is None: curr_p = [start, end]
                else: curr_p[1] = end
            elif curr_p:
                pred_spans.append(tuple(curr_p)); curr_p = None

            if label == 1: # Gold Propaganda
                if curr_g is None: curr_g = [start, end]
                else: curr_g[1] = end
            elif curr_g:
                gold_spans.append(tuple(curr_g)); curr_g = None

        # Calculate scores for this sentence
        p_val, r_val, f1_val = get_si_metrics(pred_spans, gold_spans)
        all_p.append(p_val)
        all_r.append(r_val)
        all_f1.append(f1_val)

    return {
        "si_precision": np.mean(all_p),
        "si_recall": np.mean(all_r),
        "si_f1": np.mean(all_f1)
    }

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        #Prioritize Recall: Propaganda classes weighted 3x more than background (0)
        num_labels = self.model.config.num_labels

        # weights = torch.tensor([1.0] + [3.0] * (num_labels - 1), device=model.device)
        # loss_fct = nn.CrossEntropyLoss(weight=weights)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        pos_weights = torch.tensor([1.0] + [3.0] * (num_labels - 1), device=model.device)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        loss = loss_fct(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

def get_si_metrics(predicted_spans, gold_spans):
    """Official SemEval 2020 Task 11 SI Fuzzy Overlap Math"""
    if not predicted_spans and not gold_spans: return 1.0, 1.0, 1.0
    if not predicted_spans or not gold_spans: return 0.0, 0.0, 0.0

    # Precision calculation
    p_num = 0
    for s in predicted_spans:
        max_overlap = 0
        for t in gold_spans:
            intersect = max(0, min(s[1], t[1]) - max(s[0], t[0]))
            max_overlap = max(max_overlap, intersect / (s[1] - s[0]))
        p_num += max_overlap
    precision = p_num / len(predicted_spans)

    # Recall calculation
    r_num = 0
    for t in gold_spans:
        max_overlap = 0
        for s in predicted_spans:
            intersect = max(0, min(s[1], t[1]) - max(s[0], t[0]))
            max_overlap = max(max_overlap, intersect / (t[1] - t[0]))
        r_num += max_overlap
    recall = r_num / len(gold_spans)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def save_chunk(ids, attn, lbls, tokenized_datasets, missed_chunks, clean_chunks):
        sample_input_ids = tokenized_datasets["train"][0]["input_ids"]
        sample_mask = tokenized_datasets["train"][0]["attention_mask"]
        cls_id = sample_input_ids[0]
        sep_id = sample_input_ids[sum(sample_mask) - 1]

        """Helper to cleanly cap the sequence and sort it into the right bucket."""
        valid_content = [l for l in lbls if l != -100]
        if len(valid_content) == 0:
            return

        if ids[0] != cls_id:
            ids.insert(0, cls_id)
            attn.insert(0, 1)
            lbls.insert(0, -100)

        if ids[-1] != sep_id:
            ids.append(sep_id)
            attn.append(1)
            lbls.append(-100)

        has_missed = any(l > 0 for l in valid_content)
        if has_missed:
            missed_chunks["input_ids"].append(ids)
            missed_chunks["attention_mask"].append(attn)
            missed_chunks["labels"].append(lbls)
        else:
            clean_chunks["input_ids"].append(ids)
            clean_chunks["attention_mask"].append(attn)
            clean_chunks["labels"].append(lbls)

def setup_models_from_gdrive(file_id, target_path):
    target_path = Path(target_path).resolve()
    zip_temp = target_path.with_suffix(".zip")

    #Check if files already exist in the correct spot
    if (target_path / "model.safetensors").exists() or (target_path / "pytorch_model.bin").exists():
        print(f"Model weights detected locally at {target_path}")
        return True

    print(f"Model not found. Preparing {target_path}...")

    #Ensure the specific sub-folder exists
    target_path.mkdir(exist_ok=True, parents=True)

    url = f'https://drive.google.com/uc?id={file_id}'

    try:
        #1. Download the zip
        gdown.download(url, str(zip_temp), quiet=False)

        #2. Extract to a temporary location
        temp_extract = target_path / "temp_extraction"
        if temp_extract.exists(): shutil.rmtree(temp_extract)
        temp_extract.mkdir(parents=True)

        print("Unzipping and cleaning up structure...")
        with zipfile.ZipFile(zip_temp, 'r') as zip_ref:
            members = [m for m in zip_ref.namelist() if "__MACOSX" not in m]
            zip_ref.extractall(temp_extract, members=members)

        #3. Move files from temp_extract into target_path
        for root, dirs, files in os.walk(temp_extract):
            for file in files:
                src_file = Path(root) / file
                dest_file = target_path / file
                shutil.move(str(src_file), str(dest_file))

        #4. Final Cleanup
        shutil.rmtree(temp_extract)
        if zip_temp.exists():
            os.remove(zip_temp)

        print(f"Model files are now in: {target_path}")
        return True

    except Exception as e:
        print(f"Error during setup: {e}")
        if zip_temp.exists(): os.remove(zip_temp)
        return False
    
def preprocess_function(examples, tokenizer, window_size=200):
    contextualized_inputs = []

    for i in range(len(examples["span_text"])):
        span = str(examples["span_text"][i])
        full_text = str(examples["text_content"][i])

        #Get the exact character positions
        start_idx = int(examples["start"][i])
        end_idx = int(examples["end"][i])

        #Slice the string to get the text immediately before and after the span
        before_context = full_text[max(0, start_idx - window_size) : start_idx]

        #min(len(), ...) prevents errors if the span is at the very end
        after_context = full_text[end_idx : min(len(full_text), end_idx + window_size)]

        #Reconstruct the string with our markers
        marked_text = f"{before_context} [SPAN] {span} [/SPAN] {after_context}"
        contextualized_inputs.append(marked_text)

    #Tokenize the newly windowed strings
    return tokenizer(
        contextualized_inputs,
        truncation=True,
        padding="max_length",
        max_length=256
    )

def get_raw_datasets():
    BASE_DIR = Path("..")
    DATA_DIR = BASE_DIR / "data" / "processed"
    df_si = pd.read_csv(DATA_DIR / "semeval_si_cleaned.csv")
    raw_dataset = Dataset.from_pandas(df_si)
    return raw_dataset

def get_tokenized_datasets():
    raw_dataset = get_raw_datasets()
    tokenized_samples = []
    for sample in raw_dataset:
        tokenized_sample = tokenize_and_align_labels({
            "text": [sample["text"]], 
            "propaganda_offsets": [sample["propaganda_offsets"]]
        })
        tokenized_samples.append(tokenized_sample)
    tokenized_dataset = Dataset.from_dict({key: [s[key] for s in tokenized_samples] for key in tokenized_samples[0].keys()})
    tokenized_datasets = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    return tokenized_datasets
    