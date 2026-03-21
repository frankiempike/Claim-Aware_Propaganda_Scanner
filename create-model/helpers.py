import json
import torch
from torch import nn
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, Trainer
from accelerate.state import AcceleratorState
from datasets import Dataset
import numpy as np

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
        weights = torch.tensor([1.0] + [3.0] * (num_labels - 1), device=model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

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


def get_raw_datasets():
    BASE_DIR = Path("..")
    DATA_DIR = BASE_DIR / "data" / "processed"
    df_si = pd.read_csv(DATA_DIR / "semeval_si_cleaned.csv")
    raw_dataset = Dataset.from_pandas(df_si)
    return raw_dataset

def get_tokenized_datasets():
    raw_dataset = get_raw_datasets()
    tokenized_datasets = raw_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=raw_dataset.column_names).train_test_split(test_size=0.2, seed=42)
    return tokenized_datasets