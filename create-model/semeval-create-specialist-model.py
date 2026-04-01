
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0"

import torch
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, EvalPrediction, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback)
from datasets import Dataset
from accelerate.state import AcceleratorState
import random
from helpers import WeightedTrainer, compute_metrics, get_tokenized_datasets, save_chunk

BASE_DIR = Path("..")
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_scanner"
SPECIALIST_DIR = BASE_DIR / "models" / "semeval_roberta_scanner_specialist"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)

print(f"Loaded model from {MODEL_DIR} and moved to device: {device}")

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2.5e-05,
    per_device_train_batch_size=4,
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

print(f"Training arguments set with output directory: {training_args.output_dir}")

tokenized_datasets = get_tokenized_datasets()
model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting training of the specialist model...")

#Get predictions on train set
train_output = trainer.predict(tokenized_datasets["train"])
train_preds = np.argmax(train_output.predictions, axis=2)
train_labels = train_output.label_ids
del train_output  # free large predictions array

print("Finished training the main model. Now extracting missed propaganda chunks for specialist model...")

#We need RoBERTa's specific CLS and SEP token IDs to properly cap our new chunks
sample_input_ids = tokenized_datasets["train"][0]["input_ids"]
sample_mask = tokenized_datasets["train"][0]["attention_mask"]
cls_id = sample_input_ids[0]
sep_id = sample_input_ids[sum(sample_mask) - 1]

missed_chunks = {"input_ids": [], "attention_mask": [], "labels": []}
clean_chunks = {"input_ids": [], "attention_mask": [], "labels": []}

print("Iterating through training examples to extract chunks...")

print(f"Total training examples: {len(tokenized_datasets['train'])}")

for i in range(len(tokenized_datasets["train"])):
    print(f"Processing article {i+1}/{len(tokenized_datasets['train'])} for chunk extraction...", end="\r")

    input_ids = tokenized_datasets["train"][i]["input_ids"]
    attn_mask = tokenized_datasets["train"][i]["attention_mask"]
    labels = train_labels[i]
    preds = train_preds[i]

    cur_ids, cur_attn, cur_lbls = [], [], []
    in_found_span = False

    for input_id in range(len(input_ids)):
        print(f"Processing token {input_id+1}/{len(input_ids)} for article {i+1}...", end="\r")
        if attn_mask[input_id] == 0:
            continue

        cur_label = labels[input_id]
        cur_pred = preds[input_id]

        if cur_label != -100:
            in_found_span = (cur_label > 0) and (cur_pred > 0)

        if in_found_span:
            if len(cur_ids) > 0:
                save_chunk(cur_ids, cur_attn, cur_lbls, tokenized_datasets, missed_chunks, clean_chunks)
                cur_ids, cur_attn, cur_lbls = [], [], []
        else:
            cur_ids.append(input_ids[input_id])
            cur_attn.append(attn_mask[input_id])
            cur_lbls.append(cur_label)

    if len(cur_ids) > 0:
        print(f"Saving final chunk for article {i+1}...", end="\r")
        save_chunk(cur_ids, cur_attn, cur_lbls, tokenized_datasets, missed_chunks, clean_chunks)

print(f"-> Extracted {len(missed_chunks['input_ids'])} chunks with missed propaganda.")
print(f"-> Extracted {len(clean_chunks['input_ids'])} chunks with strictly NO propaganda.")

missed_list = [{"input_ids": i, "attention_mask": a, "labels": l}
                for i, a, l in zip(missed_chunks["input_ids"], missed_chunks["attention_mask"], missed_chunks["labels"])]
clean_list = [{"input_ids": i, "attention_mask": a, "labels": l}
                for i, a, l in zip(clean_chunks["input_ids"], clean_chunks["attention_mask"], clean_chunks["labels"])]

print(f"-> Created {len(missed_list)} missed propaganda examples and {len(clean_list)} clean examples for specialist training.")

sample_size = min(len(missed_list) * 2, len(clean_list))
random.seed(42)
sampled_clean = random.sample(clean_list, sample_size)

final_list = missed_list + sampled_clean
random.shuffle(final_list)

final_dict = {
    "input_ids": [ex["input_ids"] for ex in final_list],
    "attention_mask": [ex["attention_mask"] for ex in final_list],
    "labels": [ex["labels"] for ex in final_list]
}

residual_train_dataset = Dataset.from_dict(final_dict)
print(f"Created 'residual_train_dataset' with {len(residual_train_dataset)} continuous sequences ready for the specialist model.")

# Free the first model, trainer, and intermediate data from MPS to avoid OOM
del trainer
del model
del train_preds
del train_labels
del missed_chunks
del clean_chunks
del missed_list
del clean_list
del final_list
del final_dict

# Keep only the test split needed for eval, then free the rest
specialist_eval_dataset = tokenized_datasets["test"]
del tokenized_datasets

AcceleratorState._reset_state(True)
import gc; gc.collect()
torch.mps.empty_cache()

specialist_model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=2)
specialist_model.to(device)
print(f"Loaded specialist model and moved to device: {device}")

specialist_args = TrainingArguments(
    output_dir=SPECIALIST_DIR,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2.5e-05,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
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

specialist_trainer = WeightedTrainer(
    model=specialist_model,
    args=specialist_args,
    train_dataset=residual_train_dataset,
    eval_dataset=specialist_eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

specialist_trainer.train()
specialist_trainer.save_model(SPECIALIST_DIR)
tokenizer.save_pretrained(SPECIALIST_DIR)

print(f"Specialist model trained and saved to {SPECIALIST_DIR}")