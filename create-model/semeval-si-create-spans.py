
import os
import json
import torch
from torch import nn
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, EvalPrediction, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback)
from datasets import Dataset
from accelerate.state import AcceleratorState
import zipfile
import shutil
import random
import gdown
from transformers.utils.notebook import NotebookProgressCallback


BASE_DIR = Path("..")
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_scanner"
SPECIALIST_DIR = BASE_DIR / "models" / "semeval_roberta_scanner_specialist"


def process_text(text):
    print(f"Loading existing trained model from: {MODEL_DIR}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    specialist_model = AutoModelForTokenClassification.from_pretrained(SPECIALIST_DIR)
    
    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    #Get predictions on passed text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    
    

