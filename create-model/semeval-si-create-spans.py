
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

def main():
    print(f"Loading existing trained model from: {MODEL_DIR}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model_already_trained = True

if __name__ == "__main__":
    main()


