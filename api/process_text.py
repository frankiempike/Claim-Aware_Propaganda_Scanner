import torch
from pathlib import Path
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, EvalPrediction, Trainer, DataCollatorForTokenClassification, EarlyStoppingCallback)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_scanner"
SPECIALIST_DIR = BASE_DIR / "models" / "semeval_roberta_scanner_specialist"


def process_text_si(text):
    print(f"Loading existing trained model from: {MODEL_DIR}")
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    specialist_model = AutoModelForTokenClassification.from_pretrained(SPECIALIST_DIR)
    
    model_checkpoint = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    #Get predictions on passed text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=2)

        specialist_outputs = specialist_model(**inputs)
        specialist_preds = torch.argmax(specialist_outputs.logits, dim=2)

        print(f"Predictions on input text: {preds}")
        print(f"Specialist predictions on input text: {specialist_preds}")

        return preds, specialist_preds
