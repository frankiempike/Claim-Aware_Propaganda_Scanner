import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_scanner"
SPECIALIST_DIR = BASE_DIR / "models" / "semeval_roberta_scanner_specialist"
TC_MODEL_DIR = BASE_DIR / "models" / "semeval_roberta_classifier"

CONFIDENCE_THRESHOLD = 0.9
TC_THRESHOLD = 0.33
WINDOW_SIZE = 200

_tokenizer = None
_model = None
_specialist_model = None
_tc_tokenizer = None
_tc_model = None
_tc_label_names = None


def _load_models():
    """
    Load SI models (base and specialist) and tokenizer on first call.
    Uses lazy loading with global caching to avoid reloading models.
    """
    global _tokenizer, _model, _specialist_model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
        _model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        _model.eval()
        _specialist_model = AutoModelForTokenClassification.from_pretrained(SPECIALIST_DIR)
        _specialist_model.eval()


def _load_tc_model():
    """
    Load TC model, tokenizer, and label names on first call.
    Uses lazy loading with global caching to avoid reloading models.
    """
    global _tc_tokenizer, _tc_model, _tc_label_names
    if _tc_tokenizer is None:
        _tc_tokenizer = AutoTokenizer.from_pretrained(TC_MODEL_DIR)
        _tc_model = AutoModelForSequenceClassification.from_pretrained(TC_MODEL_DIR)
        _tc_model.eval()
        # Extract label names from config
        _tc_label_names = [_tc_model.config.id2label[i] for i in range(_tc_model.config.num_labels)]


def _merge_overlapping_spans(spans):
    """
    Deduplicate and merge overlapping spans from sliding window predictions.
    
    Args:
        spans: List of dicts with 'start' and 'end' character positions
    
    Returns:
        Sorted list of unique (start, end) tuples
    """
    unique_spans = sorted({(s["start"], s["end"]) for s in spans})
    return unique_spans


def _extract_spans_from_predictions(predictions, offset_mapping, text):
    """
    Convert token-level model predictions to character-level text spans.
    Reconstructs spans from token offsets, handling special tokens and span merging.
    
    Args:
        predictions: Token-level class predictions (batch_size, seq_length)
        offset_mapping: Tokenizer offset mapping for character-to-token alignment
        text: Original text (used for validation, can be None)
    
    Returns:
        List of dicts with 'start' and 'end' character positions
    """
    spans = []
    offsets = offset_mapping.numpy()
    
    for pred_sequence, offset_sequence in zip(predictions, offsets):
        curr_span = None
        for pred, (start, end) in zip(pred_sequence, offset_sequence):
            if start == 0 and end == 0:
                if curr_span is not None:
                    spans.append({"start": curr_span[0], "end": curr_span[1]})
                    curr_span = None
                continue
            
            if pred == 1:
                if curr_span is None:
                    curr_span = [int(start), int(end)]
                else:
                    curr_span[1] = int(end)
            elif curr_span is not None:
                spans.append({"start": curr_span[0], "end": curr_span[1]})
                curr_span = None
        
        if curr_span is not None:
            spans.append({"start": curr_span[0], "end": curr_span[1]})
    
    return spans

def process_text_si(text):
    """
    Detect propaganda spans in text using soft cascade SI models.
    Uses base model with specialist model override for high-confidence predictions.
    
    Args:
        text: Article text to analyze
    
    Returns:
        List of dicts with detected propaganda spans:
            - 'start': Character offset of span start
            - 'end': Character offset of span end
            - 'text': The span text content
    """
    
    _load_models()

    if not _tokenizer or not _model or not _specialist_model:
        raise RuntimeError("Models not loaded properly.")

    # Sliding window tokenization — matches notebook tokenize_and_align_labels
    inputs = _tokenizer(
        text,
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    offset_mapping = inputs.pop("offset_mapping")
    inputs.pop("overflow_to_sample_mapping", None)

    with torch.no_grad():
        base_logits = _model(**inputs).logits.numpy()
        specialist_logits = _specialist_model(**inputs).logits.numpy()

    base_preds = np.argmax(base_logits, axis=2)

    # Pad specialist logits from 2 classes → 3 to match base model shape
    batch_size, seq_len, _ = specialist_logits.shape
    dummy_logits = np.full((batch_size, seq_len, 1), -100.0)
    padded_spec_logits = np.concatenate([specialist_logits, dummy_logits], axis=2)

    # Override base prediction with specialist where specialist is confident enough
    specialist_probs = F.softmax(torch.tensor(padded_spec_logits), dim=-1).numpy()
    propaganda_prob = specialist_probs[:, :, 1]
    mask = (base_preds == 0) & (propaganda_prob > CONFIDENCE_THRESHOLD)

    combined_logits = base_logits.copy()
    combined_logits[mask] = padded_spec_logits[mask]

    final_preds = np.argmax(combined_logits, axis=2)

    # Extract character-level propaganda spans from offset mapping
    pred_spans = _extract_spans_from_predictions(final_preds, offset_mapping, text)

    # Deduplicate spans produced by overlapping windows
    unique_spans = _merge_overlapping_spans(pred_spans)
    result = []
    for start, end in unique_spans:
        result.append({"start": start, "end": end, "text": text[start:end]})
    return result

def process_text_tc(text, spans):
    """
    Classify propaganda techniques for given spans in text.
    
    Args:
        text: The full article text
        spans: List of dicts with 'start' and 'end' character positions
    
    Returns:
        List of dicts with span info and predicted techniques
    """
    _load_tc_model()
    
    if not _tc_tokenizer or not _tc_model:
        raise RuntimeError("TC model not loaded properly.")
    
    results = []
    
    for span in spans:
        start_idx = span["start"]
        end_idx = span["end"]
        span_text = text[start_idx:end_idx]
        
        # Create contextualized input with window around span
        before_context = text[max(0, start_idx - WINDOW_SIZE):start_idx]
        after_context = text[end_idx:min(len(text), end_idx + WINDOW_SIZE)]
        
        # Mark the span with special tokens
        marked_text = f"{before_context} [SPAN] {span_text} [/SPAN] {after_context}"
        
        # Tokenize
        inputs = _tc_tokenizer(
            marked_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        # Get predictions
        with torch.no_grad():
            logits = _tc_model(**inputs).logits.numpy()[0]
        
        # Apply sigmoid and threshold
        probabilities = 1 / (1 + np.exp(-logits))
        predicted_labels = (probabilities >= TC_THRESHOLD).astype(int)
        technique_predictions = [
            {
                "technique": _tc_label_names[i],
                "probability": float(probabilities[i])
            }
            for i, pred in enumerate(predicted_labels) if pred == 1
        ]

        # If no techniques passed the threshold, fall back to the highest probability technique
        if len(technique_predictions) == 0:
            best_idx = int(np.argmax(probabilities))
            technique_predictions = [
                {
                    "technique": _tc_label_names[best_idx],
                    "probability": float(probabilities[best_idx])
                }
            ]

        results.append({
            "start": start_idx,
            "end": end_idx,
            "text": span_text,
            "techniques": technique_predictions,
            "all_probabilities": {_tc_label_names[i]: float(probabilities[i]) for i in range(len(_tc_label_names))}
        })

    return results