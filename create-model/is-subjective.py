#This file contains the is_subjective(sentence) function which takes a sentence
#as an input and returns either True or False based on whether the sentence
#was subjective or objective. Questions are also considered subjective
#because they are not objective claims

from transformers import pipeline
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()

claim_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    device=-1
)

def is_subjective(sentence):
    """Input: sentence as a string
       Output: true or false"""

    result = claim_classifier(
            sentence,
            candidate_labels=["factual claim", "personal opinion"],
            multi_label=False
        )

    if result['labels'][0] == "factual claim":
        return False
    else:
        return True