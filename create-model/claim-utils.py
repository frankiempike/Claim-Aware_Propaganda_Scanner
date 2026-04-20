import nltk
from transformers import pipeline
import spacy
from fastcoref.modeling import FCoref, FCorefModel
import torch
import re

nltk.download('punkt')
nltk.download('punkt_tab')

if not hasattr(FCorefModel, "all_tied_weights_keys"):
    FCorefModel.all_tied_weights_keys = {}

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
coref_model = FCoref(device=device)

claim_classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    device=-1 # Set to 0 if using a GPU
)

nlp = spacy.load("en_core_web_sm")

def resolve_coreferences(text):
    """
    Replaces pronouns and implicit references in the text with their explicit entities.
    """
    #Predict coreference clusters
    preds = coref_model.predict(texts=[text])

    #Get clusters as character start/end indices
    clusters = preds[0].get_clusters(as_strings=False)

    replacements = []
    for cluster in clusters:
        #The first mention in a cluster is usually the explicit entity (the antecedent)
        primary_start, primary_end = cluster[0]
        primary_text = text[primary_start:primary_end]

        #Replace all subsequent mentions (usually pronouns) with the primary text
        for mention_start, mention_end in cluster[1:]:
            replacements.append((mention_start, mention_end, primary_text))

    #Sort replacements in reverse order of their start index
    replacements.sort(key=lambda x: x[0], reverse=True)

    #Apply replacements
    resolved_text = text
    for start, end, rep_text in replacements:
        resolved_text = resolved_text[:start] + rep_text + resolved_text[end:]

    return resolved_text

def format_claim(text):
    """Helper function to clean up capitalization and punctuation."""
    #1. Strip trailing whitespace and dangling commas
    text = re.sub(r'[,\s]+$', '', text.strip())

    #2. Fix spaces before periods or commas (e.g., "year ." -> "year.")
    text = re.sub(r'\s+([.,])', r'\1', text)

    #3. Ensure it ends with a period
    if not text.endswith('.'):
        text += '.'

    #4. Capitalize the first letter (without changing the rest of the string)
    if len(text) > 0:
        text = text[0].upper() + text[1:]

    return text

def decompose_sentence(sentence):
    doc = nlp(sentence)

    #1. Find the ROOT verb of the sentence
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if not root:
        return [format_claim(sentence)]

    #Check if there are any conjunct verbs attached to the root (e.g., joined by "and")
    conjunct_verbs = [t for t in root.children if t.dep_ == "conj" and t.pos_ == "VERB"]

    #If there are no compound predicates, return the sentence EXACTLY as is
    if not conjunct_verbs:
        return [format_claim(sentence)]

    claims = []

    # 3.Identify the main subject AND any shared auxiliary verbs (like "will", "has", "is")
    subjects = [t for t in root.children if "subj" in t.dep_]
    if not subjects:
        return [format_claim(sentence)]

    subject_phrase = "".join([t.text_with_ws for t in subjects[0].subtree]).strip()
    auxs = [t for t in root.children if t.dep_ == "aux"]
    aux_phrase = "".join([t.text_with_ws for t in auxs]).strip()

    #This is the prefix we will attach to the second verb (e.g., "The infrastructure bill will")
    prefix = f"{subject_phrase} {aux_phrase}".strip()

    #4. Create Claim 1: The original sentence minus the second verb and the "and"
    ignore_tokens = set()
    for conj in conjunct_verbs:
        ignore_tokens.update([t.i for t in conj.subtree]) #Ignore the second verb's phrase
    ignore_tokens.update([t.i for t in root.children if t.dep_ == "cc"]) #Ignore the conjunction

    #Reassemble using the exact original token order
    claim1_tokens = [t for t in doc if t.i not in ignore_tokens]
    claim1 = "".join([t.text_with_ws for t in claim1_tokens]).strip()
    claims.append(format_claim(claim1))

    #5. Create Claim 2: Apply the prefix to the conjunct verb phrase
    for conj in conjunct_verbs:
        #Sort tokens to maintain correct word order
        conj_tokens = sorted(list(conj.subtree), key=lambda x: x.i)
        conj_phrase = "".join([t.text_with_ws for t in conj_tokens]).strip()

        atomic_claim = f"{prefix} {conj_phrase}"
        claims.append(format_claim(atomic_claim))

    return claims

def lightweight_claimify(text, threshold=0.6):
    """
    The complete, keyless 4-stage claim extraction pipeline.
    """
    #Stage 1: Disambiguation (Coreference Resolution)—we're doing first instead of third because it works best with our lightweight, API-free approach
    disambiguated_text = resolve_coreferences(text)

    #Stage 2: Sentence Splitting
    complex_sentences = nltk.sent_tokenize(disambiguated_text)
    atomic_sentences = []

    #Stage 3: Decomposition
    for sent in complex_sentences:
        #Only try to decompose longer sentences with conjunctions
        if " and " in sent.lower() or " but " in sent.lower() or "," in sent:
            decomposed = decompose_sentence(sent)
            atomic_sentences.extend(decomposed)
        else:
            atomic_sentences.append(sent)

    extracted_claims = []

    #Stage 4: Selection (Filtering opinions vs. facts)
    for claim in atomic_sentences:
        if len(claim.split()) < 4:
            continue

        result = claim_classifier(
            claim,
            candidate_labels=["factual claim", "personal opinion"],
            multi_label=False
        )

        #Keep only the confident factual claims
        if result['labels'][0] == "factual claim" and result['scores'][0] >= threshold:
            extracted_claims.append(claim)

    #Clean up duplicates that can happen during decomposition
    return list(set(extracted_claims))