import re
import numpy as np
from scipy.sparse import csr_matrix

#Define handcrafted linguistic features
#Each set targets words strongly associated with a specific propaganda signal

_POSITIVE_WORDS = frozenset([
    'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
    'brilliant', 'heroic', 'noble', 'brave', 'righteous', 'glorious',
    'proud', 'victory', 'success', 'freedom', 'liberty', 'justice',
    'beautiful', 'perfect', 'best', 'finest', 'superior', 'outstanding',
    'magnificent', 'exceptional', 'blessed', 'sacred', 'holy', 'pure',
    'strong', 'powerful', 'winning', 'triumph', 'love', 'hope', 'unity',
])

_NEGATIVE_WORDS = frozenset([
    'bad', 'evil', 'terrible', 'horrible', 'disgusting', 'corrupt', 'vile',
    'wicked', 'traitor', 'criminal', 'thug', 'monster', 'liar', 'crook',
    'dangerous', 'threat', 'destroy', 'destruction', 'collapse', 'fail',
    'failure', 'wrong', 'false', 'lie', 'fraud', 'coward', 'puppet',
    'oppressive', 'tyrannical', 'shame', 'shameful', 'outrageous', 'hate',
    'enemy', 'enemies', 'terror', 'catastrophe', 'disaster', 'worst',
    'stupid', 'idiot', 'moron', 'loser', 'radical', 'extremist',
])

_INTENSIFIERS = frozenset([
    'very', 'extremely', 'absolutely', 'incredibly', 'utterly', 'totally',
    'completely', 'entirely', 'truly', 'deeply', 'highly', 'strongly',
    'tremendously', 'enormously', 'terribly', 'awfully', 'dreadfully',
])

_FEAR_WORDS = frozenset([
    'danger', 'dangerous', 'threat', 'threatening', 'risk', 'crisis',
    'catastrophe', 'catastrophic', 'disaster', 'disastrous', 'destroy',
    'destruction', 'collapse', 'invasion', 'attack', 'terror', 'terrorism',
    'fear', 'afraid', 'panic', 'alarm', 'horror', 'dread', 'nightmare',
    'enemy', 'enemies', 'menace', 'peril', 'urgent', 'emergency',
    'ebola', 'plague', 'outbreak'
])

_PATRIOTIC_WORDS = frozenset([
    'nation', 'national', 'country', 'homeland', 'freedom', 'liberty',
    'patriot', 'patriotic', 'patriotism', 'american', 'america', 'citizens',
    'people', 'great', 'proud', 'pride', 'tradition', 'values', 'heritage',
    'sovereignty', 'defend', 'protect', 'glory',
])

_AUTHORITY_WORDS = frozenset([
    'expert', 'experts', 'scientist', 'scientists', 'study', 'studies',
    'research', 'researchers', 'professor', 'doctor', 'dr', 'phd',
    'evidence', 'proven', 'facts', 'data', 'according', 'report', 'reports',
    'official', 'government', 'university', 'institute', 'organization',
])


_DOUBT_WORDS = frozenset([
    'really', 'truly', 'actually', 'allegedly', 'supposedly', 'claimed',
    'question', 'doubt', 'suspicious', 'wonder', 'whether', 'failed',
    'failing', 'lied', 'lies', 'lie', 'misleading', 'wrong', 'incorrect',
    'false', 'cover', 'hide', 'hoax', 'conspiracy', 'fraud', 'suspicious'
])

_LOADED_WORDS = frozenset([
    'radical', 'extremist', 'traitor', 'corrupt', 'evil', 'wicked', 'vile',
    'disgusting', 'outrageous', 'shameful', 'coward', 'monster',
    'oppressive', 'tyrannical', 'fascist', 'socialist', 'communist',
    'heroic', 'noble', 'brave', 'righteous', 'brilliant', 'glorious',
    'sacred', 'holy', 'pure', 'blessed', 'cursed', 'infidel', 'regime',
    'puppet', 'globalist', 'elite', 'treacherous', 'hardworking',
])

_NAME_CALL_WORDS = frozenset([
    'idiot', 'moron', 'fool', 'loser', 'liar', 'crook', 'thug', 'criminal',
    'terrorist', 'coward', 'hypocrite', 'puppet', 'stooge', 'clown',
    'elitist', 'snowflake', 'racist', 'bigot', 'extremist', 'deplorable',
])

_COMMON_PHRASES = [
    #Loaded / fear phrases
    'wake up', 'open your eyes', 'the truth is', 'they want you to',
    'mainstream media', 'fake news', 'the real agenda', 'hidden agenda',
    'working class', 'ordinary people', 'our way of life',
    #Flag-waving phrases
    'our nation', 'our country', 'our people', 'our values', 'our freedom',
    'stand up for', 'fight for', 'true patriot', 'defend our',
    #Authority / doubt phrases
    'according to', 'studies show', 'experts say', 'it has been proven',
    'there is no evidence', 'no one talks about', 'nobody mentions',
    #Black-and-white / slogans
    'either you', 'with us or', 'you are either', 'there is no other',
    'only choice', 'the only way',
]

_BANDWAGON_HITLERUM_WORDS = frozenset([
    'everyone', 'everybody', 'millions', 'nobody', 'join', 'consensus',
    'popular', 'majority', 'hitler', 'nazi', 'fascist', 'reminiscent',
    'history', 'repeating', 'concentration', 'genocide', 'dictator',
    'gestapo', 'third', 'reich', 'totalitarian', 'propaganda',
])

_LOGICAL_FALLACY_WORDS = frozenset([
    'whatabout', 'what about', 'instead', 'ignore', 'anyway', 'regardless', 'distraction',
    'argument', 'point', 'actually', 'strawman', 'focus', 'deflection',
    'excuse', 'side', 'issue', 'topic', 'irrelevant', 'meanwhile',
    'incidentally', 'divert', 'aside', 'besides',
])

def compute_handcrafted_features(texts):
    """Compute dense handcrafted linguistic features for a list of texts.

    All features are computed with stdlib + numpy — no extra dependencies.
    Every feature is scaled to [0, 1] to match TF-IDF sublinear_tf range,
    preventing raw counts from dominating the regularization signal.

    Features (19 total):
        0   polarity           (pos_hits - neg_hits) / n_words  in [-1, 1]
        1   subjectivity       (pos_hits + neg_hits) / n_words  in [ 0, 1]
        2   has_exclamation    binary: text contains '!'
        3   exclamation_rate   count('!') / n_words             in [ 0, 1]
        4   has_question       binary: text contains '?'
        5   caps_word_ratio    ALL-CAPS tokens (len≥2) / n_words in [ 0, 1]
        6   intensifier_rate   intensifier hits / n_words
        7   fear_rate          fear/threat word hits / n_words
        8   patriotic_rate     patriotic word hits / n_words
        9   authority_rate     authority word hits / n_words
        10  doubt_rate         doubt/discredit word hits / n_words
        11  loaded_rate        emotionally loaded word hits / n_words
        12  name_call_rate     name-calling word hits / n_words
        13  first_person_rate  we/our/us hits / n_words
        14  second_person_rate you/your hits / n_words
        15  phrase_rate        matched phrases / total phrases (fraction hit)
        16  length_norm        log1p(n_words) / log1p(300) — approx [0, 1]
        17  bandwagon_hitler_rate   bandwagon/ad-hitlerum hits / n_words
        18  fallacy_red_herring_rate whataboutism/red-herring hits / n_words

    Returns scipy.sparse.csr_matrix of shape (len(texts), 17).
    """
    n = len(texts)
    feat = np.zeros((n, 19), dtype=np.float32)
    n_phrases = max(len(_COMMON_PHRASES), 1)

    for i, text in enumerate(texts):
        tokens = re.findall(r"[A-Za-z']+", text)
        words_lower = [t.lower() for t in tokens]
        n_words = max(len(tokens), 1)
        text_lower = text.lower()

        # 0-1: lexicon-based sentiment (already ratios)
        pos_hits = sum(1 for w in words_lower if w in _POSITIVE_WORDS)
        neg_hits = sum(1 for w in words_lower if w in _NEGATIVE_WORDS)
        feat[i, 0] = np.clip((pos_hits - neg_hits) / n_words, -1.0, 1.0)
        feat[i, 1] = np.clip((pos_hits + neg_hits) / n_words,  0.0, 1.0)

        # 2-4: punctuation signals
        feat[i, 2] = 1.0 if '!' in text else 0.0
        feat[i, 3] = text.count('!') / n_words
        feat[i, 4] = 1.0 if '?' in text else 0.0

        # 5: ALL-CAPS ratio (already a ratio)
        caps_count = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
        feat[i, 5] = caps_count / n_words

        # 6-14, 17-18: lexicon hits — all normalized to per-word rates
        feat[i, 6]  = sum(1 for w in words_lower if w in _INTENSIFIERS)  / n_words
        feat[i, 7]  = sum(1 for w in words_lower if w in _FEAR_WORDS)     / n_words
        feat[i, 8]  = sum(1 for w in words_lower if w in _PATRIOTIC_WORDS)/ n_words
        feat[i, 9]  = sum(1 for w in words_lower if w in _AUTHORITY_WORDS)/ n_words
        feat[i, 10] = sum(1 for w in words_lower if w in _DOUBT_WORDS)    / n_words
        feat[i, 11] = sum(1 for w in words_lower if w in _LOADED_WORDS)   / n_words
        feat[i, 12] = sum(1 for w in words_lower if w in _NAME_CALL_WORDS)/ n_words
        feat[i, 13] = sum(1 for w in words_lower if w in
                          ('we', 'our', 'us', 'ourselves'))                / n_words
        feat[i, 14] = sum(1 for w in words_lower if w in
                          ('you', 'your', 'yourself', 'yourselves'))       / n_words
        feat[i, 17] = sum(1 for w in words_lower if w in _BANDWAGON_HITLERUM_WORDS) / n_words
        feat[i, 18] = sum(1 for w in words_lower if w in _LOGICAL_FALLACY_WORDS) / n_words

        # 15: fraction of known propaganda phrases that appear in this text
        feat[i, 15] = sum(1 for ph in _COMMON_PHRASES if ph in text_lower) / n_phrases

        # 16: log-length normalized to approximately [0, 1]
        feat[i, 16] = np.log1p(n_words) / np.log1p(300)

    return csr_matrix(feat)