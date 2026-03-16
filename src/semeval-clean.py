import pandas as pd
from pathlib import Path
from textblob import TextBlob
import re
import nltk
import json
nltk.download('punkt')
nltk.download('punkt_tab')


def get_features(df):
    #Add Sentiment Score (-1.0 to 1.0): Propaganda often uses highly positive (Flag-Waving) or highly negative (Name-Calling) sentiment.
    df['sentiment'] = df['span_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

    #Add Punctuation Density: Excessive use of exclamation points or quotes often correlates with "Exaggeration" or "Doubt."
    df['punct_count'] = df['span_text'].apply(lambda x: len(re.findall(r'[!?"]', str(x))))

    #Add Lexical Diversity (Unique words / Total words): "Repetition" and "Slogans" have low lexical diversity.
    def lex_div(text):
        words = str(text).lower().split()
        if len(words) == 0: return 0
        return len(set(words)) / len(words)
    df['lexical_diversity'] = df['span_text'].apply(lex_div)

    #In the future, add Part-of-Speech (POS) Tags: High counts of adjectives and adverbs often signal "Loaded Language."

    return df

#Define paths
BASE_DIR = Path("..").resolve()

print(f"Base directory: {BASE_DIR}")

INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

#Load SemEval span identification data
semeval_si = pd.read_csv(INTERIM_DIR / "semeval_task1_si_merged.csv")

semeval_si['span_text'] = semeval_si.apply(lambda row: row['text_content'][row['start_char']:row['end_char']], axis=1)

semeval_si_grouped = semeval_si.groupby(['article_id', 'text_content']).agg({
    'start_char': list,
    'end_char': list
}).reset_index()
semeval_si_grouped['propaganda_offsets'] = semeval_si_grouped.apply(
    lambda x: list(zip(x['start_char'], x['end_char'])), axis=1
)
semeval_si_grouped = semeval_si_grouped.rename(columns={'text_content': 'text'})
semeval_si_grouped = semeval_si_grouped.drop(columns=['start_char', 'end_char'])

semeval_si_cleaned = semeval_si_grouped.copy()
semeval_si_cleaned['propaganda_offsets'] = semeval_si_cleaned['propaganda_offsets'].apply(json.dumps)

output_path = BASE_DIR / "data" / "processed" / "semeval_si_cleaned.csv"
semeval_si_cleaned.to_csv(output_path, index=False)

semeval_tc = pd.read_csv(INTERIM_DIR / "semeval_task2_tc_merged.csv")
semeval_tc['span_text'] = semeval_tc.apply(lambda row: row['text_content'][row['start_char']:row['end_char']], axis=1)
semeval_tc['technique'] = semeval_tc['technique'].fillna('Unknown')

semeval_tc['technique_list'] = semeval_tc['technique'].astype(str).str.split(',')
semeval_tc = semeval_tc.explode('technique_list')
semeval_tc['technique_list'] = semeval_tc['technique_list'].str.strip()
semeval_tc = semeval_tc[semeval_tc['technique_list'] != ""]

semeval_tc = get_features(semeval_tc)

output_path = BASE_DIR / "data" / "processed" / "semeval_tc_cleaned.csv"
semeval_tc.to_csv(output_path, index=False)
