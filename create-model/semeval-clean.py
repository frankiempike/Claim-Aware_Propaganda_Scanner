import os

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


def get_negative_samples(si_df, technique_cols, num_samples=1500):
    neg_samples = []
    #Group SI data to know where the "forbidden" propaganda zones are
    grouped = si_df.groupby(['article_id', 'text_content'])

    for (article_id, text), group in grouped:
        if len(neg_samples) >= num_samples: break

        #Create a mask of the whole text: True = Propaganda, False = Clean
        is_propaganda = [False] * len(text)
        for _, row in group.iterrows():
            for i in range(int(row['start_char']), int(row['end_char'])):
                if i < len(is_propaganda): is_propaganda[i] = True

        #Find "Clean" blocks (sequences of False)
        clean_start = None
        for i, val in enumerate(is_propaganda):
            if not val and clean_start is None:
                clean_start = i
            elif val and clean_start is not None:
                if (i - clean_start) > 6: #Only take meaningful snippets (>6 chars)
                    neg_samples.append({
                        'article_id': article_id,
                        'start_char': clean_start,
                        'end_char': i,
                        'span_text': text[clean_start:i][:200], #Cap length to 200
                        **{col: 0 for col in technique_cols}        #All techniques = 0
                    })
                clean_start = None
    return pd.DataFrame(neg_samples).sample(min(num_samples, len(neg_samples)))


#Define paths
BASE_DIR = Path("..").resolve()
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
if not PROCESSED_DIR.exists():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

#Load SemEval span identification data
semeval_si = pd.read_csv(INTERIM_DIR / "semeval_task1_si_merged.csv")

semeval_si['span_text'] = semeval_si.apply(lambda row: row['text_content'][row['start_char']:row['end_char']], axis=1)

semeval_si_grouped = semeval_si.groupby(['article_id', 'text_content']).agg({
    'start_char': list,
    'end_char': list
}).reset_index()

#Group spans by article because final model will only be given article/webpage full text, not spans
semeval_si_grouped['propaganda_offsets'] = semeval_si_grouped.apply(
    lambda x: list(zip(x['start_char'], x['end_char'])), axis=1
)
semeval_si_grouped = semeval_si_grouped.rename(columns={'text_content': 'text'})
semeval_si_grouped = semeval_si_grouped.drop(columns=['start_char', 'end_char'])

semeval_si_cleaned = semeval_si_grouped.copy()
semeval_si_cleaned['propaganda_offsets'] = semeval_si_cleaned['propaganda_offsets'].apply(json.dumps)

# Save cleaned SI data for model training
output_path = BASE_DIR / "data" / "processed" / "semeval_si_cleaned.csv"
semeval_si_cleaned.to_csv(output_path, index=False)

# Load and clean TC data for tc model training and future analysis
semeval_tc = pd.read_csv(INTERIM_DIR / "semeval_task2_tc_merged.csv")
semeval_tc['technique'] = semeval_tc['technique'].fillna('Unknown')

#One-hot encode techniques
technique_dummies = semeval_tc['technique'].str.get_dummies(sep=',')
semeval_tc = pd.concat([semeval_tc, technique_dummies], axis=1)

#Get relevant text being evaluated for each row (using span on text)
semeval_tc['span_text'] = semeval_tc.apply(lambda row: row['text_content'][row['start_char']:row['end_char']], axis=1)

#Aggregate to one row per unique span (a span may have multiple technique labels)
aggregation_rules = {col: 'max' for col in technique_dummies.columns}
aggregation_rules.update({
    'article_id': 'first',
    'text_content': 'first',
    'start_char': 'first',
    'end_char': 'first',
    'span_text': 'first'
})
semeval_tc = semeval_tc.groupby(['article_id', 'start_char', 'end_char']).agg(aggregation_rules).reset_index(drop=True)

#Add random non-propaganda spans to reduce model bias toward positive predictions
print("Generating negative samples to reduce model bias")
df_tc_neg = get_negative_samples(semeval_si, technique_dummies.columns)

#Combine positive (propaganda) and negative (clean) examples
semeval_tc = pd.concat([semeval_tc, df_tc_neg], ignore_index=True).fillna(0)

semeval_tc = get_features(semeval_tc)

#Reorder columns so that input columns come first, then engineered, then output columns
metadata_cols = ['article_id', 'text_content', 'span_text', 'start_char', 'end_char']
feature_cols  = ['sentiment', 'punct_count', 'lexical_diversity']
tech_cols = technique_dummies.columns.tolist()
final_column_order = metadata_cols + feature_cols + tech_cols
semeval_tc = semeval_tc[final_column_order]
semeval_tc = semeval_tc.rename(columns={'start_char': 'start', 'end_char': 'end'})

output_path = BASE_DIR / "data" / "processed" / "semeval_tc_cleaned.csv"
semeval_tc.to_csv(output_path, index=False)
