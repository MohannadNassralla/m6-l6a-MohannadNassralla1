import spacy
import torch
from transformers import pipeline
import pandas as pd
import os

# 1. Load Multilingual Models
try:
    # Load spaCy Multilingual model
    # Make sure you ran: python -m spacy download xx_ent_wiki_sm
    nlp_spacy = spacy.load("xx_ent_wiki_sm")
    
    # Load Hugging Face XLM-RoBERTa (Multilingual)
    # This will download about 1.1GB on the first run
    ner_hf = pipeline("ner", model="Davlan/xlm-roberta-base-wikiann-ner", aggregation_strategy="simple")
    print(" Models loaded successfully.")
except Exception as e:
    print(f" Error loading models: {e}")
    print("Hint: Ensure 'torch' is installed and 'xx_ent_wiki_sm' is downloaded.")
    exit()

# 2. Load Data and Handle Column Names (Fixing KeyError)
file_path = "data/climate_articles.csv" 

if os.path.exists(file_path):
    df_all = pd.read_csv(file_path)
    
    # Check if column is 'lang' or 'language'
    target_col = 'lang' if 'lang' in df_all.columns else 'language'
    
    if target_col in df_all.columns:
        # Stratified sampling: precisely 20 English and 20 Arabic rows
        df_en = df_all[df_all[target_col] == 'en'].head(20)
        df_ar = df_all[df_all[target_col] == 'ar'].head(20)
        df = pd.concat([df_en, df_ar]).reset_index(drop=True)
        
        # Standardize column name to 'lang'
        df = df.rename(columns={target_col: 'lang'})
        print(f" Data loaded: {len(df)} rows (20 EN, 20 AR).")
    else:
        print(f"Error: Could not find language column. Available: {list(df_all.columns)}")
        exit()
else:
    print(f" Error: File not found at {file_path}")
    exit()

# 3. NER Extraction Functions
def get_spacy_ents(text):
    doc = nlp_spacy(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_hf_ents(text):
    results = ner_hf(str(text))
    return [(res['word'], res['entity_group']) for res in results]

# 4. Processing Loop
final_results = []

print(" Processing texts (this might take a moment on your CPU)...")

for index, row in df.iterrows():
    text = row['text']
    lang = row['lang']
    
    spacy_tags = get_spacy_ents(text)
    hf_tags = get_hf_ents(text)
    
    final_results.append({
        "lang": lang,
        "word_count": len(str(text).split()),
        "spacy_count": len(spacy_tags),
        "hf_count": len(hf_tags),
        "spacy_tags": spacy_tags,
        "hf_tags": hf_tags
    })

# 5. Generate Final Comparison Data
analysis_df = pd.DataFrame(final_results)

# Aggregate Stats for the Summary Table
summary = analysis_df.groupby('lang').agg({
    'spacy_count': ['sum', 'mean'],
    'hf_count': ['sum', 'mean'],
    'word_count': 'sum'
})

# Calculate Entity Density per 100 words
summary['spacy_density'] = (summary[('spacy_count', 'sum')] / summary[('word_count', 'sum')]) * 100
summary['hf_density'] = (summary[('hf_count', 'sum')] / summary[('word_count', 'sum')]) * 100

print("\n" + "="*30)
print("NER BILINGUAL COMPARISON TABLE")
print("="*30)
print(summary)

# 6. Report "No Entities Found" Rate
print("\n--- Zero-Entity Detection Rate ---")
for lang in ['en', 'ar']:
    no_ents = len(analysis_df[(analysis_df['lang'] == lang) & (analysis_df['hf_count'] == 0)])
    print(f"{lang.upper()} articles with 0 entities (HF): {no_ents}/20")

print("\n Script completed. Use the data above for your Lab 6A report.")