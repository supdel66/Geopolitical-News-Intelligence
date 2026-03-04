import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import string

EDA_DIR = "eda_output"

# Keywords used during scraping mapping to geopolitical WW3 context
KEYWORDS = [
    "war", "iran", "israel", "united states",
    "us", "missile", "attack", "military",
    "retaliation", "conflict", "gaza",
    "tehran", "hezbollah", "world war 3", 
    "ww3", "escalation", "nuclear"
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def run_eda(df):
    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)
        
    if df.empty:
        print("No data available for EDA.")
        return

    print("Generating EDA charts...")
    sns.set_theme(style="whitegrid")
    
    # === 1. Sources Distribution ===
    plt.figure(figsize=(10, 6))
    source_counts = df['source'].value_counts()
    sns.barplot(x=source_counts.values, y=source_counts.index, hue=source_counts.index, palette="viridis", legend=False)
    plt.title('News Articles by Source', fontsize=14)
    plt.xlabel('Number of Articles')
    plt.ylabel('Source')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "top_sources.png"))
    plt.close()
    print("-> Saved top sources chart.")

    # === 2. Keyword Frequency Analysis ===
    df['combined_text'] = (df['title'].fillna('') + " " + df['content'].fillna('')).apply(clean_text)
    all_text = " ".join(df['combined_text'].values)
    
    # Split by whitespace and count basic occurrences of exact defined keywords
    words = all_text.split()
    word_counts = Counter()
    for w in words:
        if w in KEYWORDS:
            word_counts[w] += 1
            
    # Also attempt catching multi-word keywords explicitly
    for multi_kw in ["united states", "world war 3"]:
        count = all_text.count(multi_kw)
        if count > 0:
            word_counts[multi_kw] += count
            
    sorted_keywords = dict(sorted({k: v for k, v in word_counts.items() if v > 0}.items(), 
                                  key=lambda item: item[1], reverse=True)[:15])
    
    if sorted_keywords:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(sorted_keywords.keys()), y=list(sorted_keywords.values()), hue=list(sorted_keywords.keys()), palette="magma", legend=False)
        plt.title('Top Target Keyword Frequencies in Scraped Articles', fontsize=14)
        plt.xlabel('Keyword')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, "top_keywords.png"))
        plt.close()
        print("-> Saved top keywords chart.")

    # === 3. Timeline / Published Date Basis ===
    # Attempt to parse dates robustly
    df['parsed_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
    
    # Drop rows where dates couldn't be parsed
    df_timeline = df.dropna(subset=['parsed_date']).copy()
    if not df_timeline.empty:
        df_timeline['date_only'] = df_timeline['parsed_date'].dt.date
        daily_counts = df_timeline.groupby('date_only').size().reset_index(name='count')
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=daily_counts, x='date_only', y='count', marker='o', color="#e74c3c", linewidth=2.5)
        plt.title('Articles Published Over Time', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, "articles_over_time.png"))
        plt.close()
        print("-> Saved timeline chart.")
    else:
        print("-> Could not parse dates for timeline chart.")

    # === 4. Article Length Distribution ===
    df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
    if not df['word_count'].empty and df['word_count'].max() > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['word_count'], bins=20, color="#3498db", kde=True)
        plt.title('Distribution of Article Word Counts', fontsize=14)
        plt.xlabel('Word Count (Title + Content)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, "article_length_dist.png"))
        plt.close()
        print("-> Saved article length distribution chart.")

    print("EDA execution completed successfully.")
