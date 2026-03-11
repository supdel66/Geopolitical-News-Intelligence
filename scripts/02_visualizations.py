#!/usr/bin/env python3
"""
Advanced Data Visualizations
Conflict News Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette("husl", 10)

print("=" * 60)
print("ADVANCED VISUALIZATIONS")
print("=" * 60)

# Load data
print("\n[1/8] Loading data...")
df = pd.read_csv('../data/raw/news_articles.csv')

# Add derived columns
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['date'] = df['published_at'].dt.date
df['year_month'] = df['published_at'].dt.to_period('M')
df['day_of_week'] = df['published_at'].dt.day_name()
df['hour'] = df['published_at'].dt.hour

df['content_length'] = df['content'].fillna('').apply(len)
df['word_count'] = df['content'].fillna('').apply(lambda x: len(str(x).split()))

print(f"Loaded {len(df)} articles")

# ============================================================
# 1. Word Cloud
# ============================================================
print("\n[2/8] Creating Word Cloud...")

try:
    from wordcloud import WordCloud

    all_text = ' '.join(df['content'].fillna('').astype(str).tolist())

    stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'will', 'their',
                 'what', 'about', 'which', 'when', 'make', 'like', 'time', 'just',
                 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some',
                 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
                 'only', 'come', 'its', 'over', 'said', 'also', 'more', 'after',
                 'but', 'not', 'are', 'was', 'were', 'has', 'have', 'been', 'being',
                 'one', 'two', 'three', 'four', 'five', 'first', 'new', 'way'}

    wordcloud = WordCloud(width=1200, height=600,
                          background_color='white',
                          max_words=100,
                          stopwords=stopwords,
                          colormap='viridis').generate(all_text)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud - Conflict News Content', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('../reports/figures/wordcloud.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reports/figures/wordcloud.png")

except ImportError:
    print("WordCloud not installed. Skipping...")

# ============================================================
# 2. Source Analysis
# ============================================================
print("\n[3/8] Source Analysis...")

source_counts = df['source'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Horizontal bar chart
top_sources = source_counts.head(12)
colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_sources)))

axes[0].barh(range(len(top_sources)), top_sources.values, color=colors)
axes[0].set_yticks(range(len(top_sources)))
axes[0].set_yticklabels(top_sources.index)
axes[0].invert_yaxis()
axes[0].set_xlabel('Number of Articles')
axes[0].set_title('Top 12 News Sources', fontsize=14, fontweight='bold')

for i, v in enumerate(top_sources.values):
    axes[0].text(v + 1, i, str(v), va='center')

# Stacked area chart - top 5 sources over time
top5_sources = source_counts.head(5).index.tolist()
df['top_source'] = df['source'].apply(lambda x: x if x in top5_sources else 'Other')

source_pivot = df.groupby(['date', 'top_source']).size().unstack(fill_value=0)
source_pivot.plot(kind='area', stacked=True, ax=axes[1], alpha=0.7)
axes[1].set_title('Top Sources Over Time (Stacked)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Number of Articles')
axes[1].legend(loc='upper left', fontsize=8)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/source_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/source_analysis.png")

# ============================================================
# 3. Temporal Heatmap
# ============================================================
print("\n[4/8] Temporal Heatmap...")

df_valid = df[df['hour'].notna() & df['day_of_week'].notna()]
df_valid['hour'] = df_valid['hour'].astype(int)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = df_valid.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
heatmap_data = heatmap_data.reindex(day_order)

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='d', ax=ax)
ax.set_title('Articles by Day of Week and Hour', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Day of Week')

plt.tight_layout()
plt.savefig('../reports/figures/temporal_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/temporal_heatmap.png")

# ============================================================
# 4. Monthly Trend Analysis
# ============================================================
print("\n[5/8] Monthly Trend Analysis...")

monthly_counts = df.groupby('year_month').size()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

monthly_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Articles per Month', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Number of Articles')
axes[0].tick_params(axis='x', rotation=45)

cumulative = monthly_counts.cumsum()
cumulative.plot(ax=axes[1], color='green', linewidth=2, marker='o')
axes[1].set_title('Cumulative Articles Over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Cumulative Count')

plt.tight_layout()
plt.savefig('../reports/figures/monthly_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/monthly_trends.png")

# ============================================================
# 5. Content Length by Source
# ============================================================
print("\n[6/8] Content Analysis by Source...")

top_sources_list = source_counts.head(10).index.tolist()
df_top = df[df['source'].isin(top_sources_list)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

order = df_top.groupby('source')['content_length'].median().sort_values(ascending=False).index
sns.boxplot(data=df_top, x='source', y='content_length', order=order, ax=axes[0], palette='viridis')
axes[0].set_title('Content Length by Source', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Source')
axes[0].set_ylabel('Content Length (chars)')
axes[0].tick_params(axis='x', rotation=45)

sns.violinplot(data=df_top, x='source', y='word_count', order=order, ax=axes[1], palette='viridis')
axes[1].set_title('Word Count Distribution by Source', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Source')
axes[1].set_ylabel('Word Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/content_by_source.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/content_by_source.png")

# ============================================================
# 6. Bigram Analysis
# ============================================================
print("\n[7/8] Bigram Analysis...")

def get_bigrams(text, n=15):
    if not text:
        return []
    words = text.lower().split()
    return [' '.join(words[i:i+2]) for i in range(len(words)-1)]

all_bigrams = []
for text in df['content'].fillna(''):
    all_bigrams.extend(get_bigrams(text))

bigram_freq = Counter(all_bigrams)

stopwords_bigrams = {'the united', 'in the', 'of the', 'and the', 'to the',
                     'for the', 'on the', 'at the', 'with the', 'from the'}
filtered_bigrams = {k: v for k, v in bigram_freq.items() if k not in stopwords_bigrams}

top_bigrams = dict(Counter(filtered_bigrams).most_common(20))

fig, ax = plt.subplots(figsize=(12, 8))
words = list(top_bigrams.keys())
counts = list(top_bigrams.values())

ax.barh(range(len(words)), counts, color='coral', alpha=0.8)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_title('Top 20 Bigrams (Word Pairs)', fontsize=14, fontweight='bold')
ax.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('../reports/figures/bigrams.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/bigrams.png")

# ============================================================
# 7. Sentiment & Category Analysis (if available)
# ============================================================
print("\n[8/8] Sentiment & Category Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

if 'sentiment_label' in df.columns:
    sentiment_counts = df['sentiment_label'].value_counts()
    colors_sentiment = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
    colors_list = [colors_sentiment.get(x, 'blue') for x in sentiment_counts.index]

    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index,
                   autopct='%1.1f%%', colors=colors_list)
    axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

    df['sentiment_polarity'] = pd.to_numeric(df['sentiment_polarity'], errors='coerce')
    sentiment_daily = df.groupby('date')['sentiment_polarity'].mean()
    axes[0, 1].plot(sentiment_daily.index, sentiment_daily.values, marker='o', color='purple')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Sentiment Trend Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Average Polarity')
    axes[0, 1].tick_params(axis='x', rotation=45)
else:
    axes[0, 0].text(0.5, 0.5, 'Sentiment data not available\nRun enhanced ETL pipeline',
                    ha='center', va='center', fontsize=12)
    axes[0, 1].text(0.5, 0.5, 'Sentiment trend requires\nenhanced data', ha='center', va='center')

if 'conflict_category' in df.columns:
    category_counts = df['conflict_category'].value_counts()
    axes[1, 0].barh(range(len(category_counts)), category_counts.values, color='teal')
    axes[1, 0].set_yticks(range(len(category_counts)))
    axes[1, 0].set_yticklabels(category_counts.index)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title('Conflict Categories', fontsize=14, fontweight='bold')

    cat_sentiment = pd.crosstab(df['conflict_category'], df['sentiment_label'], normalize='index')
    cat_sentiment.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='RdYlGn')
    axes[1, 1].set_title('Sentiment by Category', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].legend(title='Sentiment')
    axes[1, 1].tick_params(axis='x', rotation=45)
else:
    axes[1, 0].text(0.5, 0.5, 'Category data not available\nRun enhanced ETL pipeline',
                    ha='center', va='center', fontsize=12)
    axes[1, 1].text(0.5, 0.5, 'Category analysis requires\nenhanced data', ha='center', va='center')

plt.tight_layout()
plt.savefig('../reports/figures/sentiment_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/sentiment_category.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print("\nGenerated files in reports/figures/:")
print("  - wordcloud.png")
print("  - source_analysis.png")
print("  - temporal_heatmap.png")
print("  - monthly_trends.png")
print("  - content_by_source.png")
print("  - bigrams.png")
print("  - sentiment_category.png")
print("\nRun enhanced ETL pipeline to see sentiment/category visualizations!")
