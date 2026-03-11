#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA)
Conflict News Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import re
import json
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - CONFLICT NEWS")
print("=" * 60)

# Load data
print("\n[1/10] Loading data...")
df = pd.read_csv('../data/raw/news_articles.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data Overview
print("\n[2/10] Data Overview...")
print("-" * 40)
print("DATA TYPES:")
print(df.dtypes)

print("\nFIRST 5 ROWS:")
print(df.head())

# Data Quality
print("\n[3/10] Data Quality Check...")
print("-" * 40)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
print("MISSING VALUES:")
print(missing_df[missing_df['Missing'] > 0])

print(f"\nDuplicates (by link): {df.duplicated(subset=['link']).sum()}")
print(f"Duplicates (by title): {df.duplicated(subset=['title']).sum()}")

# Basic Stats
print("\n[4/10] Basic Statistics...")
print("-" * 40)
print(df.describe())

# Source Distribution
print("\n[5/10] Source Distribution...")
print("-" * 40)

source_counts = df['source'].value_counts()
print("ARTICLES BY SOURCE:")
print(source_counts)

# Plot source distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
source_counts.head(15).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Top 15 News Sources', fontsize=14)
axes[0].set_xlabel('Number of Articles')
axes[0].invert_yaxis()

# Pie chart (top 10)
source_counts.head(10).plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Top 10 Sources Distribution', fontsize=14)
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('../reports/figures/source_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/source_distribution.png")

# Date Analysis
print("\n[6/10] Date Analysis...")
print("-" * 40)

df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df['date'] = df['published_at'].dt.date
df['year'] = df['published_at'].dt.year
df['month'] = df['published_at'].dt.month
df['day_of_week'] = df['published_at'].dt.day_name()

print(f"Earliest article: {df['published_at'].min()}")
print(f"Latest article: {df['published_at'].max()}")
date_diff = (df['published_at'].max() - df['published_at'].min()).days
print(f"Date range: {date_diff} days")

# Time series
daily_counts = df.groupby('date').size()

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-', alpha=0.7)
axes[0].set_title('Articles Published Over Time', fontsize=14)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Number of Articles')
axes[0].tick_params(axis='x', rotation=45)

rolling_avg = daily_counts.rolling(window=7).mean()
axes[1].plot(daily_counts.index, daily_counts.values, alpha=0.5, label='Daily')
axes[1].plot(daily_counts.index, rolling_avg, color='red', linewidth=2, label='7-day Rolling Avg')
axes[1].set_title('Daily Articles with Rolling Average', fontsize=14)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Number of Articles')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/temporal_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/temporal_distribution.png")

# Day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_counts = df['day_of_week'].value_counts().reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 5))
dow_counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
ax.set_title('Articles by Day of Week', fontsize=14)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Number of Articles')
ax.set_xticklabels(day_order, rotation=45)

plt.tight_layout()
plt.savefig('../reports/figures/day_of_week.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/day_of_week.png")

# Content Analysis
print("\n[7/10] Content Analysis...")
print("-" * 40)

df['content_length'] = df['content'].fillna('').apply(len)
df['title_length'] = df['title'].fillna('').apply(len)
df['word_count'] = df['content'].fillna('').apply(lambda x: len(str(x).split()))

print(f"Avg content length: {df['content_length'].mean():.0f} chars")
print(f"Avg title length: {df['title_length'].mean():.0f} chars")
print(f"Avg word count: {df['word_count'].mean():.0f} words")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['content_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Content Length Distribution', fontsize=14)
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['content_length'].mean(), color='red', linestyle='--', label=f"Mean: {df['content_length'].mean():.0f}")
axes[0].legend()

axes[1].hist(df['word_count'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_title('Word Count Distribution', fontsize=14)
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {df['word_count'].mean():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig('../reports/figures/content_length_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/content_length_dist.png")

# Word Frequency
print("\n[8/10] Word Frequency Analysis...")
print("-" * 40)

all_text = ' '.join(df['content'].fillna('').astype(str).tolist())
all_words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())

stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'will', 'their',
             'what', 'about', 'which', 'when', 'make', 'like', 'time', 'just',
             'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some',
             'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
             'only', 'come', 'its', 'over', 'said', 'also', 'more', 'after',
             'but', 'not', 'are', 'was', 'were', 'has', 'have', 'been', 'being'}

filtered_words = [w for w in all_words if w not in stopwords]
word_freq = Counter(filtered_words)

print("TOP 30 MOST COMMON WORDS:")
for word, count in word_freq.most_common(30):
    print(f"  {word:20} : {count:5}")

top_words = word_freq.most_common(25)
words, counts = zip(*top_words)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(words)), counts, color='darkgreen', alpha=0.7)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_title('Top 25 Most Common Words', fontsize=14)
ax.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('../reports/figures/word_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/word_frequency.png")

# Correlation
print("\n[9/10] Correlation Analysis...")
print("-" * 40)

numeric_cols = ['content_length', 'title_length', 'word_count']
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Correlation Matrix', fontsize=14)

plt.tight_layout()
plt.savefig('../reports/figures/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: reports/figures/correlation_matrix.png")

# Summary
print("\n[10/10] Summary & Export...")
print("-" * 40)

print("\n" + "=" * 60)
print("KEY FINDINGS SUMMARY")
print("=" * 60)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   - Total articles: {len(df)}")
print(f"   - Unique sources: {df['source'].nunique()}")
print(f"   - Date range: {df['published_at'].min().date()} to {df['published_at'].max().date()}")
print(f"   - Coverage: {date_diff} days")

print(f"\n📰 TOP 5 SOURCES:")
for i, (src, cnt) in enumerate(source_counts.head(5).items(), 1):
    print(f"   {i}. {src}: {cnt} articles")

print(f"\n📝 CONTENT METRICS:")
print(f"   - Avg content length: {df['content_length'].mean():.0f} chars")
print(f"   - Avg word count: {df['word_count'].mean():.0f} words")
print(f"   - Max word count: {df['word_count'].max()} words")

print(f"\n🔝 TOP 5 WORDS:")
for i, (word, count) in enumerate(word_freq.most_common(5), 1):
    print(f"   {i}. {word}: {count}")

# Save processed data
df.to_csv('../data/processed/news_articles_eda.csv', index=False)
print("\nSaved: data/processed/news_articles_eda.csv")

# Summary stats to JSON
summary = {
    'total_articles': int(len(df)),
    'unique_sources': int(df['source'].nunique()),
    'date_range': {
        'start': str(df['published_at'].min()),
        'end': str(df['published_at'].max()),
        'days': date_diff
    },
    'content_stats': {
        'avg_length': float(df['content_length'].mean()),
        'avg_words': float(df['word_count'].mean()),
        'max_words': int(df['word_count'].max())
    },
    'top_sources': {k: int(v) for k, v in source_counts.head(10).to_dict().items()},
    'top_words': {w: c for w, c in word_freq.most_common(20)}
}

with open('../reports/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Saved: reports/eda_summary.json")
print("\n" + "=" * 60)
print("✅ EDA COMPLETE!")
print("=" * 60)
