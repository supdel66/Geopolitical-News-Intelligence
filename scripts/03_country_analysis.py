#!/usr/bin/env python3
"""
Country Analysis - Extract country statistics and relationships from conflict news
"""

import sqlite3
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from piyush.config import DB_PATH

print("=" * 70)
print("COUNTRY ANALYSIS - Conflict News Dataset")
print("=" * 70)

# Connect to database
conn = sqlite3.connect(DB_PATH)

# Load articles with countries
query = """
    SELECT id, title, content, published_at, source,
           sentiment_polarity, sentiment_label, conflict_category, countries
    FROM news_articles
    WHERE countries IS NOT NULL AND countries != ''
"""

df = pd.read_sql_query(query, conn)
print(f"\nLoaded {len(df)} articles with country data")

# Parse dates
df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
df = df.dropna(subset=['published_at'])

# Get date range
print(f"Date range: {df['published_at'].min().date()} to {df['published_at'].max().date()}")

# ============================================================
# 1. Country Statistics
# ============================================================
print("\n[1/4] Calculating country statistics...")

# Explode countries (articles can have multiple countries)
df_countries = df.dropna(subset=['countries']).copy()
df_countries['country_list'] = df_countries['countries'].str.split(',')

# Count each country
country_stats = defaultdict(lambda: {
    'total_mentions': 0,
    'positive': 0,
    'negative': 0,
    'neutral': 0,
    'avg_polarity': [],
    'categories': defaultdict(int),
    'recent_mentions': 0,
    'sources': set()
})

# Define "recent" as last 30 days
recent_date = df['published_at'].max() - timedelta(days=30)

for idx, row in df_countries.iterrows():
    countries = row['country_list']
    polarity = row['sentiment_polarity']
    sentiment = row['sentiment_label']
    category = row['conflict_category']
    is_recent = row['published_at'] >= recent_date

    for country in countries:
        country = country.strip()
        if not country:
            continue

        country_stats[country]['total_mentions'] += 1
        country_stats[country]['avg_polarity'].append(polarity)
        country_stats[country][sentiment.lower()] += 1
        country_stats[country]['categories'][category] += 1
        country_stats[country]['sources'].add(row['source'])

        if is_recent:
            country_stats[country]['recent_mentions'] += 1

# Calculate averages and build final stats
country_data = []
for country, stats in country_stats.items():
    avg_polarity = np.mean(stats['avg_polarity']) if stats['avg_polarity'] else 0

    # Calculate negative ratio (indicator of conflict involvement)
    total = stats['positive'] + stats['negative'] + stats['neutral']
    negative_ratio = stats['negative'] / total if total > 0 else 0

    # Recent trend (mentions in last 30 days / total)
    recent_trend = stats['recent_mentions'] / stats['total_mentions'] if stats['total_mentions'] > 0 else 0

    country_data.append({
        'country': country,
        'total_mentions': stats['total_mentions'],
        'positive_articles': stats['positive'],
        'negative_articles': stats['negative'],
        'neutral_articles': stats['neutral'],
        'negative_ratio': round(negative_ratio * 100, 2),
        'avg_sentiment_polarity': round(avg_polarity, 3),
        'recent_mentions': stats['recent_mentions'],
        'recent_trend': round(recent_trend * 100, 2),
        'unique_sources': len(stats['sources']),
        'top_category': max(stats['categories'].items(), key=lambda x: x[1])[0] if stats['categories'] else 'Unknown'
    })

# Sort by total mentions
country_df = pd.DataFrame(country_data)
country_df = country_df.sort_values('total_mentions', ascending=False)

print(f"\nTop 15 countries by mention count:")
print("-" * 60)
for _, row in country_df.head(15).iterrows():
    print(f"  {row['country']:25} | Mentions: {row['total_mentions']:4} | Neg: {row['negative_ratio']:5.1f}% | Polarity: {row['avg_sentiment_polarity']:+.3f}")

# ============================================================
# 2. Country-Pair Relationships (Tensions)
# ============================================================
print("\n[2/4] Building country-pair relationships...")

# Create pairs from articles with multiple countries
pair_stats = defaultdict(lambda: {
    'co_mentions': 0,
    'polarities': [],
    'sentiments': []
})

for idx, row in df_countries.iterrows():
    countries = row['country_list']
    polarity = row['sentiment_polarity']
    sentiment = row['sentiment_label']

    # Create all pairs
    if len(countries) >= 2:
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                c1 = countries[i].strip()
                c2 = countries[j].strip()

                if c1 and c2:
                    pair = tuple(sorted([c1, c2]))
                    pair_stats[pair]['co_mentions'] += 1
                    pair_stats[pair]['polarities'].append(polarity)
                    pair_stats[pair]['sentiments'].append(sentiment)

# Calculate tension scores
pair_data = []
for pair, stats in pair_stats.items():
    avg_polarity = np.mean(stats['polarities']) if stats['polarities'] else 0
    negative_count = stats['sentiments'].count('negative')
    total = len(stats['sentiments'])

    # Tension score: higher when more negative co-mentions
    tension_score = (negative_count / total * 100) if total > 0 else 0

    pair_data.append({
        'country_1': pair[0],
        'country_2': pair[1],
        'co_mentions': stats['co_mentions'],
        'avg_polarity': round(avg_polarity, 3),
        'negative_mentions': negative_count,
        'tension_score': round(tension_score, 2)
    })

pair_df = pd.DataFrame(pair_data)
pair_df = pair_df.sort_values('co_mentions', ascending=False)

print(f"\nTop 15 country pairs (most mentioned together):")
print("-" * 70)
for _, row in pair_df.head(15).iterrows():
    print(f"  {row['country_1']:15} - {row['country_2']:15} | Co-mentions: {row['co_mentions']:3} | Tension: {row['tension_score']:5.1f}%")

# Top tensions (negative sentiment pairs)
print(f"\nTop 10 tension pairs (highest negative ratio):")
print("-" * 70)
negative_pairs = pair_df[pair_df['negative_mentions'] >= 2].sort_values('tension_score', ascending=False)
for _, row in negative_pairs.head(10).iterrows():
    print(f"  {row['country_1']:15} - {row['country_2']:15} | Neg: {row['negative_mentions']:2}/{row['co_mentions']} | Tension: {row['tension_score']:5.1f}%")

# ============================================================
# 3. Top Tensions per Country
# ============================================================
print("\n[3/4] Analyzing tensions per country...")

# For each country, find what countries it's most in tension with
country_tensions = {}
for _, row in country_df.head(20).iterrows():
    country = row['country']

    # Find pairs involving this country
    mask = (pair_df['country_1'] == country) | (pair_df['country_2'] == country)
    country_pairs = pair_df[mask].copy()

    if len(country_pairs) > 0:
        tensions = []
        for _, p in country_pairs.iterrows():
            other = p['country_2'] if p['country_1'] == country else p['country_1']
            tensions.append({
                'country': other,
                'co_mentions': p['co_mentions'],
                'tension_score': p['tension_score']
            })

        # Sort by tension score
        tensions = sorted(tensions, key=lambda x: x['tension_score'], reverse=True)[:5]
        country_tensions[country] = tensions

# ============================================================
# 4. Save Results
# ============================================================
print("\n[4/4] Saving results...")

# Save country stats
country_df.to_csv('../data/processed/country_stats.csv', index=False)
print("Saved: data/processed/country_stats.csv")

# Save country pairs
pair_df.to_csv('../data/processed/country_pairs.csv', index=False)
print("Saved: data/processed/country_pairs.csv")

# Save tensions as JSON
with open('../data/processed/country_tensions.json', 'w') as f:
    json.dump(country_tensions, f, indent=2)
print("Saved: data/processed/country_tensions.json")

# Summary statistics
summary = {
    'total_articles_analyzed': len(df),
    'date_range': {
        'start': str(df['published_at'].min().date()),
        'end': str(df['published_at'].max().date())
    },
    'unique_countries': len(country_df),
    'unique_country_pairs': len(pair_df),
    'top_countries': country_df.head(10)[['country', 'total_mentions', 'negative_ratio', 'avg_sentiment_polarity']].to_dict('records'),
    'top_tensions': negative_pairs.head(10)[['country_1', 'country_2', 'tension_score']].to_dict('records')
}

with open('../data/processed/country_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("Saved: data/processed/country_analysis_summary.json")

conn.close()

print("\n" + "=" * 70)
print("COUNTRY ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nKey findings:")
print(f"  - {summary['unique_countries']} unique countries identified")
print(f"  - {summary['unique_country_pairs']} country pairs with co-mentions")
print(f"  - Top country: {summary['top_countries'][0]['country']} ({summary['top_countries'][0]['total_mentions']} mentions)")
print(f"  - Highest tension: {summary['top_tensions'][0]['country_1']} vs {summary['top_tensions'][0]['country_2']} ({summary['top_tensions'][0]['tension_score']}% negative)")
