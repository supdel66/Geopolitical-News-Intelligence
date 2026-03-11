# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ETL pipeline for collecting and analyzing conflict-related news from multiple sources (NewsAPI + RSS feeds). The project scrapes news articles about global conflicts, performs NLP analysis (sentiment, entities, keywords), and stores them in a SQLite database.

## Running the ETL Pipeline

```bash
cd piyush
python main.py
```

This runs the complete pipeline: extract from NewsAPI and RSS feeds → transform (filter, clean, analyze) → load to SQLite database.

## Running Analysis Scripts

```bash
# Run EDA
cd scripts
python 01_eda.py

# Run visualizations
python 02_visualizations.py
```

## Architecture

The project follows an ETL (Extract-Transform-Load) pattern:

```
main.py → extract.py → transform.py → load.py → database.py
```

**extract.py**: Fetches articles from two sources
- NewsAPI (paid service, uses API key in config.py)
- RSS feeds (13 sources configured in config.py)
- Applies rate limiting between requests

**transform.py**: Processes raw articles
- Filters by conflict-related keywords (defined in config.py)
- Cleans text (removes HTML, URLs, special characters)
- Analyzes sentiment using TextBlob
- Extracts named entities using spaCy
- Categorizes conflict type (Middle East, Ukraine/Russia, Asia Pacific, etc.)
- Performs topic modeling with sklearn (when enough articles available)

**load.py**: Stores processed articles in SQLite database with indexes on published_at, source, sentiment_label, and conflict_category.

**config.py**: Central configuration
- NEWS_API_KEY: API key for NewsAPI (default provided, override with env var)
- RSS_FEEDS: List of RSS feed URLs
- CONFLICT_KEYWORDS: Keywords used to filter relevant articles
- DB_PATH: SQLite database location

## Database Schema

The `news_articles` table contains: source, title, content, author, published_at, link, image_url, sentiment_polarity, sentiment_subjectivity, sentiment_label, conflict_category, keywords, entities_persons, entities_organizations, entities_locations, content_length.

## Dependencies

Install via: `pip install -r piyush/requirements.txt`

Required packages: requests, feedparser, textblob, spacy, scikit-learn

spaCy requires model download: `python -m spacy download en_core_web_sm`
