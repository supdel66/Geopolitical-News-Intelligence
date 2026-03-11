# Geopolitical News Pipeline & WW3 Context Analyzer

An automated, end-to-end Python ETL pipeline designed to scrape, store, embed, and analyze geopolitical news articles with a specific focus on global conflicts, escalation themes, and WW3 context analysis. The project features dual-database architecture, semantic search capabilities via RAG (Retrieval-Augmented Generation), and comprehensive exploratory data analysis.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Component Documentation](#component-documentation)
  - [1. Data Extraction Layer (Scraper)](#1-data-extraction-layer-scraper)
  - [2. Database Layer](#2-database-layer)
  - [3. EDA & Analytics Layer](#3-eda--analytics-layer)
  - [4. RAG API Layer](#4-rag-api-layer)
  - [5. Frontend Layer](#5-frontend-layer)
- [Output Artifacts](#output-artifacts)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a complete data pipeline for monitoring and analyzing global conflict news:

1. **Extract**: Fetches articles from 10+ RSS feeds and NewsAPI using keyword-based filtering
2. **Transform**: Cleans text, extracts metadata, and performs NLP analysis
3. **Load**: Stores structured data in SQLite and vector embeddings in ChromaDB
4. **Analyze**: Generates comprehensive EDA with 13+ visualization types
5. **Query**: Provides a RAG-powered API for semantic search and Q&A

The system is specifically tuned for conflict-related content with custom keyword dictionaries, escalation scoring, and geopolitical actor tracking.

---

## Features

### Data Collection
- **Multi-Source RSS Scraping**: BBC, Associated Press, Al Jazeera, Guardian, CNN, Fox News, New York Times, ABC News, Washington Post, Yahoo News
- **NewsAPI Integration**: Query-based article fetching with custom geopolitical search terms
- **Smart Filtering**: 60+ conflict-related keywords including escalation indicators, geographic hotspots, and military terminology
- **Full-Text Extraction**: BeautifulSoup-powered content extraction from article pages
- **Duplicate Prevention**: URL-based deduplication to avoid storing the same article twice

### Storage & Embeddings
- **SQLite Database**: Persistent storage for article metadata, content, and timestamps
- **ChromaDB Vector Store**: Semantic embeddings using Ollama's `nomic-embed-text` / `qwen3-embedding` models
- **Intelligent Chunking**: 500-character text chunks with sentence-aware splitting

### Analytics & Intelligence
- **WW3 Threat Level Indicator**: Composite scoring based on nuclear keywords, escalation terms, sentiment, and missile mentions
- **Conflict Theater Classification**: Automatic categorization (Middle East, Russia/Ukraine, Asia-Pacific, Global WW3, US Policy)
- **Actor Co-Occurrence Analysis**: Heatmap showing which geopolitical actors appear together in articles
- **Escalation Scoring**: Weighted keyword scoring with de-escalation term credits
- **Sentiment Analysis**: TextBlob-based polarity scoring with trend tracking
- **N-Gram Analysis**: Bigram and trigram extraction for phrase intelligence
- **Publication Velocity**: Source activity tracking over time

### Visualization & Reporting
- **Dark-Themed HTML Dashboard**: Auto-generated intelligence briefing with all metrics
- **13+ Chart Types**: Bar charts, pie charts, heatmaps, trend lines, area charts
- **Threat Level Gauge**: Visual composite score with color-coded severity zones

### RAG API
- **Semantic Search**: Query articles by meaning, not just keywords
- **LLM Integration**: Local Ollama-powered question answering with source attribution
- **Source Sidebar**: Displays matching articles with similarity distances

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          GEOPOLITICAL NEWS PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────────┐ │
│  │   RSS FEEDS  │     │   NewsAPI    │     │         KEYWORD FILTER           │ │
│  │  (10 sources)│     │  (API calls) │     │  60+ conflict-related terms      │ │
│  └──────┬───────┘     └──────┬───────┘     └─────────────┬────────────────────┘ │
│         │                    │                           │                      │
│         └────────────────────┼───────────────────────────┘                      │
│                              ▼                                                  │
│                    ┌─────────────────┐                                          │
│                    │     SCRAPER     │                                          │
│                    │  scraper.py     │                                          │
│                    │  - Full-text    │                                          │
│                    │  - Image URLs   │                                          │
│                    │  - Deduplication│                                          │
│                    └────────┬────────┘                                          │
│                             │                                                   │
│         ┌───────────────────┼───────────────────┐                               │
│         ▼                   ▼                   ▼                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐                      │
│  │   SQLITE    │    │  CHROMADB   │    │      EDA        │                      │
│  │  news.db    │    │  Vectors    │    │   Analysis      │                      │
│  │             │    │             │    │                 │                      │
│  │ - Metadata  │    │ - Embeddings│    │ - Threat Level  │                      │
│  │ - Content   │    │ - Semantic  │    │ - Sentiment     │                      │
│  │ - Timestamps│    │   Search    │    │ - Escalation    │                      │
│  └──────┬──────┘    └──────┬──────┘    │ - Actor Co-occur│                      │
│         │                  │           └────────┬────────┘                      │
│         │                  │                    │                               │
│         └──────────────────┼────────────────────┘                               │
│                            ▼                                                    │
│                  ┌─────────────────┐                                            │
│                  │  HTML DASHBOARD │                                            │
│                  │  report.html    │                                            │
│                  └─────────────────┘                                            │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                            RAG API LAYER                                    │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐ │ │
│  │  │   FastAPI   │───▶│  ChromaDB   │───▶│    Ollama LLM (llama3.2:3b)     │ │ │
│  │  │   api.py    │    │   Query     │    │    Semantic Q&A with Sources    │ │ │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                        FRONTEND LAYER (Next.js)                             │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────────┐ │ │
│  │  │  Chat UI    │    │  Sources    │    │    EDA Report Sidebar           │ │ │
│  │  │  Chatbot.tsx│    │  Sidebar    │    │    Embedded Dashboard View      │ │ │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Elective-Project-/
│
├── main.py                      # Central orchestrator - runs complete pipeline
├── api.py                       # FastAPI RAG endpoint with semantic search
├── utils.py                     # Logging decorators and utilities
├── requirements.txt             # Python dependencies
├── pipeline.log                 # Execution logs (auto-generated)
│
├── scraper/
│   └── scraper.py               # RSS & NewsAPI fetching, BeautifulSoup parsing
│
├── database/
│   ├── db.py                    # SQLite CRUD operations and initialization
│   └── vector_db.py             # ChromaDB embeddings with Ollama integration
│
├── eda_code/
│   ├── eda.py                   # Comprehensive EDA with 13 chart types
│   ├── vector_eda.py            # Semantic theme analysis via vector queries
│   └── report_generator.py      # HTML dashboard compilation
│
├── scripts/
│   ├── 01_eda.py                # Standalone EDA script
│   ├── 02_visualizations.py     # Advanced visualizations
│   └── 03_country_analysis.py   # Country-pair tension analysis
│
├── sqlite_databases/
│   └── news.db                  # SQLite database (auto-generated)
│
├── chromadb/                    # Vector database (auto-generated)
│
├── eda_output/                  # Analysis outputs (auto-generated)
│   ├── report.html              # Final intelligence dashboard
│   ├── top_sources.png          # Source distribution chart
│   ├── top_keywords.png         # Keyword frequency chart
│   ├── article_length_dist.png  # Article length histogram
│   ├── articles_over_time.png   # Timeline visualization
│   ├── country_actor_mentions.png
│   ├── conflict_theaters.png    # Theater pie chart
│   ├── actor_cooccurrence_heatmap.png
│   ├── escalation_trend.png     # Escalation score over time
│   ├── sentiment_analysis.png   # Sentiment by source and time
│   ├── source_velocity.png      # Stacked area chart
│   ├── weekly_heatmap.png       # Day × Week heatmap
│   └── threat_level.png         # WW3 threat gauge
│
├── ragfrontend/                 # Next.js frontend (optional)
│   ├── app/
│   │   ├── page.tsx             # Main chat interface
│   │   ├── layout.tsx           # App layout
│   │   └── globals.css          # Styles
│   ├── components/
│   │   ├── Chatbot.tsx          # RAG chat component
│   │   ├── SourceSidebar.tsx    # Article sources panel
│   │   └── EdaSidebar.tsx       # Dashboard embed
│   └── package.json
│
└── piyush/                      # Legacy/experimental directory
    └── venv/                    # Python virtual environment
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM and embeddings)
- **Node.js 18+** (optional, for frontend)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Elective-Project-
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
```
requests          # HTTP client for API calls
beautifulsoup4    # HTML parsing
feedparser        # RSS feed parsing
pandas            # Data manipulation
matplotlib        # Visualization
seaborn           # Statistical visualization
chromadb          # Vector database
ollama            # Local LLM integration
fastapi           # API framework
uvicorn           # ASGI server
textblob          # Sentiment analysis
scikit-learn      # N-gram extraction
```

### Step 3: Install spaCy Model (optional, for NER)

```bash
python -m spacy download en_core_web_sm
```

### Step 4: Install and Configure Ollama

1. **Download Ollama**: Visit [ollama.com](https://ollama.com) and install

2. **Pull Required Models**:
```bash
# Embedding model for ChromaDB
ollama pull nomic-embed-text

# Alternative embedding model (used in vector_db.py)
ollama pull qwen3-embedding:0.6B

# LLM for RAG Q&A
ollama pull llama3.2:3b
```

3. **Start Ollama Service**:
```bash
ollama serve
```

### Step 5: Set Environment Variables (Optional)

```bash
# Override default NewsAPI key
export NEWSAPI_KEY="your-api-key-here"
```

### Step 6: Install Frontend Dependencies (Optional)

```bash
cd ragfrontend
npm install
```

---

## Configuration

### Keyword Configuration

The scraper uses two keyword sets defined in `scraper/scraper.py`:

**Primary Keywords (RSS Filtering):**
```python
KEYWORDS = [
    "war", "iran", "israel", "united states", "us", "missile", "attack",
    "military", "retaliation", "conflict", "gaza", "tehran", "hezbollah",
    "world war 3", "ww3", "nuclear", "escalation", "world war iii",
    "global conflict", "third world war", "thermonuclear", "armageddon",
    "doomsday", "nato article 5", "mutual assured destruction", ...
]
```

**Extended Keywords (NewsAPI Queries):**
Includes additional terms for targeted API searches.

### Conflict Theater Classification

Defined in `eda_code/eda.py`:

```python
CONFLICT_THEATERS = {
    "Middle East": ["israel", "iran", "gaza", "hezbollah", ...],
    "Russia/Ukraine": ["russia", "ukraine", "putin", "zelensky", ...],
    "Asia-Pacific": ["china", "taiwan", "north korea", ...],
    "Global WW3": ["world war", "ww3", "nuclear", "armageddon", ...],
    "US Policy/Sanctions": ["pentagon", "white house", "us sanction", ...]
}
```

### Escalation Weights

Keywords are weighted for escalation scoring:

```python
ESCALATION_WEIGHTS = {
    "nuclear": 10, "thermonuclear": 10, "world war 3": 10, "ww3": 10,
    "missile": 5, "war": 5, "strike": 5, "drone": 4,
    "ceasefire": -3, "peace talks": -4, "diplomacy": -3  # De-escalation
}
```

---

## Usage

### Run the Complete Pipeline

```bash
python main.py
```

This executes the full ETL process:
1. Initialize SQLite database
2. Scrape articles from RSS feeds and NewsAPI
3. Save articles to SQLite
4. Generate embeddings in ChromaDB
5. Run EDA analysis
6. Generate HTML dashboard

### Run Specific Components

```python
# Only run EDA on existing data
from database.db import get_all_articles
from eda_code.eda import run_eda

df = get_all_articles()
stats = run_eda(df)
```

```python
# Only update vector database
from database.db import get_all_articles
from database.vector_db import store_articles_in_vector_db

df = get_all_articles()
store_articles_in_vector_db(df)
```

### Start the RAG API Server

```bash
python api.py
```

Or with uvicorn:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

### Start the Frontend (Optional)

```bash
cd ragfrontend
npm run dev
```

Frontend will be available at `http://localhost:3000`

---

## Component Documentation

### 1. Data Extraction Layer (Scraper)

**File**: `scraper/scraper.py`

#### RSS Feed Sources

| Source | Feed URL |
|--------|----------|
| BBC | `feeds.bbci.co.uk/news/rss.xml` |
| Associated Press | `feedx.net/rss/ap.xml` |
| Al Jazeera | `aljazeera.com/xml/rss/all.xml` |
| Guardian | `theguardian.com/world/rss` |
| CNN | `rss.cnn.com/rss/edition_world.rss` |
| Fox News | `moxie.foxnews.com/google-publisher/world.xml` |
| New York Times | `rss.nytimes.com/services/xml/rss/nyt/HomePage.xml` |
| ABC News | `abcnews.go.com/abcnews/topstories` |
| Washington Post | `feeds.washingtonpost.com/rss/national` |
| Yahoo News | `news.yahoo.com/rss/topstories` |

#### Functions

| Function | Description |
|----------|-------------|
| `fetch_rss_news()` | Iterates all RSS sources, extracts 15 latest per source |
| `fetch_newsapi_news(api_key)` | Queries NewsAPI with conflict keywords |
| `scrape_all_sources(api_key)` | Master orchestrator for all sources |
| `is_conflict_related(text)` | Keyword filtering function |
| `extract_full_article_*(url)` | Site-specific BeautifulSoup extractors |

#### Scraping Process

```
RSS Feed → Parse Entries → Check Duplicate (URL) → Extract Full Text
    ↓
Keyword Filter → Article Dict → Return List
```

---

### 2. Database Layer

#### SQLite Storage (`database/db.py`)

**Schema**:
```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    title TEXT,
    content TEXT,
    published_at TEXT,
    link TEXT UNIQUE,           -- Deduplication key
    scraped_at TEXT,
    img_link TEXT
);
```

**Functions**:
| Function | Description |
|----------|-------------|
| `init_db()` | Creates database and tables |
| `save_articles(articles)` | Inserts articles with deduplication |
| `get_all_articles()` | Returns all articles as DataFrame |
| `is_article_saved(url)` | Checks if URL already exists |

#### Vector Database (`database/vector_db.py`)

**Chunking Strategy**:
- Maximum chunk size: 500 characters
- Sentence-aware splitting (splits on ". ")
- Hard limit enforcement for oversized sentences

**Embedding Process**:
```python
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:0.6B"
)
```

**Metadata Stored with Each Chunk**:
```python
{
    "original_article_id": article_id,  # Links back to SQLite
    "source": source_name,
    "url": article_url,
    "published_at": timestamp
}
```

---

### 3. EDA & Analytics Layer

#### Main EDA (`eda_code/eda.py`)

**Generates 13 Charts:**

| Chart | File | Description |
|-------|------|-------------|
| 1. Top Sources | `top_sources.png` | Horizontal bar of article counts by source |
| 2. Keyword Frequency | `top_keywords.png` | Conflict keyword occurrence counts |
| 3. Article Timeline | `articles_over_time.png` | Daily publication counts |
| 4. Length Distribution | `article_length_dist.png` | Word count histogram |
| 5. Actor Mentions | `country_actor_mentions.png` | Geopolitical entity frequency |
| 6. Conflict Theaters | `conflict_theaters.png` | Donut chart of theater distribution |
| 7. Actor Co-occurrence | `actor_cooccurrence_heatmap.png` | Which actors appear together |
| 8. Escalation Trend | `escalation_trend.png` | Score distribution + rolling average |
| 9. Sentiment Analysis | `sentiment_analysis.png` | By source + over time |
| 10. Source Velocity | `source_velocity.png` | Stacked area of daily volume |
| 11. Weekly Heatmap | `weekly_heatmap.png` | Day-of-week × calendar week |
| 12. N-grams | `top_ngrams.png` | Bigrams and trigrams |
| 13. Threat Level | `threat_level.png` | Composite WW3 gauge |

**Key Metrics Computed**:
- `total_articles`: Count of articles in database
- `top_sources`: Article distribution by source
- `mean_words`, `median_words`: Article length statistics
- `avg_sentiment`: Mean TextBlob polarity
- `sentiment_distribution`: {Positive, Neutral, Negative} percentages
- `avg_escalation_score`: Weighted keyword score (0-100)
- `ww3_threat_level`: Composite threat indicator (0-100)
- `dominant_theater`: Most common conflict region
- `actor_mentions`: Entity frequency counts
- `top_bigrams`, `top_trigrams`: Phrase intelligence

#### Vector EDA (`eda_code/vector_eda.py`)

**Semantic Themes Queried**:
```python
THEMES = {
    "WW3 / Escalation": "world war 3 ww3 nuclear escalation...",
    "US-Israel": "us israel united states biden trump...",
    "Israel-Iran": "israel iran attack strike tehran...",
    "Iran-US": "iran us america sanction...",
    "Middle East Conflict": "gaza west bank beirut damascus...",
    "Europe/Russia Context": "russia ukraine putin moscow...",
    "Global Economy & Oil": "oil price brent crude opec..."
}
```

**Output**:
- Theme article counts
- Average semantic distance per theme (lower = more relevant)

#### Report Generator (`eda_code/report_generator.py`)

Generates a dark-themed, responsive HTML dashboard with:
- Threat level banner with color-coded severity
- Metrics cards (article count, sources, sentiment, etc.)
- All 13 charts embedded
- Tables for top sources, actors, theaters, n-grams
- Vector analysis section

---

### 4. RAG API Layer

**File**: `api.py`

#### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | RAG query with source attribution |
| GET | `/report/` | Static dashboard serving |

#### Chat Request/Response

**Request**:
```json
{
  "query": "What is the latest on Iran-Israel tensions?"
}
```

**Response**:
```json
{
  "answer": "Based on the articles...",
  "sources": [
    {
      "id": 42,
      "title": "Israel Strikes Iran...",
      "source": "BBC",
      "url": "https://...",
      "img_link": "https://...",
      "published_at": "2024-01-15",
      "content_snippet": "Full text excerpt...",
      "similarity_distance": 0.23
    }
  ]
}
```

#### RAG Flow

```
User Query → Ollama Embedding → ChromaDB Query → Top 60 Chunks
    ↓
Deduplicate by article_id → Top 20 Unique Articles
    ↓
Fetch Full Article Info from SQLite
    ↓
Build Context from Chunks → LLM Prompt → Response + Sources
```

#### LLM Prompt Template

```python
prompt = f"""You are a geopolitical intelligence analyst assistant.
Use the following retrieved news article context to answer the user's question.
If the answer isn't firmly in the context, do your best based on the context provided,
but acknowledge limitations if necessary. Do not hallucinate facts.

Context:
{context_chunks}

User Question: {query}

Answer:"""
```

---

### 5. Frontend Layer

**Directory**: `ragfrontend/`

A Next.js-based web interface for the RAG system.

#### Components

| Component | Description |
|-----------|-------------|
| `Chatbot.tsx` | Main chat interface with message history |
| `SourceSidebar.tsx` | Displays matching articles with snippets |
| `EdaSidebar.tsx` | Embeds the HTML dashboard |

#### Features
- Real-time chat with the RAG API
- Source article cards with images
- Expandable article snippets
- Embedded EDA dashboard

---

## Output Artifacts

### SQLite Database (`sqlite_databases/news.db`)

Contains all scraped articles with metadata:
- Article content and titles
- Source attribution
- Publication timestamps
- Image URLs
- Duplicate prevention via unique link constraint

### ChromaDB Vector Store (`chromadb/`)

Contains embedded text chunks:
- 500-character chunks with sentence boundaries
- Ollama embeddings (`qwen3-embedding:0.6B`)
- Metadata linking back to SQLite articles

### HTML Dashboard (`eda_output/report.html`)

A self-contained intelligence briefing including:
- All visualizations
- Summary statistics
- Trend analysis
- Threat assessment

---

## API Reference

### POST /api/chat

Submit a natural language query and receive an AI-generated response with sources.

**Request Body**:
```json
{
  "query": "string"
}
```

**Response**:
```json
{
  "answer": "string",
  "sources": [
    {
      "id": "integer",
      "title": "string",
      "source": "string",
      "url": "string | null",
      "img_link": "string | null",
      "published_at": "string | null",
      "content_snippet": "string | null",
      "similarity_distance": "float | null"
    }
  ]
}
```

**Error Responses**:
- `500`: ChromaDB not found (run pipeline first)
- `500`: Ollama connection error

### GET /report/

Serves the static HTML dashboard.

---

## Dependencies

### Python Packages

```
requests>=2.28.0
beautifulsoup4>=4.11.0
feedparser>=6.0.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
chromadb>=0.4.0
ollama>=0.1.0
fastapi>=0.100.0
uvicorn>=0.23.0
textblob>=0.17.0
scikit-learn>=1.2.0
numpy>=1.24.0
```

### External Services

| Service | Purpose | Setup |
|---------|---------|-------|
| Ollama | Local LLM & embeddings | `ollama serve` |
| NewsAPI | Optional news source | API key from newsapi.org |

---

## Troubleshooting

### Common Issues

**1. ChromaDB Connection Error**
```
Error: Could not connect to ChromaDB
```
- Ensure the pipeline has been run at least once to create embeddings
- Check that `chromadb/` directory exists

**2. Ollama Model Not Found**
```
Error: model 'nomic-embed-text' not found
```
- Run: `ollama pull nomic-embed-text`
- Run: `ollama pull qwen3-embedding:0.6B`
- Run: `ollama pull llama3.2:3b`

**3. Empty Dashboard**
- Check that `sqlite_databases/news.db` contains articles
- Verify EDA output directory has PNG files
- Review `pipeline.log` for errors

**4. RSS Feed Timeouts**
- Some feeds may be temporarily unavailable
- The scraper handles errors gracefully and continues

**5. NewsAPI Rate Limits**
- Free tier: 100 requests/day
- Set `NEWSAPI_KEY` environment variable or use default key

### Logging

All pipeline operations are logged to `pipeline.log`:
```
2024-01-15 10:30:00 - PipelineLogger - INFO - Starting execution of 'scrape_all_sources'
2024-01-15 10:30:05 - PipelineLogger - INFO - Finished execution of 'scrape_all_sources' in 5.2341 seconds.
```

---

## License

This project is for educational and research purposes.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Acknowledgments

- **NewsAPI** for API access to global news sources
- **Ollama** for local LLM and embedding capabilities
- **ChromaDB** for vector storage
- **TextBlob** for sentiment analysis
- **scikit-learn** for NLP utilities