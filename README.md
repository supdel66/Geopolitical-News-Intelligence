# Geopolitical News Intelligence Pipeline

A sophisticated **ETL (Extract-Transform-Load) pipeline** for collecting, analyzing, and querying conflict-related news from multiple global sources. Features NLP-powered sentiment analysis, escalation scoring, statistical hypothesis testing, and a **RAG (Retrieval-Augmented Generation) chatbot** interface powered by local LLMs.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the ETL Pipeline](#running-the-etl-pipeline)
  - [Starting the RAG Backend](#starting-the-rag-backend)
  - [Starting the Frontend](#starting-the-frontend)
- [Components Deep Dive](#components-deep-dive)
  - [Data Extraction (Scraper)](#data-extraction-scraper)
  - [Database Layer](#database-layer)
  - [EDA & Statistical Analysis](#eda--statistical-analysis)
  - [RAG Backend](#rag-backend)
  - [Frontend Interface](#frontend-interface)
- [Analysis Output](#analysis-output)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This project builds an end-to-end news intelligence system that:

1. **Extracts** geopolitical news from RSS feeds (10 sources) and NewsAPI
2. **Filters** articles using 60+ conflict-related keywords (Iran, Israel, war, nuclear, escalation, etc.)
3. **Stores** articles in SQLite for structured queries and ChromaDB for semantic search
4. **Analyzes** content using NLP techniques (sentiment, entity extraction, topic modeling)
5. **Generates** comprehensive EDA visualizations and statistical reports
6. **Provides** a RAG chatbot interface for querying news using natural language

The system is designed for researchers, journalists, and analysts tracking global conflict dynamics, escalation patterns, and geopolitical sentiment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA EXTRACTION LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  scraper/scraper.py                                                          │
│  ├─ RSS_SOURCES: 10 news feeds (BBC, Al Jazeera, CNN, Fox, NYT, etc.)       │
│  ├─ NewsAPI: Query-based article search                                      │
│  ├─ 60+ conflict keywords for filtering                                      │
│  └─ Deduplication via SQLite link checking                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATABASE LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  database/db.py                     │  database/vector_db.py                  │
│  ├─ SQLite (Structured Storage)    │  ├─ ChromaDB (Semantic Search)          │
│  │  └─ articles table               │  │  └─ news_articles collection        │
│  └─ Full-text search + metadata     │  └─ Ollama embeddings (qwen3)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALYSIS LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  eda_code/eda.py (Exploratory Data Analysis)                                 │
│  ├─ Actor mention extraction (15+ geopolitical entities)                     │
│  ├─ Conflict theater classification (5 theaters)                             │
│  ├─ Escalation scoring (weighted keyword analysis)                          │
│  ├─ Sentiment analysis (hybrid lexicon + TextBlob)                          │
│  └─ WW3 threat level calculation                                             │
│                                                                              │
│  eda_code/statistical_analysis.py (Inferential Statistics)                   │
│  ├─ Chi-Square Test: China-USA mention independence                         │
│  ├─ One-Way ANOVA: Sentiment across theaters                                │
│  ├─ OLS Regression: Escalation ~ word_count + sentiment                    │
│  └─ Zipf's Law verification                                                 │
│                                                                              │
│  eda_code/vector_eda.py (Semantic Theme Analysis)                           │
│  └─ 7 geopolitical themes with semantic distance metrics                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VISUALIZATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  eda_code/report_generator.py                                               │
│  └─ Interactive HTML dashboard with 20+ charts (threat level, actors,      │
│     sentiment, correlation matrices, regression plots)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG BACKEND (FastAPI)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  api.py                                                                      │
│  ├─ POST /api/chat: Query endpoint with semantic search                      │
│  ├─ ChromaDB: Retrieves top-N relevant article chunks                       │
│  └─ Ollama LLM (llama3.2:3b): Generates contextual answers                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Next.js)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ragfrontend/                                                                │
│  ├─ Chatbot.tsx: Query interface with source citations                      │
│  ├─ SourceSidebar.tsx: Article references panel                             │
│  └─ EdaSidebar.tsx: Statistics overview                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### Data Extraction
- **Multi-source scraping**: 10 RSS feeds + NewsAPI for comprehensive coverage
- **Intelligent filtering**: 60+ conflict-specific keywords for relevance
- **Deduplication**: SQLite-based URL tracking prevents duplicates
- **Full article extraction**: BeautifulSoup-based content extraction with source-specific parsers

### Analysis Capabilities
- **Actor Mention Tracking**: Frequency analysis for 15+ geopolitical entities (Israel, Iran, US, Russia, China, Hamas, Hezbollah, etc.)
- **Conflict Theater Classification**: Articles categorized into 5 theaters (Middle East, Russia/Ukraine, Asia-Pacific, Global WW3, US Policy)
- **Escalation Scoring**: Weighted keyword analysis with positive weights for threats (nuclear: +10, missile: +5) and negative weights for diplomacy (ceasefire: -3, peace talks: -4)
- **Hybrid Sentiment Analysis**: Custom conflict-domain lexicon combined with TextBlob for domain-accurate sentiment
- **N-gram Analysis**: Bigram and trigram extraction for phrase-level insights
- **WW3 Threat Level**: Composite score combining escalation, nuclear mentions, and global conflict indicators

### Statistical Testing
- **Chi-Square Test**: Verifies China-USA mentions are not independent in news coverage
- **One-Way ANOVA**: Tests if sentiment differs significantly across conflict theaters
- **OLS Regression**: Models escalation score as function of word count and sentiment
- **Zipf's Law Verification**: Validates word frequency follows power law distribution

### RAG Chatbot
- **Semantic Search**: ChromaDB vector database with Ollama embeddings
- **Local LLM**: llama3.2:3b for privacy-preserving inference
- **Source Citations**: Every response includes links to source articles
- **Context-aware**: Retrieves top-20 most relevant article chunks

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.14, FastAPI |
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS 4 |
| **Databases** | SQLite (structured), ChromaDB (vector) |
| **NLP/ML** | TextBlob, scikit-learn, spaCy |
| **LLM/Embeddings** | Ollama (llama3.2:3b, qwen3-embedding:0.6B, mnomic-embed-text) |
| **Visualization** | Matplotlib, Seaborn |
| **Scraping** | BeautifulSoup, feedparser, requests |

---

## Project Structure

```
Elective-Project-/
├── main.py                      # Main pipeline orchestrator
├── api.py                       # FastAPI RAG backend
├── utils.py                     # Logging and timing decorators
├── requirements.txt             # Python dependencies
├── pipeline.log                 # Execution logs
│
├── database/
│   ├── db.py                    # SQLite operations (init, save, fetch)
│   └── vector_db.py             # ChromaDB operations (embed, store)
│
├── scraper/
│   └── scraper.py               # RSS + NewsAPI scraping with filtering
│
├── eda_code/
│   ├── eda.py                   # Exploratory data analysis (13 charts)
│   ├── statistical_analysis.py  # Hypothesis testing (5 charts)
│   ├── vector_eda.py            # Semantic theme analysis (2 charts)
│   └── report_generator.py      # HTML dashboard generator
│
├── ragfrontend/                 # Next.js frontend
│   ├── app/
│   │   ├── page.tsx             # Main page layout
│   │   ├── layout.tsx           # Root layout
│   │   └── globals.css          # Global styles
│   └── components/
│       ├── Chatbot.tsx          # Chat interface
│       ├── SourceSidebar.tsx    # Article sources panel
│       └── EdaSidebar.tsx       # Statistics panel
│
├── data/
│   └── raw/
│       └── news_articles.csv    # Raw scraped articles backup
│
├── sqlite_databases/
│   └── news.db                  # SQLite database
│
├── chromadb/                    # ChromaDB vector store
│   └── chroma.sqlite3
│
└── eda_output/                  # Generated outputs
    ├── report.html              # Interactive HTML dashboard
    ├── threat_level.png         # WW3 threat visualization
    ├── top_sources.png          # Source distribution
    └── ... (18 other charts)
```

---

## Installation

### Prerequisites
- Python 3.10+ (tested on 3.14)
- Node.js 18+ (for frontend)
- Ollama (for LLM inference)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/elective-project.git
cd elective-project
```

### 2. Install Python Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install Ollama and Download Models
```bash
# Install Ollama (follow instructions at https://ollama.ai)
ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6B
# Alternative embedding model:
ollama pull nomic-embed-text
```

### 4. Install Frontend Dependencies
```bash
cd ragfrontend
npm install
cd ..
```

### 5. Configure Environment (Optional)
```bash
# Set NewsAPI key (optional - default key provided)
export NEWSAPI_KEY="your-api-key-here"
```

---

## Usage

### Running the ETL Pipeline

```bash
python main.py
```

This executes the complete pipeline:

1. **Initialize SQLite database** - Creates `sqlite_databases/news.db`
2. **Scrape RSS feeds** - Fetches from 10 configured sources
3. **Scrape NewsAPI** - Queries 12 conflict-related terms
4. **Save articles** - Deduplicates and stores in SQLite
5. **Run EDA** - Generates 13 exploratory charts
6. **Run Statistical Analysis** - Performs hypothesis tests (5 charts)
7. **Embed to ChromaDB** - Creates semantic search index
8. **Run Vector EDA** - Semantic theme analysis (2 charts)
9. **Generate Report** - Builds `eda_output/report.html`

### Starting the RAG Backend

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Starting the Frontend

```bash
cd ragfrontend
npm run dev
```

Access the chatbot at `http://localhost:3000`

---

## Components Deep Dive

### Data Extraction (Scraper)

The scraper module (`scraper/scraper.py`) implements:

**RSS Sources (10 feeds)**:
- BBC, Associated Press, Al Jazeera, Guardian
- CNN, Fox News, NYT, ABC News
- Washington Post, Yahoo News

**Filtering Logic**:
- 60+ conflict keywords including: `war`, `iran`, `israel`, `nuclear`, `hezbollah`, `hamas`, `missile`, `escalation`, `ww3`, `armageddon`
- Articles must match at least one keyword in title or content
- Source-specific HTML parsers for full article extraction

**Deduplication**:
```python
def is_article_saved(url: str) -> bool:
    # Checks if URL already exists in SQLite
    cursor.execute("SELECT 1 FROM articles WHERE link = ?", (url,))
    return cursor.fetchone() is not None
```

### Database Layer

**SQLite (`database/db.py`)**:
```sql
CREATE TABLE articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    title TEXT,
    content TEXT,
    published_at TEXT,
    link TEXT UNIQUE,
    scraped_at TEXT,
    img_link TEXT
);
```

**ChromaDB (`database/vector_db.py`)**:
- Embedding model: `qwen3-embedding:0.6B`
- Chunk size: 500 characters with sentence-aware splitting
- Metadata: `original_article_id`, `source`, `url`, `published_at`

### EDA & Statistical Analysis

**Key Analysis Functions**:

```python
def compute_escalation_score(text):
    """Weighted sum of escalation keywords; clamped [0, 100]."""
    # Nuclear/WW3 keywords: +10
    # Missile/strike keywords: +5
    # Ceasefire/diplomacy: -3 to -4

def classify_theater(text):
    """Assign dominant conflict theater."""
    # Middle East: israel, iran, gaza, hezbollah...
    # Russia/Ukraine: russia, ukraine, putin, zelensky...
    # Asia-Pacific: china, taiwan, north korea...
```

**Conflict-Domain Sentiment Lexicon**:
- Custom positive/negative word lists for conflict journalism
- Combined with TextBlob for hybrid sentiment scoring
- Handles domain-specific terms like `airstrike` (negative), `ceasefire` (positive)

### RAG Backend

**Query Flow**:
```
User Query → Embed Query → ChromaDB Search → Retrieve Top-20 Chunks
    → Fetch Article Metadata from SQLite → LLM Generation → Response + Sources
```

**LLM Prompt Template**:
```
You are a geopolitical intelligence analyst assistant.
Use the following retrieved news article context to answer the user's question.
If the answer isn't firmly in the context, acknowledge limitations.
Do not hallucinate facts.

Context: {article_chunks_joined}
User Question: {query}
```

### Frontend Interface

**Three-Column Layout**:
- **Left (25%)**: Source sidebar showing article citations
- **Center (50%)**: Chat interface with message history
- **Right (25%)**: EDA statistics panel

**Features**:
- Real-time chat with loading indicators
- Source article previews with similarity scores
- Responsive dark theme UI

---

## Analysis Output

The pipeline generates 20+ visualizations in `eda_output/`:

### Exploratory Analysis (13 charts)
| Chart | Description |
|-------|-------------|
| `top_sources.png` | Article count by news source |
| `top_keywords.png` | Most frequent conflict keywords |
| `articles_over_time.png` | Timeline of article publication |
| `article_length_dist.png` | Content length distribution |
| `country_actor_mentions.png` | Geopolitical actor frequency |
| `conflict_theaters.png` | Theater classification breakdown |
| `actor_cooccurrence_heatmap.png` | Actor co-mention patterns |
| `escalation_trend.png` | Escalation score over time |
| `sentiment_analysis.png` | Sentiment distribution |
| `source_velocity.png` | Publishing frequency by source |
| `top_ngrams.png` | Bigram/trigram analysis |
| `threat_level.png` | WW3 threat composite score |
| `weekly_heatmap.png` | Publishing patterns by day/hour |

### Statistical Analysis (5 charts)
| Chart | Description |
|-------|-------------|
| `stats_distributions.png` | Skewness and kurtosis |
| `stats_correlation_matrices.png` | Pearson correlation & covariance |
| `stats_zipf_distribution.png` | Zipf's Law verification |
| `stats_ols_regression.png` | Escalation regression model |
| `stats_hypothesis_tests.png` | Chi-Square & ANOVA results |

### Vector Analysis (2 charts)
| Chart | Description |
|-------|-------------|
| `vector_theme_counts.png` | Semantic theme distribution |
| `vector_theme_distances.png` | Theme coherence analysis |

### HTML Report
`report.html` combines all charts into an interactive dark-themed dashboard.

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEWSAPI_KEY` | NewsAPI API key | Built-in key (rate-limited) |

### Configurable Constants

**RSS Sources** (`scraper/scraper.py`):
```python
RSS_SOURCES = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
    "Al_Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    # ... add more sources
}
```

**Conflict Keywords** (`scraper/scraper.py`):
```python
KEYWORDS = [
    "war", "iran", "israel", "nuclear", "missile",
    # ... add domain-specific keywords
]
```

**LLM Models** (`api.py`):
```python
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"  # or "qwen3-embedding:0.6B"
```

---

## API Reference

### POST /api/chat

Query the news database using natural language.

**Request**:
```json
{
  "query": "What are the latest developments in the Middle East?"
}
```

**Response**:
```json
{
  "answer": "Based on the retrieved articles, recent developments include...",
  "sources": [
    {
      "id": 42,
      "title": "Israel-Iran Escalation Continues",
      "source": "BBC",
      "url": "https://...",
      "img_link": "https://...",
      "published_at": "2024-01-15T10:30:00Z",
      "content_snippet": "Iran has launched...",
      "similarity_distance": 0.23
    }
  ]
}
```

### GET /report

Access the EDA HTML dashboard.

---

## Dependencies

### Python (requirements.txt)
```
requests          # HTTP requests
beautifulsoup4    # HTML parsing
feedparser        # RSS feed parsing
pandas            # Data manipulation
matplotlib        # Visualization
seaborn           # Statistical visualization
chromadb          # Vector database
ollama            # LLM interface
fastapi           # Web framework
uvicorn           # ASGI server
```

### Additional Requirements
```bash
# spaCy NLP model (optional - for advanced NLP)
python -m spacy download en_core_web_sm
```

### Frontend (package.json)
```json
{
  "next": "16.1.6",
  "react": "19.2.3",
  "react-dom": "19.2.3",
  "tailwindcss": "^4",
  "typescript": "^5"
}
```

---

## License

This project is for educational purposes. News content scraped from RSS feeds remains property of respective publishers.

---

## Acknowledgments

- News sources: BBC, Al Jazeera, CNN, Fox News, NYT, Guardian, AP, ABC, Washington Post, Yahoo News
- LLM inference: [Ollama](https://ollama.ai)
- Vector database: [ChromaDB](https://www.trychroma.com)
- Frontend framework: [Next.js](https://nextjs.org)