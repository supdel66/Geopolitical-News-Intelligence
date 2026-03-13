# Geopolitical News Intelligence Pipeline - Comprehensive Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture & Design](#2-architecture--design)
3. [Data Extraction Layer](#3-data-extraction-layer)
4. [Database Layer](#4-database-layer)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Statistical Analysis Module](#6-statistical-analysis-module)
7. [Vector Database & Semantic Search](#7-vector-database--semantic-search)
8. [RAG Backend API](#8-rag-backend-api)
9. [Frontend Interface](#9-frontend-interface)
10. [Technologies & Libraries](#10-technologies--libraries)
11. [Configuration & Setup](#11-configuration--setup)
12. [Execution Pipeline](#12-execution-pipeline)

---

## 1. Project Overview

### 1.1 Purpose

This project implements a comprehensive **ETL (Extract-Transform-Load) pipeline** for collecting, analyzing, and querying conflict-related news from multiple global sources. It features NLP-powered sentiment analysis, escalation scoring, statistical hypothesis testing, and a **RAG (Retrieval-Augmented Generation) chatbot** interface powered by local LLMs.

### 1.2 Key Objectives

- **Data Collection**: Aggregate geopolitical news from RSS feeds and NewsAPI
- **Intelligent Filtering**: Filter articles using 60+ conflict-related keywords
- **NLP Analysis**: Perform sentiment analysis, entity extraction, and topic modeling
- **Statistical Testing**: Verify hypotheses about news coverage patterns
- **Semantic Search**: Enable natural language queries via vector embeddings
- **Visualization**: Generate comprehensive analytics dashboards

### 1.3 Use Cases

- **Researchers**: Track global conflict dynamics and escalation patterns
- **Journalists**: Monitor geopolitical sentiment and actor mentions
- **Analysts**: Query historical news data using natural language
- **Intelligence**: Assess threat levels based on news coverage intensity

---

## 2. Architecture & Design

### 2.1 High-Level Architecture

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
│  └─ Full-text search + metadata     │  └─ Ollama embeddings                   │
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

### 2.2 Data Flow

```
External Sources → Scraper → SQLite → EDA/Stats Analysis → Vector DB → RAG API → Frontend
```

### 2.3 Design Principles

1. **Separation of Concerns**: Each module handles a specific responsibility
2. **Persistent Storage**: SQLite for structured data, ChromaDB for semantic search
3. **Local LLM**: Privacy-preserving inference using Ollama
4. **Modular Visualization**: Charts generated as PNG files, aggregated in HTML
5. **Type Safety**: Pydantic models for API validation

---

## 3. Data Extraction Layer

### 3.1 File: `scraper/scraper.py`

#### 3.1.1 RSS Feed Collection

**Purpose**: Fetch articles from multiple RSS news feeds.

**Implementation**:
```python
RSS_SOURCES = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
    "Associated_Press": "https://feedx.net/rss/ap.xml",
    "Al_Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "CNN": "http://rss.cnn.com/rss/edition_world.rss",
    "Fox_News": "https://moxie.foxnews.com/google-publisher/world.xml",
    "New_York_Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "ABC_News": "https://abcnews.go.com/abcnews/topstories",
    "Washington_Post": "https://feeds.washingtonpost.com/rss/national",
    "Yahoo_News": "https://news.yahoo.com/rss/topstories"
}
```

**Why RSS?**
- **Standardized Format**: RSS provides a consistent XML structure across sources
- **Real-time Updates**: Feeds are updated frequently by news organizations
- **No Authentication Required**: Most RSS feeds are publicly accessible
- **Lightweight**: Efficient for periodic polling without heavy API usage

#### 3.1.2 Keyword Filtering

**Purpose**: Filter articles for conflict relevance before storage.

**Keyword List** (60+ terms):
```python
KEYWORDS = [
    "war", "iran", "israel", "united states", "us", "missile", "attack",
    "military", "retaliation", "conflict", "gaza", "tehran", "hezbollah",
    "world war 3", "ww3", "nuclear", "escalation", "world war iii",
    "global conflict", "third world war", "thermonuclear", "armageddon",
    "doomsday", "nato article 5", "mutual assured destruction",
    "us israel", "united states israel", "biden israel", "trump israel",
    "american israel", "iron dome", "us aid israel", "f-35 israel",
    "idf us", "military aid israel", "israel lobby", "israel iran",
    "iran attack", "iran strike", "tehran israel", "idf iran",
    "iranian drone", "iranian missile", "mossad iran", "nuclear iran",
    "iran nuclear deal", "jcpoa", "khamenei israel", "iran us",
    "iran america", "iran sanction", "us strike iran", "iran proxy",
    "hezbollah", "hamas", "houthi", "irgc", "strait of hormuz",
    "persian gulf us", "gaza", "west bank", "beirut", "damascus",
    "iraq militia", "red sea attack", "tanker attack", "drone swarm",
    "ballistic missile"
]
```

**Why This Approach?**
- **Precision**: Carefully curated terms minimize false positives
- **Coverage**: Includes synonyms and related phrases (ww3/world war 3)
- **Entity-Based**: Captures specific actors (Israel, Iran, Hamas, Hezbollah)
- **Event-Based**: Captures military actions (missile, strike, retaliation)

#### 3.1.3 Full Article Extraction

**Purpose**: Extract complete article content beyond RSS summaries.

**Implementation**:
Each news source has a custom extractor function:
```python
def extract_full_article_BBC(url):
    """Extract full article content from BBC articles."""
    response = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article")
    img_link = article.find("img").get("src") if article else ""
    paragraphs = article.find_all("p") if article else []
    fulltext = "".join([p.text for p in paragraphs])
    return img_link, fulltext
```

**Why Source-Specific Extractors?**
- **HTML Structure Variations**: Each news site has different DOM structures
- **Reliability**: Custom parsers are more robust than generic selectors
- **Image Extraction**: Captures article images for the frontend
- **Error Handling**: Graceful degradation with try/except blocks

#### 3.1.4 NewsAPI Integration

**Purpose**: Fetch articles via NewsAPI for additional coverage.

**Implementation**:
```python
def fetch_newsapi_news(api_key, queries=None):
    url = "https://newsapi.org/v2/everything"
    for query in queries:
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 10,
            "apiKey": api_key,
        }
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        # Process articles...
```

**Why NewsAPI?**
- **Query-Based Search**: Direct search for conflict-related terms
- **Metadata Rich**: Includes publish dates, authors, and images
- **Pagination Control**: Limit results per query for efficiency

#### 3.1.5 Deduplication

**Purpose**: Prevent duplicate articles from being stored.

**Implementation**:
```python
def is_article_saved(url: str) -> bool:
    """Check if article URL already exists in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM articles WHERE link = ?", (url,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists
```

**Why URL-Based Deduplication?**
- **Deterministic**: URLs are unique identifiers for articles
- **Efficient**: Simple SQL query for existence check
- **Prevents Redundancy**: Avoids storing identical content multiple times

---

## 4. Database Layer

### 4.1 SQLite Database (`database/db.py`)

#### 4.1.1 Schema Design

**Table Structure**:
```sql
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    title TEXT,
    content TEXT,
    published_at TEXT,
    link TEXT UNIQUE,
    scraped_at TEXT,
    img_link TEXT,
    UNIQUE(link)
);
```

**Why This Schema?**
- **INTEGER PRIMARY KEY**: Auto-incrementing ID for efficient joins
- **TEXT Types**: Flexible storage for variable-length content
- **UNIQUE Constraint on Link**: Enforces deduplication at database level
- **Timestamp Fields**: `published_at` for article date, `scraped_at` for collection date

#### 4.1.2 Core Operations

**Initialization**:
```python
def init_db():
    """Create database and articles table if not exists."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS articles (...)''')
    conn.commit()
    conn.close()
```

**Save Articles**:
```python
def save_articles(articles):
    """Insert articles into database, skipping duplicates."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for article in articles:
        cursor.execute('''
            INSERT OR IGNORE INTO articles
            (source, title, content, published_at, link, scraped_at, img_link)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (...))
    conn.commit()
    conn.close()
```

**Why SQLite?**
- **Zero Configuration**: No server setup required
- **File-Based**: Simple backup and portability
- **Adequate Performance**: Handles millions of rows for news data
- **Full ACID Compliance**: Reliable data integrity

### 4.2 Vector Database (`database/vector_db.py`)

#### 4.2.1 Purpose

ChromaDB stores article embeddings for semantic search, enabling natural language queries against the news corpus.

#### 4.2.2 Text Chunking Strategy

**Implementation**:
```python
def chunk_text(text, max_length=500):
    """Chunk text into pieces of strictly <= max_length characters."""
    if not text or not isinstance(text, str):
        return []

    sentences = text.replace('\n', ' ').split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue
        sentence += ". "

        # Force-split excessively long sentences
        if len(sentence) > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            for i in range(0, len(sentence), max_length):
                sub_chunk = sentence[i:i+max_length]
                if sub_chunk:
                    chunks.append(sub_chunk.strip())
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
```

**Why Chunking?**
- **Embedding Limits**: Models have maximum token limits
- **Semantic Granularity**: Smaller chunks enable more precise retrieval
- **Context Preservation**: Sentence-aware splitting maintains meaning
- **Hard Limit Enforcement**: Guarantees no oversized chunks

#### 4.2.3 Embedding Model

**Configuration**:
```python
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name="qwen3-embedding:0.6B",
)
```

**Why Ollama Embeddings?**
- **Local Inference**: No external API calls, preserving privacy
- **Quality**: Modern embedding models capture semantic relationships
- **Cost-Free**: No per-query charges unlike cloud APIs
- **Offline Capable**: Works without internet after model download

#### 4.2.4 Metadata Storage

**Implementation**:
```python
metadatas.append({
    "original_article_id": article_id,
    "source": str(row.get('source', '')),
    "url": str(row.get('link', '')),
    "published_at": str(row.get('published_at', ''))
})
```

**Why Store Metadata?**
- **Link Back to Source**: Enables retrieval of full articles from SQLite
- **Filtering**: Can filter by source, date in vector queries
- **Attribution**: Shows source information in RAG responses

---

## 5. Exploratory Data Analysis (EDA)

### 5.1 File: `eda_code/eda.py`

#### 5.1.1 Reference Data Structures

**Conflict Actors** (15+ entities):
```python
CONFLICT_ACTORS = {
    "Israel":       ["israel", "idf", "netanyahu", "tel aviv", "jerusalem"],
    "Iran":         ["iran", "tehran", "khamenei", "irgc", "iranians"],
    "USA":          ["united states", " us ", "america", "biden", "trump",
                    "pentagon", "washington dc"],
    "Russia":       ["russia", "putin", "moscow", "kremlin", "russian"],
    "Ukraine":      ["ukraine", "zelensky", "kyiv", "kiev", "ukrainian"],
    "China":        ["china", "beijing", "xi jinping", "pla", "chinese army"],
    "Hamas":        ["hamas", "qassam", "sinwar"],
    "Hezbollah":    ["hezbollah", "nasrallah"],
    "Houthis":      ["houthi", "ansarallah", "yemen"],
    "NATO":         ["nato", "alliance members", "article 5"],
    "Saudi Arabia": ["saudi", "riyadh", "mbs"],
    "Palestine":    ["palestine", "palestinian", "west bank", "gaza strip"],
    "North Korea":  ["north korea", "pyongyang", "kim jong"],
    "Syria":        ["syria", "damascus", "syrian"],
    "Lebanon":      ["lebanon", "beirut"],
}
```

**Why Alias-Based Actor Detection?**
- **Comprehensive Coverage**: Captures multiple ways to refer to entities
- **Context Awareness**: Includes leaders, capitals, and military branches
- **Flexibility**: Easy to extend with new aliases

**Conflict Theaters** (5 regions):
```python
CONFLICT_THEATERS = {
    "Middle East":     ["israel", "iran", "gaza", "hezbollah", "hamas", "beirut",
                        "tehran", "houthi", "red sea", "syria", "damascus", "west bank"],
    "Russia/Ukraine":  ["russia", "ukraine", "putin", "zelensky", "nato", "moscow",
                        "kyiv", "kiev", "eastern europe", "donbas", "crimea"],
    "Asia-Pacific":    ["china", "taiwan", "north korea", "south korea", "beijing",
                        "kim jong", "south china sea", "indo-pacific", "japan", "pyongyang"],
    "Global WW3":      ["world war", "ww3", "nuclear", "armageddon", "global conflict",
                        "thermonuclear", "doomsday", "mutual assured destruction", "icbm"],
    "US Policy/Sanctions": ["pentagon", "white house", "us sanction", "trump", "biden",
                            "us military", "congress", "state department", "cia"],
}
```

**Why Theater Classification?**
- **Regional Analysis**: Enables comparative analysis between conflict zones
- **Trend Tracking**: Monitor which regions receive more coverage
- **Hypothesis Testing**: Statistical tests on theater-based sentiment differences

#### 5.1.2 Escalation Scoring

**Purpose**: Quantify the intensity of conflict language in articles.

**Implementation**:
```python
ESCALATION_WEIGHTS = {
    # Extreme escalation (nuclear/WW3)
    "nuclear":                  10,
    "thermonuclear":            10,
    "armageddon":               10,
    "world war 3":              10,
    "ww3":                      10,
    "doomsday":                  9,
    "mutual assured destruction":10,
    "icbm":                      9,

    # Military escalation
    "hypersonic":                8,
    "ballistic missile":         8,
    "chemical weapon":           9,
    "biological weapon":         9,
    "nuclear deal":              5,
    "escalation":                7,
    "invasion":                  6,
    "siege":                     5,
    "airstrike":                 6,
    "bombardment":               6,
    "retaliation":               6,
    "offensive":                 5,
    "strike":                    5,
    "missile":                   5,
    "war":                       5,
    "drone":                     4,
    "attack":                    4,
    "conflict":                  3,

    # De-escalation (negative weights)
    "ceasefire":                -3,
    "peace talks":              -4,
    "diplomacy":                -3,
    "negotiation":              -2,
    "agreement":                -2,
}

def compute_escalation_score(text):
    """Weighted sum of escalation keywords; clamped [0, 100]."""
    if not isinstance(text, str):
        return 0.0
    tl = text.lower()
    score = sum(tl.count(kw) * w for kw, w in ESCALATION_WEIGHTS.items())
    return float(max(0.0, min(score, 100.0)))
```

**Why Weighted Scoring?**
- **Differentiated Impact**: Nuclear threats > diplomatic meetings
- **Negative Weights**: Peace efforts reduce escalation score
- **Clamping**: Prevents unrealistic scores, enables meaningful comparisons
- **Interpretable**: Score of 0-100 is easily understood

#### 5.1.3 Sentiment Analysis

**Purpose**: Measure the emotional tone of conflict news coverage.

**Challenge**: General-purpose sentiment tools (like TextBlob) fail on conflict journalism because:
- "Airstrike" is neutral in general corpora but negative in conflict context
- "Ceasefire" is neutral generally but positive in conflict context

**Solution**: Hybrid sentiment scoring with domain-specific lexicon.

**Implementation**:
```python
_CONFLICT_NEG_LEX = {
    # Kinetic / lethal events
    "killed": -4, "kill": -3, "killing": -3, "kills": -3,
    "massacred": -5, "massacre": -5, "executed": -4, "execution": -4,
    "slaughtered": -4, "genocide": -5, "ethnic cleansing": -5,
    "war crime": -4, "atrocity": -4, "atrocities": -4,
    "bombed": -3, "bombing": -3, "bomb": -2, "airstrike": -3,
    "struck": -3, "strike": -2, "attacked": -3, "attack": -2,
    "wounded": -3, "casualties": -3, "casualty": -3,
    # ... more negative terms
}

_CONFLICT_POS_LEX = {
    # Ceasefires / peace agreements
    "ceasefire": 5, "cease-fire": 5, "truce": 5,
    "peace deal": 5, "peace agreement": 5, "peace": 4,
    # Diplomacy
    "diplomatic": 3, "diplomacy": 3, "mediation": 3,
    "negotiation": 3, "negotiate": 2, "talks": 2,
    # Resolution / recovery
    "released": 3, "freed": 3, "liberation": 3,
    "humanitarian aid": 3, "humanitarian": 2, "relief": 3,
    # ... more positive terms
}

def get_sentiment_polarity(text):
    """
    Hybrid conflict-domain sentiment scorer.
    Primary: domain-specific geopolitical lexicon (70%)
    Secondary: TextBlob general tone (30%)
    Returns polarity in [-1.0, 1.0].
    """
    # Calculate domain lexicon score
    raw_score = 0.0
    for phrase, weight in _CONFLICT_NEG_LEX.items():
        raw_score += text.lower().count(phrase) * weight
    for phrase, weight in _CONFLICT_POS_LEX.items():
        raw_score += text.lower().count(phrase) * weight

    # Normalize by log(word_count)
    log_wc = max(float(np.log(word_count)), 1.0)
    lexicon_score = raw_score / log_wc * 0.3

    # Blend with TextBlob
    tb_pol = float(TextBlob(text[:3000]).sentiment.polarity)
    final = 0.7 * lexicon_score + 0.3 * tb_pol

    return round(max(-1.0, min(1.0, final)), 4)
```

**Why 70/30 Blend?**
- **Domain Expertise**: Conflict lexicon captures context-specific sentiment
- **General Tone**: TextBlob captures writing style and rhetoric
- **Normalization**: Log-scaled by word count prevents length bias

#### 5.1.4 WW3 Threat Level Calculation

**Purpose**: Create a composite indicator of global conflict escalation.

**Implementation**:
```python
def calculate_threat_level(df, stats, all_text):
    total_words = df['word_count'].sum()

    # Normalized components (all 0-100)
    nuclear_density = min((all_text.count("nuclear") / total_words) * 5000, 100)
    avg_esc_norm = min(stats.get('avg_escalation_score', 0) * 1.5, 100)
    ww3_density = min((all_text.count("ww3") + all_text.count("world war 3"))
                      / max(len(df), 1) * 20, 100)
    missile_density = min(all_text.count("missile") / total_words * 3000, 100)
    sentiment_neg = sent_dist.get("Negative", 0)  # percentage

    threat_level = (
        avg_esc_norm * 0.35 +
        nuclear_density * 0.25 +
        sentiment_neg * 0.15 +
        ww3_density * 0.15 +
        missile_density * 0.10
    )

    return min(round(float(threat_level), 1), 100.0)
```

**Why This Formula?**
- **Escalation Score (35%)**: Primary indicator of conflict intensity
- **Nuclear Density (25%)**: Highest-risk keywords
- **Negative Sentiment (15%)**: Emotional tone of coverage
- **WW3 Keywords (15%)**: Direct threat references
- **Missile Mentions (10%)**: Military hardware indicators

#### 5.1.5 Visualizations Generated (13 Charts)

| Chart | Description | Why It Matters |
|-------|-------------|---------------|
| `top_sources.png` | Article count by source | Source bias analysis |
| `top_keywords.png` | Conflict keyword frequency | Topic prevalence |
| `articles_over_time.png` | Publication timeline | Temporal patterns |
| `article_length_dist.png` | Content length distribution | Data quality check |
| `country_actor_mentions.png` | Actor frequency | Key players in news |
| `conflict_theaters.png` | Theater distribution | Regional coverage |
| `actor_cooccurrence_heatmap.png` | Co-mention patterns | Relationship networks |
| `escalation_trend.png` | Escalation over time | Trend monitoring |
| `sentiment_analysis.png` | Sentiment distribution | Emotional tone |
| `source_velocity.png` | Publication frequency | Source activity |
| `top_ngrams.png` | Bigram/trigram analysis | Phrase-level insights |
| `threat_level.png` | WW3 threat gauge | Composite indicator |
| `weekly_heatmap.png` | Day/week patterns | Editorial cycles |

---

## 6. Statistical Analysis Module

### 6.1 File: `eda_code/statistical_analysis.py`

#### 6.1.1 Purpose

Apply rigorous statistical testing to validate or reject hypotheses about the news corpus.

#### 6.1.2 Hypotheses Tested

**H1: Chi-Square Test of Independence**
- **Null Hypothesis (H0)**: China and USA mentions are independent in news articles
- **Alternative (H1)**: China and USA mentions are NOT independent (they co-occur more/less than chance)
- **Statistical Test**: Pearson's Chi-Square
- **Effect Size**: Cramér's V

```python
def test_china_usa_independence(df):
    # Create contingency table
    contingency = pd.crosstab(df["has_china"], df["has_usa"])

    # Chi-Square test
    chi2, p_value, dof, expected = sp_stats.chi2_contingency(contingency)

    # Cramér's V (effect size)
    n = contingency.values.sum()
    phi2 = chi2 / n
    cramers_v = np.sqrt(phi2 / min(k-1, r-1))
```

**Why This Test?**
- Tests whether news frames China and USA as connected topics
- Reveals editorial patterns in geopolitical coverage
- Effect size shows practical significance beyond statistical significance

**H2: One-Way ANOVA**
- **Null Hypothesis (H0)**: Mean sentiment is equal across conflict theaters
- **Alternative (H1)**: At least one theater has significantly different sentiment
- **Groups**: Middle East, Russia/Ukraine, Asia-Pacific
- **Effect Size**: Eta-squared (η²)

```python
def test_sentiment_by_theater(df):
    theater_groups = {
        t: df[df["theater"] == t]["sentiment_polarity"].dropna()
        for t in ["Middle East", "Russia/Ukraine", "Asia-Pacific"]
    }

    # One-way ANOVA
    f_stat, p_value = sp_stats.f_oneway(*theater_groups.values())

    # Eta-squared (effect size)
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in theater_groups.values())
    ss_total = sum((v - grand_mean)**2 for v in all_values)
    eta_sq = ss_between / ss_total
```

**Why This Test?**
- Tests whether different conflict regions receive different emotional framing
- Identifies potential bias in coverage
- Effect size shows proportion of variance explained by theater

#### 6.1.3 Descriptive Statistics

**Computed Metrics**:
- **Skewness**: Asymmetry of distribution
  - Positive: Right-tailed (many low values, few high values)
  - Negative: Left-tailed (many high values, few low values)
- **Excess Kurtosis**: Tail heaviness
  - Positive: Leptokurtic (heavy tails, more outliers)
  - Negative: Platykurtic (thin tails, fewer outliers)

```python
for col in ["word_count", "escalation_score", "sentiment_polarity"]:
    data = df[col].dropna().values
    desc[col] = {
        "skewness": sp_stats.skew(data),
        "excess_kurtosis": sp_stats.kurtosis(data, fisher=True),
        # ... other metrics
    }
```

#### 6.1.4 Zipf's Law Verification

**Purpose**: Validate that the corpus follows natural language power-law distribution.

**Theory**: In natural language, word frequency is inversely proportional to rank:
- f(r) ∝ r^(-1)
- Log-log plot should yield slope ≈ -1

```python
def verify_zipf(df):
    all_words = " ".join(df["combined_text"]).split()
    freq_ctr = Counter(all_words)

    ranks = np.arange(1, top_N + 1)
    freqs = [freq_ctr.most_common(top_N)[i][1] for i in range(top_N)]

    # Power-law fit
    slope, intercept, r, _, _ = sp_stats.linregress(np.log(ranks), np.log(freqs))
```

**Why This Matters**:
- Slope near -1 confirms authentic natural language
- Deviation may indicate:
  - Machine-generated content
  - Heavily templated articles
  - Data quality issues

#### 6.1.5 OLS Regression

**Purpose**: Model what predicts escalation score.

**Model**: `escalation_score ~ word_count + sentiment_polarity`

```python
def run_ols_regression(df):
    X = sm.add_constant(df[["word_count", "sentiment_polarity"]])
    y = df["escalation_score"]

    model = sm.OLS(y, X).fit()

    stats["ols_regression"] = {
        "r_squared": model.rsquared,
        "f_statistic": model.fvalue,
        "f_p_value": model.f_pvalue,
        "word_count_coef": model.params["word_count"],
        "sentiment_coef": model.params["sentiment_polarity"],
    }
```

**Interpretation**:
- **R²**: Proportion of variance in escalation explained by the model
- **Word Count Coefficient**: How much escalation increases per word
- **Sentiment Coefficient**: How much negative sentiment increases escalation

---

## 7. Vector Database & Semantic Search

### 7.1 File: `eda_code/vector_eda.py`

#### 7.1.1 Purpose

Analyze semantic themes using vector embeddings and similarity search.

#### 7.1.2 Theme Definitions

```python
THEMES = {
    "WW3 / Escalation": "world war 3 ww3 nuclear escalation global conflict third thermonuclear armageddon doomsday nato article 5 mutual assured destruction",
    "US-Israel": "us israel united states biden trump american iron dome us aid military idf israel lobby",
    "Israel-Iran": "israel iran attack strike tehran idf iranian drone missile mossad nuclear deal jcpoa khamenei",
    "Iran-US": "iran us america sanction us strike iran proxy hezbollah hamas houthi irgc strait of hormuz persian gulf",
    "Middle East Conflict": "gaza west bank beirut damascus iraq militia red sea tanker attack drone swarm ballistic missile",
    "Europe/Russia Context": "russia ukraine putin moscow kiev nato expansion eastern europe border poland baltic black sea",
    "Global Economy & Oil": "oil price brent crude opec shipping route supply chain inflation global economy market trade route"
}
```

**Why Semantic Themes?**
- **Query-Based Discovery**: Find articles matching conceptual themes, not just keywords
- **Distance Measurement**: Quantifies how closely articles match themes
- **Cross-Topic Analysis**: Articles can match multiple themes

#### 7.1.3 Semantic Search Implementation

```python
def run_vector_eda():
    client = chromadb.PersistentClient(path="chromadb")

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )

    collection = client.get_collection(name="news_articles", embedding_function=ollama_ef)

    for theme_name, theme_query in THEMES.items():
        results = collection.query(
            query_texts=[theme_query],
            n_results=100
        )

        # Filter by distance threshold
        DISTANCE_THRESHOLD = 1.0
        for dist, meta in zip(results['distances'][0], results['metadatas'][0]):
            if dist <= DISTANCE_THRESHOLD:
                article_id = meta.get("original_article_id")
                valid_articles.add(article_id)
```

**Why Distance Threshold?**
- **Quality Control**: Only include semantically relevant matches
- **Threshold = 1.0**: Based on L2 distance for nomic-embed-text
- **Tuning Required**: Different models may need different thresholds

---

## 8. RAG Backend API

### 8.1 File: `api.py`

#### 8.1.1 Purpose

Provide a FastAPI backend for the RAG chatbot, enabling natural language queries against the news corpus.

#### 8.1.2 Architecture

```
User Query → Embed Query → ChromaDB Search → Retrieve Top-20 Chunks
    → Fetch Article Metadata from SQLite → LLM Generation → Response + Sources
```

#### 8.1.3 Query Processing

```python
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query

    # 1. Retrieve similar chunks from ChromaDB
    unique_articles, context_chunks = query_chromadb(query, top_n=20)

    # 2. Fetch article metadata from SQLite
    sources = []
    conn = get_db_connection()
    for item in unique_articles:
        cursor.execute(
            "SELECT id, title, source, link, img_link, published_at, content "
            "FROM articles WHERE id = ?",
            (item["article_id"],)
        )
        row = cursor.fetchone()
        sources.append(ArticleInfo(
            id=row["id"],
            title=row["title"],
            source=row["source"],
            url=row["link"],
            similarity_distance=item["distance"]
        ))
    conn.close()

    # 3. Generate LLM response
    answer = generate_llm_response(query, context_chunks)

    # 4. Return response
    return ChatResponse(answer=answer, sources=sources)
```

#### 8.1.4 LLM Prompt Engineering

```python
def generate_llm_response(query: str, context: List[str]) -> str:
    prompt = f"""You are a geopolitical intelligence analyst assistant.
    Use the following retrieved news article context to answer the user's question.
    If the answer isn't firmly in the context, do your best based on the context provided,
    but acknowledge limitations if necessary. Do not hallucinate facts.

    Context:
    {' --- '.join(context)}

    User Question: {query}

    Answer:"""

    response = ollama.generate(model="llama3.2:3b", prompt=prompt)
    return response.response
```

**Why This Prompt?**
- **Role Definition**: Sets analyst persona for appropriate tone
- **Context Injection**: Provides retrieved documents for grounding
- **Hallucination Prevention**: Explicit instruction to avoid fabrication
- **Chunk Separator**: `---` clearly separates article chunks

#### 8.1.5 CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Why CORS?**
- Frontend (port 3000) and backend (port 8000) are different origins
- Required for browser-based API calls from Next.js

---

## 9. Frontend Interface

### 9.1 Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | Next.js 16 |
| UI Library | React 19 |
| Language | TypeScript |
| Styling | Tailwind CSS 4 |
| HTTP Client | Fetch API |

### 9.2 File: `ragfrontend/app/page.tsx`

**Layout Structure**:
```tsx
<div className="flex h-screen bg-[#0a0a0a] text-white overflow-hidden">
  {/* LEFT: Sources View (25%) */}
  <div className="w-1/4 min-w-[300px]">
    <SourceSidebar sources={currentSources} />
  </div>

  {/* CENTER: Chatbot (50%) */}
  <div className="flex-1 w-1/2">
    <Chatbot
      messages={messages}
      setMessages={setMessages}
      onNewSources={setCurrentSources}
    />
  </div>

  {/* RIGHT: EDA Stats (25%) */}
  <div className="w-1/4 min-w-[250px]">
    <EdaSidebar />
  </div>
</div>
```

**Why Three Columns?**
- **Sources (Left)**: Citation transparency for RAG responses
- **Chat (Center)**: Primary interaction area
- **Stats (Right)**: Dashboard-style analytics overview

### 9.3 File: `ragfrontend/components/Chatbot.tsx`

**State Management**:
```tsx
const [messages, setMessages] = useState<Message[]>([]);
const [input, setInput] = useState('');
const [isLoading, setIsLoading] = useState(false);
```

**API Communication**:
```tsx
const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: userMessage.content }),
});

const data = await response.json();
setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
onNewSources(data.sources);
```

**Why React State?**
- **Message History**: Preserves conversation context
- **Loading State**: Shows visual feedback during API calls
- **Source Sync**: Updates sidebar with relevant articles

---

## 10. Technologies & Libraries

### 10.1 Python Dependencies

| Library | Purpose | Why Selected |
|---------|---------|--------------|
| `requests` | HTTP requests | Simple, reliable HTTP client |
| `beautifulsoup4` | HTML parsing | Robust extraction from diverse sources |
| `feedparser` | RSS feed parsing | Handles RSS/Atom feed standards |
| `pandas` | Data manipulation | Efficient DataFrame operations |
| `matplotlib` | Visualization | Fine-grained chart control |
| `seaborn` | Statistical visualization | Beautiful statistical plots |
| `numpy` | Numerical operations | Array operations for statistics |
| `scipy` | Statistical tests | Chi-Square, ANOVA, etc. |
| `statsmodels` | Regression analysis | OLS with detailed statistics |
| `textblob` | Sentiment analysis | Simple polarity scoring |
| `scikit-learn` | N-gram extraction | CountVectorizer, TfidfVectorizer |
| `chromadb` | Vector database | Persistent semantic search |
| `ollama` | LLM inference | Local model execution |
| `fastapi` | Web framework | Async, type-safe APIs |
| `uvicorn` | ASGI server | Production-ready Python server |

### 10.2 Frontend Dependencies

| Library | Purpose |
|---------|---------|
| `next` | React framework with SSR |
| `react` | UI components |
| `react-dom` | DOM rendering |
| `tailwindcss` | Utility-first CSS |
| `typescript` | Type safety |

### 10.3 External Services

| Service | Purpose | Configuration |
|---------|---------|---------------|
| NewsAPI | News article search | API key (environment variable) |
| Ollama | LLM & embedding inference | Local service (port 11434) |

---

## 11. Configuration & Setup

### 11.1 Environment Variables

```bash
# Optional: Override default NewsAPI key
export NEWSAPI_KEY="your-api-key-here"
```

### 11.2 Ollama Setup

```bash
# Install Ollama (follow instructions at https://ollama.ai)

# Download required models
ollama pull llama3.2:3b          # Chat model
ollama pull nomic-embed-text     # Embedding model (alternative: qwen3-embedding:0.6B)

# Verify installation
ollama list
```

### 11.3 Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### 11.4 Frontend Setup

```bash
cd ragfrontend
npm install
npm run dev  # Development server on port 3000
```

---

## 12. Execution Pipeline

### 12.1 Main Pipeline (`main.py`)

```python
@timer_logger
def main():
    # 1. Initialize SQLite Database
    init_db()

    # 2. Scrape News (RSS + NewsAPI)
    articles = scrape_all_sources(api_key=newsapi_key)

    # 3. Save to SQLite
    save_articles(articles)

    # 4. Load for Analysis
    df = get_all_articles()

    # 5. Run EDA (13 charts)
    sqlite_stats = run_eda(df)

    # 6. Run Statistical Analysis (5 charts)
    statistical_stats = run_statistical_analysis(df)

    # 7. Embed in ChromaDB
    store_articles_in_vector_db(df)

    # 8. Run Vector EDA (2 charts)
    vector_stats = run_vector_eda()

    # 9. Generate HTML Report
    generate_html_report(sqlite_stats, vector_stats, statistical_stats)
```

### 12.2 Output Files

```
eda_output/
├── report.html                  # Interactive dashboard
├── top_sources.png              # Source distribution
├── top_keywords.png             # Keyword frequency
├── articles_over_time.png       # Timeline
├── article_length_dist.png      # Length distribution
├── country_actor_mentions.png   # Actor frequency
├── conflict_theaters.png        # Theater pie chart
├── actor_cooccurrence_heatmap.png  # Co-mentions
├── escalation_trend.png         # Escalation timeline
├── sentiment_analysis.png       # Sentiment charts
├── source_velocity.png         # Publication velocity
├── top_ngrams.png               # N-gram analysis
├── threat_level.png             # WW3 threat gauge
├── weekly_heatmap.png           # Day/week patterns
├── stats_distributions.png     # Descriptive stats
├── stats_correlation_matrices.png  # Correlation/covariance
├── stats_zipf_distribution.png # Zipf's Law
├── stats_ols_regression.png    # OLS regression
├── stats_hypothesis_tests.png  # Chi-Square & ANOVA
├── vector_themes_volume.png    # Semantic theme volume
└── vector_themes_relevance.png # Theme relevance
```

### 12.3 Running the System

```bash
# Terminal 1: Run ETL Pipeline
python main.py

# Terminal 2: Start RAG Backend
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 3: Start Frontend
cd ragfrontend && npm run dev

# Access:
# - Dashboard: eda_output/report.html
# - API: http://localhost:8000/docs
# - Frontend: http://localhost:3000
```

---

## Appendix A: Data Dictionary

### A.1 SQLite Articles Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `source` | TEXT | News source name (BBC, CNN, etc.) |
| `title` | TEXT | Article headline |
| `content` | TEXT | Full article text |
| `published_at` | TEXT | Publication timestamp (ISO 8601) |
| `link` | TEXT | Article URL (unique) |
| `scraped_at` | TEXT | Collection timestamp |
| `img_link` | TEXT | Featured image URL |

### A.2 ChromaDB Metadata

| Field | Description |
|-------|-------------|
| `original_article_id` | SQLite article ID for linking |
| `source` | News source name |
| `url` | Article URL |
| `published_at` | Publication date |

---

## Appendix B: Key Metrics Reference

### B.1 Escalation Score Components

| Keyword | Weight | Category |
|---------|--------|----------|
| nuclear, ww3 | +10 | Extreme escalation |
| icbm, doomsday | +9 | Severe escalation |
| hypersonic, ballistic missile | +8 | High escalation |
| invasion, airstrike | +6 | Moderate escalation |
| missile, strike | +5 | Military action |
| ceasefire | -3 | De-escalation |
| peace talks | -4 | Diplomatic resolution |

### B.2 WW3 Threat Level Formula

```
threat = 0.35 × escalation_score_norm
       + 0.25 × nuclear_density
       + 0.15 × negative_sentiment_pct
       + 0.15 × ww3_keyword_density
       + 0.10 × missile_density
```

---

## Appendix C: Troubleshooting

### C.1 Common Issues

| Issue | Solution |
|-------|----------|
| ChromaDB not found | Run `main.py` to create embeddings |
| Ollama connection error | Ensure Ollama is running: `ollama serve` |
| No articles scraped | Check RSS feed URLs and internet connection |
| Empty vector results | Lower DISTANCE_THRESHOLD in `vector_eda.py` |
| API CORS errors | Verify frontend runs on port 3000 |

### C.2 Performance Tips

- **Batch Processing**: ChromaDB insertion uses batches of 50 to avoid overwhelming Ollama
- **Caching**: EDA outputs are saved as PNGs for fast dashboard loading
- **Index Optimization**: SQLite has indexes on `published_at` and `source`

---

*Documentation generated: 2026-03-12*
*Project: Geopolitical News Intelligence Pipeline*