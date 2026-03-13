# Geopolitical News ETL Pipeline Architecture

This document serves to explain the Object-Oriented (OOP) Refactoring pattern, Iterator optimizations, and Scheduling capabilities deployed in the newly optimized ETL (Extract, Transform, Load) Pipeline.

## 1. Object-Oriented Programming (OOP) Classes
The codebase now uses a modular OOP design across its core logic barriers, replacing standard procedural functions with structured, portable classes:

- **`NewsScraper`** (`scraper/scraper.py`):
    - Replaces scattered procedural scraping definitions.
    - Initialized optionally with an API Key and instantiated with a linkage to the Database instance.
    - Consolidates functionality across dozens of distinct sources (e.g., RSS Feeds, NewsAPI queries).
- **`SQLiteDatabase`** (`database/db.py`):
    - Replaces procedural SQLite interaction points.
    - Sets up the database automatically on initialization. Controls schema addition transparently.
    - Designed specifically to consume Python Generators dynamically via parameter injection instead of allocating lists.
- **`VectorDatabase`** (`database/vector_db.py`):
    - Abstract layer over `chromadb`, interacting with the native `OllamaEmbeddingFunction`.
    - Implements text-chunking as a native generator method before upserting embeddings.
- **`ETLPipeline`** (`main.py`):
    - The top-level orchestration class controlling Extract (`NewsScraper`), Load (`SQLiteDatabase`, `VectorDatabase`), and Transform & Statistics outputs (`eda_code`).

## 2. Iterators & Generators for Memory Optimization

Instead of holding tens of thousands of articles or sub-chunks in RAM simultaneously (an approach inherently prone to memory-leaks and Out-Of-Memory/OOM kills), the pipeline strictly abides by Python Generator workflows:

### A. Scraping Generator (`yield` articles)
In `scraper.py`, as each source processes the incoming feed, it extracts the article and immediately yields an object using `yield {article_data}`. The scraper never holds a list of compiled articles in memory globally.

### B. SQLite Feed Generator (`yield` DB chunks)
In `db.py`, when transferring data to downstream Vector Storage processes, utilizing Pandas or direct `fetchall()` queries forces scaling issues. The `SQLiteDatabase` provides `get_all_articles_iterator(batch_size=1000)`. This uses SQLite's `.fetchmany()` API to dynamically spool thousands of articles into smaller, iterable blocks using `yield`.

### C. Vector Chunking Generator (`yield` text chunks)
In `vector_db.py`, large articles are split sequentially. Instead of creating a massive string list (`chunks = []`), the class uses `chunk_text_generator()` to mathematically subset the string and immediately `yield` it out to the ChromaDB uploader. This significantly prevents Ollama ingestion errors locally over long runtimes.

## 3. How to Run the Application

The `main.py` orchestrator supports `argparse` parameters. The integrated `schedule` toolkit allows continuous autonomy without reliance on external CRON or linux schedulers.

### Run Exactly ONCE:
This creates the objects, runs the extract/load phase, parses the stats, builds the `eda_output/report.html`, and completely exits python.
```bash
python main.py
```

### Run ENDLESSLY on a Schedule (Hourly):
This executes exactly as above immediately, but does not kill the terminal upon finishing. It instead holds indefinitely, re-launching uniformly every `60 minutes`.
```bash
python main.py --hourly
```
