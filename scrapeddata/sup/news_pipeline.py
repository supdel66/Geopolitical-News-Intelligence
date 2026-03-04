"""
=============================================================================
 NEWS SCRAPING → ETL PIPELINE → NLP EDA
 Topic: WW3 / US-Israel-Iran Geopolitical Risk Analyzer
 Covers ENCT 325: Ch2 (APIs/Scraping), Ch3 (Wrangling + Lazy Eval),
                   Ch4 (EDA + Stats), Ch5 (Visualization), Ch6 (ETL + Automation)
=============================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import json
import time
import logging
import hashlib
import sqlite3
import datetime
import itertools
from pathlib import Path
from typing import Generator, Iterator
from collections import Counter

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONFIG & LOGGING  (Ch 1.5 / 6.4)
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
log = logging.getLogger("NewsPipeline")

DB_PATH   = Path("news_pipeline.db")
OUT_DIR   = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 50          # lazy evaluation chunk size

# ── Geopolitical keyword groups ───────────────────────────────────────────────
KEYWORD_GROUPS = {
    "WW3 / Escalation": [
        "world war 3", "ww3", "nuclear", "escalation", "world war iii",
        "global conflict", "third world war", "thermonuclear", "armageddon",
        "doomsday", "nato article 5", "mutual assured destruction",
    ],
    "US-Israel": [
        "us israel", "united states israel", "biden israel", "trump israel",
        "american israel", "iron dome", "us aid israel", "f-35 israel",
        "idf us", "military aid israel", "israel lobby",
    ],
    "Israel-Iran": [
        "israel iran", "iran attack", "iran strike", "tehran israel",
        "idf iran", "iranian drone", "iranian missile", "mossad iran",
        "nuclear iran", "iran nuclear deal", "jcpoa", "khamenei israel",
    ],
    "Iran-US": [
        "iran us", "iran america", "iran sanction", "us strike iran",
        "iran proxy", "hezbollah", "hamas", "houthi", "irgc",
        "strait of hormuz", "persian gulf us",
    ],
    "Middle East Conflict": [
        "gaza", "west bank", "beirut", "damascus", "iraq militia",
        "red sea attack", "tanker attack", "drone swarm", "ballistic missile",
    ],
}

ALL_KEYWORDS = list(itertools.chain.from_iterable(KEYWORD_GROUPS.values()))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TIMER DECORATOR  (Ch 1.3 / 1.2)
# ══════════════════════════════════════════════════════════════════════════════

def timer(func):
    """Decorator that logs execution time of any pipeline stage."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        log.info(f"▶ Starting  : {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info(f"✔ Completed : {func.__name__} in {elapsed:.2f}s")
        return result
    return wrapper

def retry(max_attempts=3, delay=2):
    """Decorator: retry on network failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    log.warning(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            log.error(f"{func.__name__} failed after {max_attempts} attempts.")
            return []
        return wrapper
    return decorator

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA INGESTION / SCRAPING  (Ch 2.3 / 2.4)
# ══════════════════════════════════════════════════════════════════════════════

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}

# ── 2a. NewsAPI (REST API) ─────────────────────────────────────────────────────
@timer
@retry(max_attempts=3, delay=1)
def fetch_newsapi(query: str, api_key: str, page_size: int = 50) -> list[dict]:
    """Fetch articles from NewsAPI /v2/everything."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q"        : query,
        "sortBy"   : "publishedAt",
        "language" : "en",
        "pageSize" : page_size,
        "apiKey"   : api_key,
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("articles", [])
    log.info(f"  NewsAPI → {len(articles)} articles for '{query}'")
    return articles

# ── 2b. RSS Scraper (BeautifulSoup)  ──────────────────────────────────────────
@timer
@retry(max_attempts=3, delay=1)
def scrape_rss_feed(url: str, source_name: str) -> list[dict]:
    """Scrape an RSS feed and return normalised article dicts."""
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "xml")
    items = soup.find_all("item")
    articles = []
    for item in items:
        articles.append({
            "source"      : source_name,
            "title"       : (item.find("title") or {}).get_text(strip=True),
            "description" : (item.find("description") or {}).get_text(strip=True)[:500],
            "url"         : (item.find("link") or {}).get_text(strip=True),
            "publishedAt" : (item.find("pubDate") or {}).get_text(strip=True),
            "content"     : (item.find("description") or {}).get_text(strip=True),
        })
    log.info(f"  RSS ({source_name}) → {len(articles)} items")
    return articles

# ── 2c. GDELT mock / fallback dataset ─────────────────────────────────────────
def generate_mock_articles(n: int = 300) -> list[dict]:
    """
    Generate realistic mock articles for offline/demo use.
    Simulates a real pipeline without burning API quota.
    """
    import random, string

    np.random.seed(42)
    random.seed(42)

    templates = [
        ("US threatens new sanctions against Iran over nuclear enrichment programme",   "Reuters"),
        ("Israel launches airstrikes on Iranian-backed militia positions in Syria",      "AP"),
        ("Iran vows retaliation after Israeli strike kills IRGC commanders",            "Al Jazeera"),
        ("Hezbollah fires barrage of rockets into northern Israel amid ceasefire talks","BBC"),
        ("US deploys additional carrier strike group to Persian Gulf amid tensions",     "CNN"),
        ("Houthi rebels attack commercial vessels in Red Sea; US Navy intercepts drones","Reuters"),
        ("Netanyahu warns Iran: any nuclear breakout will trigger military response",    "Times of Israel"),
        ("Tehran holds military exercise near Strait of Hormuz amid US pressure",       "AFP"),
        ("G7 condemns Iran's missile program; new European sanctions expected",          "Guardian"),
        ("Analysts warn Middle East could tip into regional war within weeks",           "FT"),
        ("Hamas announces ceasefire deal; Israel yet to confirm acceptance",             "Al Jazeera"),
        ("US backs two-state solution; Netanyahu government signals opposition",         "Washington Post"),
        ("Russia and China block UN resolution condemning Iran nuclear activities",      "Reuters"),
        ("IAEA inspectors denied access to Iranian underground facility",                "BBC"),
        ("Israel's intelligence chief flies to Washington for emergency consultations",  "Haaretz"),
        ("Oil prices surge on fears of wider Middle East escalation",                   "Bloomberg"),
        ("World War 3 fears grow as US and Iran exchange warnings over strait",         "Independent"),
        ("Pentagon confirms deployment of B-52 bombers to Diego Garcia base",           "AP"),
        ("Turkey calls for emergency UN Security Council session on Gaza conflict",     "TRT World"),
        ("Iran state media confirms enrichment reaching 84% purity",                    "Reuters"),
        ("Jordan and Egypt warn against regional spillover from Gaza fighting",         "AFP"),
        ("Saudi Arabia mediates back-channel talks between Iran and Israel: sources",   "Reuters"),
        ("French FM: 'We are closer to a regional war than at any point since 2006'",  "Le Monde"),
        ("US Congress debates authorisation for military force against Iran proxies",   "Politico"),
        ("Cyber attacks on Israeli infrastructure traced to Iranian hacker group",      "CyberScoop"),
        ("Iron Dome intercepts 90% of rockets fired during overnight barrage",          "i24 News"),
        ("Iran nuclear deal talks collapse; diplomats cite lack of trust",              "Guardian"),
        ("CENTCOM: US forces have neutralised 400 drone/missile threats this month",   "DoD"),
        ("UN Secretary-General warns: 'Middle East is on the edge of the abyss'",      "UN News"),
        ("Israel's cabinet approves ground offensive into Rafah despite US objections", "NYT"),
    ]

    # noise articles (unrelated)
    noise = [
        ("Apple reports record quarterly earnings amid AI product launches",            "CNBC"),
        ("Champions League semi-final results: Madrid vs Bayern ends 2-2",             "ESPN"),
        ("Scientists discover new exoplanet in habitable zone",                        "Nature"),
        ("Federal Reserve holds rates steady; hints at July cut",                      "Bloomberg"),
        ("Tour de France route unveiled for upcoming season",                           "Cycling Weekly"),
    ]

    articles = []
    today = datetime.datetime.utcnow()
    for i in range(n):
        if random.random() < 0.82:
            tmpl = random.choice(templates)
        else:
            tmpl = random.choice(noise)
        title, source = tmpl
        # add slight variation
        suffix = random.choice(["", " — report", " (update)", " sources say", ""])
        days_ago = random.randint(0, 30)
        pub_date = (today - datetime.timedelta(days=days_ago)).isoformat()
        articles.append({
            "source"      : source,
            "title"       : title + suffix,
            "description" : title + ". " + "Full story developing. " * random.randint(1, 4),
            "url"         : f"https://example.com/article/{i:04d}",
            "publishedAt" : pub_date,
            "content"     : title + ". " + "Additional reporting. " * random.randint(2, 8),
        })
    random.shuffle(articles)
    log.info(f"  Mock generator → {len(articles)} articles")
    return articles

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LAZY EVALUATION / GENERATOR PIPELINE  (Ch 2.5 / 3.4 / 3.5)
# ══════════════════════════════════════════════════════════════════════════════

def article_stream(raw_articles: list[dict]) -> Generator[dict, None, None]:
    """Generator: yield one article at a time — memory-efficient streaming."""
    for article in raw_articles:
        yield article

def chunked(iterable, size: int) -> Generator[list, None, None]:
    """Yield successive chunks of `size` from any iterable (lazy)."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk            # yield the leftover partial chunk

def normalize_article(raw: dict) -> dict | None:
    """
    Transformation stage: clean + normalise a single article.
    Returns None if the article should be dropped (deduplication / missing data).
    """
    title = (raw.get("title") or "").strip()
    desc  = (raw.get("description") or "").strip()
    if not title or title.lower() in ("[removed]", "none", ""):
        return None            # drop empty / removed articles

    # Build combined text blob
    text = f"{title}. {desc}"
    text = text.replace("\n", " ").replace("\r", " ")

    # Deduplicate by content hash
    content_hash = hashlib.md5(title.lower().encode()).hexdigest()

    # Parse date
    raw_date = raw.get("publishedAt", "")
    try:
        pub_dt = pd.to_datetime(raw_date, utc=True, errors="coerce")
    except Exception:
        pub_dt = pd.NaT

    source = raw.get("source") or ""
    if isinstance(source, dict):
        source = source.get("name", "Unknown")

    return {
        "title"       : title,
        "description" : desc[:400],
        "url"         : raw.get("url", ""),
        "source"      : source,
        "published_at": pub_dt,
        "text"        : text,
        "content_hash": content_hash,
    }

@timer
def etl_transform_chunked(raw_articles: list[dict]) -> pd.DataFrame:
    """
    ETL Transform stage using LAZY EVALUATION + CHUNKING.
    Processes articles in chunks — demonstrates Ch 2.5 & 3.4 concepts.
    """
    log.info(f"  ETL: processing {len(raw_articles)} articles in chunks of {CHUNK_SIZE}")
    seen_hashes: set[str] = set()
    clean_records: list[dict] = []

    stream = article_stream(raw_articles)           # generator (lazy)
    for i, chunk in enumerate(chunked(stream, CHUNK_SIZE), 1):
        chunk_clean = []
        for raw in chunk:
            norm = normalize_article(raw)
            if norm is None:
                continue
            if norm["content_hash"] in seen_hashes:
                continue                           # deduplicate
            seen_hashes.add(norm["content_hash"])
            chunk_clean.append(norm)
        clean_records.extend(chunk_clean)
        log.info(f"    chunk {i:>3} → kept {len(chunk_clean)}/{len(chunk)}")

    df = pd.DataFrame(clean_records)
    log.info(f"  ETL: {len(df)} unique articles retained")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NLP SCORING  (Ch 4 / 3.3)
# ══════════════════════════════════════════════════════════════════════════════

def keyword_score(text: str, keywords: list[str]) -> float:
    """Count keyword hits (case-insensitive) normalised by text length."""
    text_l = text.lower()
    hits = sum(1 for kw in keywords if kw in text_l)
    return hits

def group_scores(text: str) -> dict[str, float]:
    """Return hit count for each geopolitical keyword group."""
    return {grp: keyword_score(text, kws) for grp, kws in KEYWORD_GROUPS.items()}

@timer
def tfidf_cosine_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    TF-IDF + cosine similarity to reference query strings.
    Gives a continuous relevance score per article.
    """
    reference_queries = {
        "ww3_risk"    : "world war nuclear escalation global conflict nato military strike",
        "us_israel"   : "united states israel military aid idf american policy",
        "israel_iran" : "israel iran nuclear strike missile drone irgc attack tehran",
        "us_iran"     : "united states iran sanction proxy houthi hezbollah persian gulf",
        "mideast"     : "middle east conflict gaza beirut red sea hamas hezbollah missile",
    }
    texts = df["text"].fillna("").tolist()
    ref_texts = list(reference_queries.values())
    ref_keys  = list(reference_queries.keys())

    # Fit TF-IDF on combined corpus
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        stop_words="english",
        min_df=1,
    )
    all_texts = texts + ref_texts
    tfidf_matrix = tfidf.fit_transform(all_texts)

    article_vecs = tfidf_matrix[:len(texts)]
    ref_vecs     = tfidf_matrix[len(texts):]

    # Cosine similarity → each article vs each reference
    sims = cosine_similarity(article_vecs, ref_vecs)   # (n_articles × n_refs)
    for j, key in enumerate(ref_keys):
        df[f"sim_{key}"] = sims[:, j]

    # Aggregate relevance score (max across all reference topics)
    sim_cols = [f"sim_{k}" for k in ref_keys]
    df["geo_relevance_score"] = df[sim_cols].max(axis=1)

    # Keyword group scores
    for grp in KEYWORD_GROUPS:
        df[f"kw_{grp.lower().replace(' ','_').replace('/','_').replace('-','_')}"] = (
            df["text"].apply(lambda t: keyword_score(t, KEYWORD_GROUPS[grp]))
        )

    df["total_keyword_hits"] = df[[c for c in df.columns if c.startswith("kw_")]].sum(axis=1)

    # Final composite score (weighted blend)
    df["composite_score"] = (
        0.55 * df["geo_relevance_score"] +
        0.45 * (df["total_keyword_hits"] / (df["total_keyword_hits"].max() + 1e-9))
    )

    # Risk tier classification
    df["risk_tier"] = pd.cut(
        df["composite_score"],
        bins=[-np.inf, 0.04, 0.12, 0.22, np.inf],
        labels=["Low", "Moderate", "High", "Critical"],
    )

    log.info(
        f"  NLP scoring complete | "
        f"Critical: {(df['risk_tier']=='Critical').sum()} | "
        f"High: {(df['risk_tier']=='High').sum()}"
    )
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STORAGE  (Ch 2.2 / 6.5)
# ══════════════════════════════════════════════════════════════════════════════

@timer
def load_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH):
    """Load the processed DataFrame into SQLite (Ch 2.2 — relational DB)."""
    con = sqlite3.connect(db_path)
    df_save = df.copy()
    df_save["published_at"] = df_save["published_at"].astype(str)
    df_save["risk_tier"]    = df_save["risk_tier"].astype(str)
    df_save.to_sql("articles", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_score ON articles(composite_score DESC)")
    con.commit()
    con.close()
    log.info(f"  SQLite: {len(df)} rows → {db_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EDA & VISUALIZATION  (Ch 4 / 5)
# ══════════════════════════════════════════════════════════════════════════════

@timer
def run_eda(df: pd.DataFrame):
    """Full EDA suite with statistical summaries and rich visualisations."""

    # ── 6.1 Descriptive stats ─────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(" EXPLORATORY DATA ANALYSIS — Geopolitical News Pipeline")
    print("═"*60)
    print(f"\n Dataset shape     : {df.shape}")
    print(f" Total articles    : {len(df)}")
    print(f" Unique sources    : {df['source'].nunique()}")
    print(f" Date range        : {df['published_at'].min()} → {df['published_at'].max()}")
    print(f"\n Risk tier distribution:")
    print(df["risk_tier"].value_counts().to_string())
    print(f"\n Composite score stats:")
    print(df["composite_score"].describe().round(4).to_string())

    # Skewness and kurtosis (Ch 4.1)
    sk = stats.skew(df["composite_score"].dropna())
    ku = stats.kurtosis(df["composite_score"].dropna())
    print(f"\n Skewness : {sk:.4f}   (>0 = right-skewed, most articles low risk)")
    print(f" Kurtosis : {ku:.4f}   (>3 = heavy tails / extreme articles)")

    # Correlation matrix
    sim_cols = [c for c in df.columns if c.startswith("sim_")]
    corr = df[["composite_score", "total_keyword_hits", "geo_relevance_score"] + sim_cols].corr()
    print(f"\n Correlation matrix (top feature pairs):")
    print(corr.round(3).to_string())

    # ── 6.2  MASTER FIGURE (10 subplots) ─────────────────────────────────────
    sns.set_theme(style="darkgrid", palette="deep", font_scale=0.95)
    fig = plt.figure(figsize=(22, 26), facecolor="#0f1117")
    fig.suptitle(
        "🌐 Geopolitical News Risk Analyzer\n"
        "WW3 / US–Israel–Iran Relevance Pipeline",
        color="white", fontsize=16, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.35)

    # ────────────────────────────── Plot 1: Score distribution ───────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#1a1d2e")
    n, bins, patches = ax1.hist(
        df["composite_score"], bins=40, edgecolor="none", color="#4c72b0"
    )
    for patch, left in zip(patches, bins):
        if   left >= 0.22: patch.set_facecolor("#e74c3c")
        elif left >= 0.12: patch.set_facecolor("#e67e22")
        elif left >= 0.04: patch.set_facecolor("#f1c40f")
    ax1.set_title("Composite Risk Score Distribution", color="white", pad=8)
    ax1.set_xlabel("Composite Score", color="#aaa"); ax1.set_ylabel("Count", color="#aaa")
    ax1.tick_params(colors="#aaa")
    ax1.legend(["Low", "Moderate", "High", "Critical"], fontsize=7)

    # ────────────────────────────── Plot 2: Risk tier pie ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a1d2e")
    tier_counts = df["risk_tier"].value_counts()
    colors_pie = {"Low":"#2ecc71","Moderate":"#f1c40f","High":"#e67e22","Critical":"#e74c3c"}
    wedge_colors = [colors_pie.get(t, "#888") for t in tier_counts.index]
    wedges, texts, autotexts = ax2.pie(
        tier_counts, labels=tier_counts.index, autopct="%1.1f%%",
        colors=wedge_colors, startangle=90,
        wedgeprops={"edgecolor":"#0f1117","linewidth":2},
    )
    for txt in texts + autotexts: txt.set_color("white")
    ax2.set_title("Risk Tier Breakdown", color="white", pad=8)

    # ────────────────────────────── Plot 3: Top sources ──────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#1a1d2e")
    top_src = df["source"].value_counts().head(10)
    bars = ax3.barh(top_src.index[::-1], top_src.values[::-1], color="#5dade2")
    ax3.set_title("Top 10 News Sources", color="white", pad=8)
    ax3.set_xlabel("Article Count", color="#aaa")
    ax3.tick_params(colors="#aaa"); ax3.invert_xaxis()
    ax3.invert_yaxis()

    # ────────────────────────────── Plot 4: Keyword group heatmap ────────────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor("#1a1d2e")
    kw_cols = [c for c in df.columns if c.startswith("kw_")]
    kw_labels = [c.replace("kw_","").replace("_"," ").title() for c in kw_cols]
    kw_by_tier = df.groupby("risk_tier", observed=True)[kw_cols].mean()
    sns.heatmap(
        kw_by_tier.rename(columns=dict(zip(kw_cols, kw_labels))),
        ax=ax4, cmap="YlOrRd", annot=True, fmt=".2f",
        linewidths=0.5, linecolor="#0f1117",
        cbar_kws={"shrink":0.8, "label":"Avg Keyword Hits"},
    )
    ax4.set_title("Keyword Group Hits by Risk Tier", color="white", pad=8)
    ax4.tick_params(colors="#aaa", labelsize=9)
    ax4.set_ylabel("Risk Tier", color="#aaa")
    ax4.yaxis.set_tick_params(rotation=0)

    # ────────────────────────────── Plot 5: TF-IDF similarity radar-bar ──────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1a1d2e")
    sim_cols_plot = [c for c in df.columns if c.startswith("sim_")]
    sim_labels    = [c.replace("sim_","").replace("_"," ").upper() for c in sim_cols_plot]
    mean_sims = df[sim_cols_plot].mean()
    bar_colors = ["#e74c3c","#3498db","#9b59b6","#1abc9c","#e67e22"]
    ax5.bar(range(len(sim_labels)), mean_sims, color=bar_colors[:len(sim_labels)])
    ax5.set_xticks(range(len(sim_labels)))
    ax5.set_xticklabels(sim_labels, rotation=35, ha="right", fontsize=7, color="#aaa")
    ax5.set_title("Mean TF-IDF Similarity by Topic", color="white", pad=8)
    ax5.set_ylabel("Avg Cosine Similarity", color="#aaa")
    ax5.tick_params(colors="#aaa")

    # ────────────────────────────── Plot 6: Time-series articles/day ─────────
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.set_facecolor("#1a1d2e")
    df_time = df.copy()
    df_time["date"] = pd.to_datetime(df_time["published_at"], errors="coerce").dt.date
    daily = df_time.groupby(["date","risk_tier"], observed=True).size().unstack(fill_value=0)
    tier_order = ["Critical","High","Moderate","Low"]
    tier_colors = {"Critical":"#e74c3c","High":"#e67e22","Moderate":"#f1c40f","Low":"#2ecc71"}
    bottom = np.zeros(len(daily))
    for tier in tier_order:
        if tier in daily.columns:
            ax6.bar(
                range(len(daily)), daily[tier].values,
                bottom=bottom, color=tier_colors[tier],
                label=tier, width=0.85
            )
            bottom += daily[tier].values
    ax6.set_title("Articles per Day by Risk Tier", color="white", pad=8)
    ax6.set_xlabel("Days (oldest → newest)", color="#aaa")
    ax6.set_ylabel("Article Count", color="#aaa")
    ax6.legend(loc="upper left", fontsize=8)
    ax6.tick_params(colors="#aaa")

    # ────────────────────────────── Plot 7: Score vs keyword scatter ─────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor("#1a1d2e")
    tier_num = {"Low":0,"Moderate":1,"High":2,"Critical":3}
    colors_scatter = df["risk_tier"].map(
        {"Low":"#2ecc71","Moderate":"#f1c40f","High":"#e67e22","Critical":"#e74c3c"}
    ).fillna("#888")
    ax7.scatter(
        df["total_keyword_hits"], df["composite_score"],
        c=colors_scatter, alpha=0.6, s=18, edgecolors="none"
    )
    # regression line
    mask = df["total_keyword_hits"].notna() & df["composite_score"].notna()
    m, b, r, p, _ = stats.linregress(
        df.loc[mask,"total_keyword_hits"], df.loc[mask,"composite_score"]
    )
    x_line = np.linspace(0, df["total_keyword_hits"].max(), 100)
    ax7.plot(x_line, m*x_line+b, color="white", lw=1.5, linestyle="--",
             label=f"r={r:.2f}, p={p:.3f}")
    ax7.set_title("Keyword Hits vs Composite Score", color="white", pad=8)
    ax7.set_xlabel("Total Keyword Hits", color="#aaa")
    ax7.set_ylabel("Composite Score", color="#aaa")
    ax7.tick_params(colors="#aaa")
    ax7.legend(fontsize=8)

    # ────────────────────────────── Plot 8: Correlation heatmap ──────────────
    ax8 = fig.add_subplot(gs[3, :2])
    ax8.set_facecolor("#1a1d2e")
    corr_cols = ["composite_score","total_keyword_hits","geo_relevance_score"] + sim_cols_plot
    corr_labels = [c.replace("sim_","").replace("_"," ").upper()
                   if c.startswith("sim_") else c.replace("_"," ").title()
                   for c in corr_cols]
    corr_mat = df[corr_cols].corr()
    mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat.rename(columns=dict(zip(corr_cols, corr_labels)),
                        index=dict(zip(corr_cols, corr_labels))),
        ax=ax8, mask=mask_upper, cmap="coolwarm", center=0,
        annot=True, fmt=".2f", linewidths=0.5,
        cbar_kws={"shrink":0.7},
    )
    ax8.set_title("Correlation Matrix: NLP Features", color="white", pad=8)
    ax8.tick_params(colors="#aaa", labelsize=8)

    # ────────────────────────────── Plot 9: Top critical headlines ────────────
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.set_facecolor("#0f1117"); ax9.axis("off")
    critical = df[df["risk_tier"] == "Critical"].nlargest(8, "composite_score")
    ax9.set_title("🔴 Top Critical Headlines", color="#e74c3c", pad=8, fontsize=10)
    y_pos = 0.97
    for _, row in critical.iterrows():
        headline = row["title"][:65] + ("…" if len(row["title"]) > 65 else "")
        score    = row["composite_score"]
        ax9.text(
            0.02, y_pos, f"[{score:.3f}] {headline}",
            transform=ax9.transAxes, fontsize=7.5,
            color="#f8c8c8", verticalalignment="top", wrap=True,
        )
        y_pos -= 0.12

    plt.savefig(
        "geo_news_eda.png",
        dpi=160, bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    log.info(f"  EDA figure saved → geo_news_eda.png")
    plt.close()

    return df

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — REPORTING  (Ch 6.6 / 5.6)
# ══════════════════════════════════════════════════════════════════════════════

@timer
def generate_html_report(df: pd.DataFrame):
    """Auto-generate an HTML analytics report (Ch 6.6)."""
    top10 = (
        df.nlargest(10, "composite_score")
        [["title","source","composite_score","risk_tier","total_keyword_hits"]]
        .round(4)
    )
    tier_dist = df["risk_tier"].value_counts().to_dict()

    rows_html = ""
    tier_bg   = {"Critical":"#e74c3c","High":"#e67e22","Moderate":"#f1c40f","Low":"#2ecc71"}
    for _, r in top10.iterrows():
        bg = tier_bg.get(str(r["risk_tier"]), "#888")
        rows_html += f"""
        <tr>
          <td>{r['title'][:90]}</td>
          <td>{r['source']}</td>
          <td><b>{r['composite_score']:.4f}</b></td>
          <td style="background:{bg};color:#000;font-weight:bold;">{r['risk_tier']}</td>
          <td>{int(r['total_keyword_hits'])}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Geopolitical Risk Report</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background:#0f1117; color:#eee; margin:0; padding:20px; }}
    h1   {{ color:#e74c3c; }} h2 {{ color:#5dade2; border-bottom:1px solid #333; padding-bottom:4px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
    th   {{ background:#1a1d2e; color:#5dade2; padding:10px; text-align:left; }}
    td   {{ padding:8px; border-bottom:1px solid #222; font-size:13px; }}
    tr:hover {{ background:#1e2130; }}
    .stat-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:20px 0; }}
    .stat-card {{ background:#1a1d2e; border-radius:8px; padding:16px; text-align:center; }}
    .stat-card .val {{ font-size:2em; font-weight:bold; color:#e74c3c; }}
    .stat-card .lbl {{ font-size:12px; color:#aaa; margin-top:4px; }}
    img {{ width:100%; border-radius:8px; margin-top:20px; }}
  </style>
</head>
<body>
  <h1>🌐 Geopolitical Risk Report</h1>
  <p style="color:#aaa">Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} |
     Pipeline: News Scraping → ETL → NLP → EDA</p>

  <div class="stat-grid">
    <div class="stat-card"><div class="val">{len(df)}</div><div class="lbl">Total Articles</div></div>
    <div class="stat-card"><div class="val" style="color:#e74c3c">{tier_dist.get('Critical',0)}</div><div class="lbl">Critical Risk</div></div>
    <div class="stat-card"><div class="val" style="color:#e67e22">{tier_dist.get('High',0)}</div><div class="lbl">High Risk</div></div>
    <div class="stat-card"><div class="val" style="color:#5dade2">{df['source'].nunique()}</div><div class="lbl">News Sources</div></div>
  </div>

  <h2>Top 10 Most Geopolitically Relevant Articles</h2>
  <table>
    <tr><th>Headline</th><th>Source</th><th>Score</th><th>Risk Tier</th><th>KW Hits</th></tr>
    {rows_html}
  </table>

  <h2>EDA Dashboard</h2>
  <img src="geo_news_eda.png" alt="EDA Dashboard">
</body>
</html>"""

    rpt_path = "report.html"
    rpt_path.write_text(html, encoding="utf-8")
    log.info(f"  HTML report saved → {rpt_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN PIPELINE ORCHESTRATOR  (Ch 6.2)
# ══════════════════════════════════════════════════════════════════════════════

@timer
def run_pipeline(newsapi_key: str | None = None):
    """
    Full ETL pipeline:
      1. Ingest  — API / RSS / mock
      2. Extract — Lazy chunked stream
      3. Transform — Normalize, deduplicate
      4. Enrich  — NLP scoring (TF-IDF + cosine + keywords)
      5. Load    — SQLite
      6. Analyze — EDA + visualization
      7. Report  — HTML report
    """
    t_total = time.perf_counter()
    log.info("="*60)
    log.info(" PIPELINE START")
    log.info("="*60)

    # ── INGEST ─────────────────────────────────────────────────────────────────
    raw_articles: list[dict] = []

    if newsapi_key:
        for q in ["israel iran war", "us iran escalation", "world war 3 nuclear"]:
            raw_articles += fetch_newsapi(q, newsapi_key)
    else:
        log.info("  No NewsAPI key — using mock dataset (set NEWSAPI_KEY env var for live data)")
        raw_articles = generate_mock_articles(n=400)

    # Optional: add RSS feeds (uncomment if network available)
    # rss_feeds = {
    #     "Reuters World" : "https://feeds.reuters.com/reuters/worldnews",
    #     "BBC World"     : "http://feeds.bbci.co.uk/news/world/rss.xml",
    # }
    # for name, url in rss_feeds.items():
    #     raw_articles += scrape_rss_feed(url, name)

    # ── TRANSFORM (lazy chunked ETL) ───────────────────────────────────────────
    df = etl_transform_chunked(raw_articles)
    if df.empty:
        log.error("No articles after cleaning. Exiting.")
        return

    # ── ENRICH (NLP) ───────────────────────────────────────────────────────────
    df = tfidf_cosine_scoring(df)

    # ── LOAD ───────────────────────────────────────────────────────────────────
    load_to_sqlite(df)

    # ── ANALYZE ────────────────────────────────────────────────────────────────
    df = run_eda(df)

    # ── REPORT ─────────────────────────────────────────────────────────────────
    generate_html_report(df)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "articles_scored.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"  CSV saved → {csv_path}")

    total = time.perf_counter() - t_total
    log.info("="*60)
    log.info(f" PIPELINE COMPLETE in {total:.2f}s")
    log.info(f" Outputs: {OUT_DIR}/")
    log.info("="*60)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    NEWSAPI_KEY = 'a752bdf354ee4625b3ed58d906eec969'  # export NEWSAPI_KEY=your_key
    df_result = run_pipeline(newsapi_key=NEWSAPI_KEY)

    if df_result is not None:
        print("\n" + "─"*60)
        print(" TOP 5 MOST CRITICAL ARTICLES")
        print("─"*60)
        top5 = df_result.nlargest(5, "composite_score")[
            ["title","source","composite_score","risk_tier"]
        ]
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"\n {i}. [{row['risk_tier']}] Score={row['composite_score']:.4f}")
            print(f"    {row['title']}")
            print(f"    Source: {row['source']}")
