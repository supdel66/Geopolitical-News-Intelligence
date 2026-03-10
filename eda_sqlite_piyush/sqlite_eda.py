from __future__ import annotations

import os
import re
import sqlite3
import string
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "sqlite_databases" / "news.db"
OUT_DIR = PROJECT_ROOT / "eda_sqlite_piyush" / "output"

TOP_N_SOURCES = 15
SIMILARITY_THRESHOLD = 0.92  # tune: 0.88–0.95

# Pie chart readability controls.
# We keep the underlying counts unchanged; this only affects labeling.
PIE_MIN_PCT_FOR_LABEL = 4.0
PIE_MIN_PCT_FOR_PERCENT = 2.0


def clean_title(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_articles() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found at: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        return pd.read_sql_query("SELECT * FROM articles", conn)
    finally:
        conn.close()


def with_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["published_dt"] = pd.to_datetime(df2.get("published_at"), errors="coerce", utc=True)
    df2["date"] = df2["published_dt"].dt.date
    df2["hour"] = df2["published_dt"].dt.hour
    return df2


def with_length_cols(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["title_len"] = df2["title"].fillna("").astype(str).str.split().str.len()
    df2["content_len"] = df2["content"].fillna("").astype(str).str.split().str.len()
    df2["article_len"] = df2["title_len"] + df2["content_len"]
    return df2


def plot_sources(df: pd.DataFrame) -> None:
    """Generate source distribution charts (bar + pie) from SQLite data."""
    sns.set_theme(style="whitegrid")

    counts = df["source"].fillna("Unknown").astype(str).value_counts()
    top = counts.head(TOP_N_SOURCES)

    # Bar chart
    plt.figure(figsize=(11, 6))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.title(f"Top {TOP_N_SOURCES} Sources by Article Count")
    plt.xlabel("Articles")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bar_articles_per_source.png")
    plt.close()

    # Pie chart (top N + Other)
    other = counts.iloc[TOP_N_SOURCES:].sum()
    pie_series = top.copy()
    if other > 0:
        pie_series.loc["Other"] = other

    # Pie charts get unreadable fast when there are many small slices.
    # Strategy:
    # - Show labels in an external legend (no overlap).
    # - Only show % text on the wedge if the slice is large enough.
    # - Only include the wedge label for sufficiently large slices.
    def _autopct(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= PIE_MIN_PCT_FOR_PERCENT else ""

    total = float(pie_series.sum()) if float(pie_series.sum()) else 1.0
    legend_labels = [f"{name} ({value} / {value/total*100:.1f}%)" for name, value in pie_series.items()]
    wedge_labels = [name if (value / total * 100.0) >= PIE_MIN_PCT_FOR_LABEL else "" for name, value in pie_series.items()]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        pie_series.values,
        labels=wedge_labels,
        autopct=_autopct,
        startangle=140,
        labeldistance=1.05,
        pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(9)

    ax.set_title("Source Distribution")
    ax.legend(
        wedges,
        legend_labels,
        title="Sources",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=9,
        title_fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pie_source_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_time(df: pd.DataFrame) -> None:
    """Generate time-based charts: line over days, histogram of daily counts, peak hours."""
    sns.set_theme(style="whitegrid")

    df_ok = df.dropna(subset=["published_dt"]).copy()
    if df_ok.empty:
        return

    daily = df_ok.groupby("date").size().reset_index(name="count").sort_values("date")

    # Line graph: articles over time (daily)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily, x="date", y="count", marker="o")
    plt.title("News Articles Over Time (Daily)")
    plt.xlabel("Date")
    plt.ylabel("Articles")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "line_articles_over_time.png")
    plt.close()

    # Histogram: publishing frequency (distribution of daily counts)
    plt.figure(figsize=(10, 6))
    sns.histplot(daily["count"], bins=20, kde=True, color="#3498db")
    plt.title("Publishing Frequency (Histogram of Daily Article Counts)")
    plt.xlabel("Articles per day")
    plt.ylabel("Number of days")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hist_publishing_frequency.png")
    plt.close()

    # Peak hours (UTC)
    hourly = df_ok.groupby("hour").size().reindex(range(24), fill_value=0).reset_index()
    hourly.columns = ["hour", "count"]

    plt.figure(figsize=(12, 5))
    sns.barplot(data=hourly, x="hour", y="count", color="#7f8c8d")
    plt.title("Peak News Hours (UTC, from published_at)")
    plt.xlabel("Hour (0-23)")
    plt.ylabel("Articles")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bar_peak_hours.png")
    plt.close()


@dataclass(frozen=True)
class DuplicateStats:
    duplicate_links: int
    duplicate_titles_exact: int
    duplicate_titles_normalized: int
    similar_title_pairs: int


def duplicate_detection(df: pd.DataFrame) -> DuplicateStats:
    """Detect duplicates by exact keys and near-duplicates by fuzzy title similarity.

    Notes:
    - Exact duplicates: link/title repeated.
    - Normalized duplicates: title after punctuation/spacing normalization.
    - Similar titles: uses a lightweight, stdlib-only similarity ratio.
      This can create false positives for generic headlines; tune SIMILARITY_THRESHOLD.
    """
    # Exact duplicates
    duplicate_links = int(df["link"].duplicated(keep=False).sum()) if "link" in df else 0
    duplicate_titles_exact = int(df["title"].duplicated(keep=False).sum()) if "title" in df else 0

    # Normalized title duplicates
    normalized = df["title"].fillna("").astype(str).map(clean_title)
    duplicate_titles_normalized = int(normalized.duplicated(keep=False).sum())

    # Similar title detection (blocked) to reduce O(n^2)
    # Block by first 12 characters of normalized title.
    blocks: dict[str, list[int]] = {}
    normalized_list = normalized.tolist()
    for idx, t in enumerate(normalized_list):
        if not t:
            continue
        key = t[:12]
        blocks.setdefault(key, []).append(idx)

    pairs: list[tuple[int, int, float]] = []
    for idxs in blocks.values():
        if len(idxs) < 2:
            continue
        # Compare within block
        for i_pos in range(len(idxs)):
            i = idxs[i_pos]
            a = normalized_list[i]
            for j_pos in range(i_pos + 1, len(idxs)):
                j = idxs[j_pos]
                b = normalized_list[j]
                if a == b:
                    continue
                # quick length filter
                if abs(len(a) - len(b)) > 15:
                    continue
                score = similarity(a, b)
                if score >= SIMILARITY_THRESHOLD:
                    pairs.append((i, j, float(score)))

    # Visualize similarity distribution as a graph instead of CSV
    if pairs:
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Create similarity score distribution histogram
        scores = [p[2] for p in pairs]
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, color='#e67e22', edgecolor='black', alpha=0.7)
        plt.axvline(SIMILARITY_THRESHOLD, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold ({SIMILARITY_THRESHOLD})')
        plt.title('Distribution of Similar Title Similarity Scores', fontsize=14)
        plt.xlabel('Similarity Score')
        plt.ylabel('Number of Pairs')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "similarity_distribution.png")
        plt.close()

    return DuplicateStats(
        duplicate_links=duplicate_links,
        duplicate_titles_exact=duplicate_titles_exact,
        duplicate_titles_normalized=duplicate_titles_normalized,
        similar_title_pairs=len(pairs),
    )


def hypothesis_test_lengths(df: pd.DataFrame) -> dict:
    """Hypothesis test: H0 = all sources have similar mean article lengths.

    Implementation:
    - Uses one-way ANOVA via scipy if available.
    - Skips sources with <10 samples to avoid noisy groups.
    """
    result: dict[str, object] = {}

    try:
        from scipy.stats import f_oneway  # type: ignore
    except Exception:
        result["anova_status"] = "skipped (scipy not installed)"
        return result

    df2 = df.dropna(subset=["source", "article_len"]).copy()
    groups = []
    labels = []
    for src, g in df2.groupby("source"):
        if len(g) >= 10:
            groups.append(g["article_len"].values)
            labels.append(str(src))

    if len(groups) < 2:
        result["anova_status"] = "skipped (not enough sources with >=10 samples)"
        return result

    stat, p_value = f_oneway(*groups)
    result["anova_status"] = "ok"
    result["anova_f_stat"] = float(stat)
    result["anova_p_value"] = float(p_value)
    result["anova_groups_used"] = int(len(groups))
    return result


def simple_trend_prediction(df: pd.DataFrame) -> dict:
    """Very basic 7-day forecast using 7-day MA + weekday seasonal baseline.

    This is intentionally simple EDA-style forecasting (not a production model).
    Output is written as a forecast visualization graph.
    """
    out: dict[str, object] = {}

    df_ok = df.dropna(subset=["published_dt"]).copy()
    if df_ok.empty:
        out["forecast_status"] = "skipped (no parseable dates)"
        return out

    daily = df_ok.groupby("date").size().sort_index()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.asfreq("D", fill_value=0)

    out["forecast_status"] = "ok"
    out["last_date"] = str(daily.index.max().date())
    out["last_30d_mean"] = float(daily.tail(30).mean()) if len(daily) else 0.0

    last7_ma = daily.rolling(7).mean().iloc[-1]
    weekday_means = daily.groupby(daily.index.dayofweek).mean()

    future_predictions = []
    future_dates = []
    start = daily.index.max() + pd.Timedelta(days=1)
    for k in range(7):
        day = start + pd.Timedelta(days=k)
        seasonal = float(weekday_means.loc[day.dayofweek])
        baseline = float(last7_ma) if pd.notna(last7_ma) else seasonal
        pred = max(0.0, 0.5 * baseline + 0.5 * seasonal)
        future_predictions.append(float(pred))
        future_dates.append(day)

    # Create forecast visualization
    plt.figure(figsize=(12, 6))
    
    # Plot historical data (last 14 days)
    historical = daily.tail(14)
    plt.plot(historical.index, historical.values, marker='o', label='Historical', 
             color='#2c3e50', linewidth=2)
    
    # Plot forecast
    plt.plot(future_dates, future_predictions, marker='s', label='7-Day Forecast', 
             color='#e74c3c', linewidth=2, linestyle='--')
    
    plt.title("7-Day Article Count Forecast", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Article Count")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "forecast_7_days.png")
    plt.close()

    out["forecast_rows"] = len(future_predictions)
    return out


def main() -> None:
    """Run the SQLite EDA and write charts/CSVs under eda_sqlite_piyush/output/."""
    ensure_output_dir()

    df = load_articles()
    if df.empty:
        print("No rows found in articles table.")
        return

    df = with_datetime_cols(df)
    df = with_length_cols(df)

    plot_sources(df)
    plot_time(df)

    # Duplicates
    dup = duplicate_detection(df)

    # Hypothesis testing
    hypo = hypothesis_test_lengths(df)

    # Trend prediction
    pred = simple_trend_prediction(df)

    # Summary visualization as graphs instead of CSV
    summary = {
        "Total Rows": int(len(df)),
        "Unique Sources": int(df["source"].nunique()) if "source" in df else 0,
        "Unique Links": int(df["link"].nunique()) if "link" in df else 0,
        "Parseable Dates": int(df["published_dt"].notna().sum()),
        "Duplicate Links": dup.duplicate_links,
        "Exact Duplicate Titles": dup.duplicate_titles_exact,
        "Normalized Duplicates": dup.duplicate_titles_normalized,
        "Similar Title Pairs": dup.similar_title_pairs,
    }

    # Create summary statistics visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Basic statistics
    basic_stats = {
        "Total Rows": summary["Total Rows"],
        "Unique Sources": summary["Unique Sources"],
        "Unique Links": summary["Unique Links"],
        "Parseable Dates": summary["Parseable Dates"],
    }
    ax1.barh(list(basic_stats.keys()), list(basic_stats.values()), color='#3498db')
    ax1.set_xlabel('Count', fontsize=12)
    ax1.set_title('Basic Statistics', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Right plot: Duplicate statistics
    dup_stats = {
        "Duplicate Links": summary["Duplicate Links"],
        "Exact Duplicates": summary["Exact Duplicate Titles"],
        "Normalized Dups": summary["Normalized Duplicates"],
        "Similar Pairs": summary["Similar Title Pairs"],
    }
    colors = ['#e74c3c' if v > 0 else '#95a5a6' for v in dup_stats.values()]
    ax2.barh(list(dup_stats.keys()), list(dup_stats.values()), color=colors)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_title('Duplicate/Similarity Detection', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_statistics.png")
    plt.close()

    # Write a small README for humans explaining what the visualizations contain.
    readme = OUT_DIR / "README.md"
    readme.write_text(
        """# eda_sqlite_piyush outputs

These files are generated from `sqlite_databases/news.db` (table: `articles`).

## Visualizations

### Source Analysis
- `bar_articles_per_source.png`: Bar chart of top sources by article count.
- `pie_source_distribution.png`: Pie chart of source distribution (labels moved to legend to avoid overlap; tiny slices hide % text).

### Time-based Analysis
- `line_articles_over_time.png`: Daily article counts over time.
- `hist_publishing_frequency.png`: Histogram of daily article counts (publishing frequency).
- `bar_peak_hours.png`: Articles by hour-of-day (UTC).

### Duplicate Detection
- `similarity_distribution.png`: Histogram showing distribution of similarity scores for near-duplicate title pairs.

### Summary & Forecast
- `summary_statistics.png`: Key summary metrics including rows, unique sources/links, parseable dates, and duplicates displayed as bar charts.
- `forecast_7_days.png`: 7-day article count forecast visualization (moving-average + weekday baseline).

All outputs are in PNG format for easy visualization and understanding.
""",
        encoding="utf-8",
    )

    print("Done. Outputs saved to:", str(OUT_DIR))


if __name__ == "__main__":
    # Helps on some Windows setups with non-UTF8 default encodings
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
