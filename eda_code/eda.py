import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import pandas as pd
import string
from utils import timer_logger, logger

# Optional heavy deps — caught gracefully if missing
try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False
    logger.warning("[EDA] textblob not installed — sentiment analysis will be skipped.")

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning("[EDA] scikit-learn not installed — n-gram / TF-IDF analyses will be skipped.")

EDA_DIR = "eda_output"

# ─────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────

KEYWORDS = [
    "war", "iran", "israel", "united states",
    "us", "missile", "attack", "military",
    "retaliation", "conflict", "gaza",
    "tehran", "hezbollah", "world war 3",
    "ww3", "escalation", "nuclear", "drone",
    "strike", "idf", "hamas", "houthi", "sanction"
]

# Geopolitical actors + aliases
CONFLICT_ACTORS = {
    "Israel":       ["israel", "idf", "netanyahu", "tel aviv", "jerusalem"],
    "Iran":         ["iran", "tehran", "khamenei", "irgc", "iranians"],
    "USA":          ["united states", " us ", "america", "biden", "trump", "pentagon", "washington dc"],
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

# Conflict theater classification
CONFLICT_THEATERS = {
    "Middle East":       ["israel", "iran", "gaza", "hezbollah", "hamas", "beirut",
                          "tehran", "houthi", "red sea", "syria", "damascus", "west bank"],
    "Russia/Ukraine":    ["russia", "ukraine", "putin", "zelensky", "nato", "moscow",
                          "kyiv", "kiev", "eastern europe", "donbas", "crimea"],
    "Asia-Pacific":      ["china", "taiwan", "north korea", "south korea", "beijing",
                          "kim jong", "south china sea", "indo-pacific", "japan", "pyongyang"],
    "Global WW3":        ["world war", "ww3", "nuclear", "armageddon", "global conflict",
                          "thermonuclear", "doomsday", "mutual assured destruction", "icbm"],
    "US Policy/Sanctions": ["pentagon", "white house", "us sanction", "trump", "biden",
                             "us military", "congress", "state department", "cia"],
}

# Escalation keyword weights (negative = de-escalatory)
ESCALATION_WEIGHTS = {
    "nuclear":                  10,
    "thermonuclear":            10,
    "armageddon":               10,
    "world war 3":              10,
    "ww3":                      10,
    "doomsday":                  9,
    "mutual assured destruction":10,
    "icbm":                      9,
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
    "ceasefire":                -3,
    "peace talks":              -4,
    "diplomacy":                -3,
    "negotiation":              -2,
    "agreement":                -2,
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def compute_escalation_score(text):
    """Weighted sum of escalation keywords; clamped [0, 100]."""
    if not isinstance(text, str):
        return 0.0
    tl = text.lower()
    score = sum(tl.count(kw) * w for kw, w in ESCALATION_WEIGHTS.items())
    return float(max(0.0, min(score, 100.0)))


def classify_theater(text):
    """Assign the dominant conflict theater to an article."""
    if not isinstance(text, str):
        return "Other"
    tl = text.lower()
    scores = {t: sum(tl.count(k) for k in kws) for t, kws in CONFLICT_THEATERS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def get_sentiment_polarity(text):
    """TextBlob polarity on first 500 chars. Returns 0.0 if unavailable."""
    if not _HAS_TEXTBLOB or not isinstance(text, str) or not text.strip():
        return 0.0
    try:
        return float(TextBlob(text[:500]).sentiment.polarity)
    except Exception:
        return 0.0


def label_sentiment(polarity):
    if polarity > 0.05:
        return "Positive"
    if polarity < -0.05:
        return "Negative"
    return "Neutral"


# ─────────────────────────────────────────────
# MAIN EDA FUNCTION
# ─────────────────────────────────────────────

@timer_logger
def run_eda(df):
    stats = {}

    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)

    if df.empty:
        logger.warning("No data available for EDA.")
        return {}

    logger.info("Generating EDA charts...")
    sns.set_theme(style="darkgrid")
    DARK_BG    = "#0d1117"
    CARD_BG    = "#161b22"
    ACCENT     = "#58a6ff"
    WARN       = "#f0883e"
    DANGER     = "#ff4444"
    SUCCESS    = "#3fb950"
    TEXT_COLOR = "#c9d1d9"

    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    CARD_BG,
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        "#21262d",
        "grid.linewidth":    0.6,
    })

    stats['total_articles'] = int(len(df))

    # Pre-compute combined cleaned text once
    df = df.copy()
    df['combined_text'] = (
        df['title'].fillna('') + " " + df['content'].fillna('')
    ).apply(clean_text)

    # ═══════════════════════════════════════════════════════
    # SECTION 1 — BASIC OVERVIEW (original 4 charts)
    # ═══════════════════════════════════════════════════════

    # 1a. Sources Distribution
    logger.info("[EDA 1/13] Top sources...")
    source_counts = df['source'].value_counts().head(10)
    stats['top_sources'] = source_counts.to_dict()

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=DARK_BG)
    colors = sns.color_palette("Blues_r", len(source_counts))
    bars = ax.barh(source_counts.index[::-1], source_counts.values[::-1], color=colors[::-1])
    for bar, val in zip(bars, source_counts.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', color=TEXT_COLOR, fontsize=9)
    ax.set_title("Top 10 News Sources", fontsize=14, color=ACCENT, pad=12)
    ax.set_xlabel("Article Count", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "top_sources.png"), dpi=120)
    plt.close(fig)

    # 1b. Keyword Frequency
    logger.info("[EDA 2/13] Keyword frequency...")
    all_text = " ".join(df['combined_text'].values)
    words = all_text.split()
    word_counts = Counter(w for w in words if w in KEYWORDS)
    for mk in ["united states", "world war 3"]:
        cnt = all_text.count(mk.replace(" ", ""))  # already cleaned, spaces stripped
        word_counts[mk] = all_text.count(mk)
    sorted_kw = dict(sorted({k: v for k, v in word_counts.items() if v > 0}.items(),
                             key=lambda x: x[1], reverse=True)[:15])
    stats['top_keywords'] = sorted_kw

    if sorted_kw:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK_BG)
        colors = sns.color_palette("magma", len(sorted_kw))
        ax.bar(list(sorted_kw.keys()), list(sorted_kw.values()), color=colors, edgecolor=DARK_BG)
        ax.set_title("Conflict Keyword Frequencies", fontsize=14, color=ACCENT, pad=12)
        ax.set_xlabel("Keyword")
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "top_keywords.png"), dpi=120)
        plt.close(fig)

    # 1c. Timeline
    logger.info("[EDA 3/13] Timeline...")
    df['parsed_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
    df_tl = df.dropna(subset=['parsed_date']).copy()
    if not df_tl.empty:
        df_tl['date_only'] = df_tl['parsed_date'].dt.date
        daily = df_tl.groupby('date_only').size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(13, 5), facecolor=DARK_BG)
        ax.plot(daily['date_only'], daily['count'], color=DANGER, linewidth=2.2, marker='o', markersize=4)
        ax.fill_between(daily['date_only'], daily['count'], alpha=0.15, color=DANGER)
        ax.set_title("Articles Published Over Time", fontsize=14, color=ACCENT, pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Article Count")
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(os.path.join(EDA_DIR, "articles_over_time.png"), dpi=120)
        plt.close(fig)

    # 1d. Article Length Distribution
    logger.info("[EDA 4/13] Length distribution...")
    df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))
    mean_wc  = df['word_count'].mean()
    median_wc = df['word_count'].median()
    stats['mean_words']   = float(mean_wc)
    stats['median_words'] = float(median_wc)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    ax.hist(df['word_count'], bins=30, color=ACCENT, edgecolor=DARK_BG, alpha=0.85)
    ax.axvline(mean_wc,   color=DANGER,  linestyle='--', linewidth=2, label=f"Mean: {mean_wc:.0f}")
    ax.axvline(median_wc, color=SUCCESS, linestyle='--', linewidth=2, label=f"Median: {median_wc:.0f}")
    ax.legend(facecolor=CARD_BG)
    ax.set_title("Article Word Count Distribution", fontsize=14, color=ACCENT, pad=12)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "article_length_dist.png"), dpi=120)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SECTION 2 — GEOPOLITICAL ACTOR INTELLIGENCE
    # ═══════════════════════════════════════════════════════

    # 2a. Country / Actor Mention Frequency
    logger.info("[EDA 5/13] Actor mentions...")
    actor_counts = {}
    for actor, terms in CONFLICT_ACTORS.items():
        total = sum(df['combined_text'].str.count(re.escape(t)).sum() for t in terms)
        actor_counts[actor] = int(total)
    sorted_actors = dict(sorted(actor_counts.items(), key=lambda x: x[1], reverse=True))
    stats['actor_mentions'] = sorted_actors

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
    a_names = list(sorted_actors.keys())[::-1]
    a_vals  = [sorted_actors[k] for k in a_names]
    palette = sns.color_palette("YlOrRd", len(a_names))
    bars = ax.barh(a_names, a_vals, color=palette)
    for bar, val in zip(bars, a_vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', color=TEXT_COLOR, fontsize=9)
    ax.set_title("Geopolitical Actor Mention Frequency", fontsize=14, color=ACCENT, pad=12)
    ax.set_xlabel("Total Mentions (Title + Content)")
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "country_actor_mentions.png"), dpi=120)
    plt.close(fig)

    # 2b. Conflict Theater Classification
    logger.info("[EDA 6/13] Conflict theaters...")
    df['theater'] = df['combined_text'].apply(classify_theater)
    theater_counts = df['theater'].value_counts()
    stats['conflict_theaters'] = theater_counts.to_dict()
    stats['dominant_theater']  = str(theater_counts.idxmax()) if not theater_counts.empty else "Unknown"

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=DARK_BG)
    theater_palette = ["#e05252", "#e0a020", "#52b0e0", "#8b52e0", "#52e085", "#888888"]
    wedge_props = dict(width=0.55, edgecolor=DARK_BG, linewidth=2)
    wedges, texts, autotexts = ax.pie(
        theater_counts.values, labels=theater_counts.index,
        autopct='%1.1f%%', startangle=140,
        colors=theater_palette[:len(theater_counts)],
        wedgeprops=wedge_props,
        textprops={'color': TEXT_COLOR, 'fontsize': 11}
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title("Article Distribution by Conflict Theater", fontsize=14, color=ACCENT, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "conflict_theaters.png"), dpi=120)
    plt.close(fig)

    # 2c. Actor Co-Occurrence Heatmap
    logger.info("[EDA 7/13] Actor co-occurrence heatmap...")
    top_actors = [a for a, _ in sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    comat = pd.DataFrame(0, index=top_actors, columns=top_actors)

    for text in df['combined_text']:
        mentioned = [
            a for a in top_actors
            if any(t in text for t in CONFLICT_ACTORS[a])
        ]
        for i_a in mentioned:
            for i_b in mentioned:
                if i_a != i_b:
                    comat.loc[i_a, i_b] += 1

    np.fill_diagonal(comat.values, 0)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)
    mask = comat == 0
    sns.heatmap(
        comat, annot=True, fmt='d', cmap='YlOrRd',
        linewidths=0.5, linecolor=DARK_BG,
        ax=ax, mask=mask,
        annot_kws={'size': 9, 'color': 'black'},
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title("Actor Co-Occurrence in Articles\n(Higher = More Articles Mention Both Together)",
                 fontsize=13, color=ACCENT, pad=12)
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(axis='y', labelrotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "actor_cooccurrence_heatmap.png"), dpi=120)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SECTION 3 — ESCALATION SCORING
    # ═══════════════════════════════════════════════════════

    # 3a. Escalation Score Distribution + Trend
    logger.info("[EDA 8/13] Escalation scores...")
    df['escalation_score'] = df['combined_text'].apply(compute_escalation_score)
    mean_esc = df['escalation_score'].mean()
    stats['avg_escalation_score'] = float(mean_esc)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=DARK_BG)

    # Distribution
    axes[0].hist(df['escalation_score'], bins=30, color=WARN, edgecolor=DARK_BG, alpha=0.85)
    axes[0].axvline(mean_esc, color=DANGER, linestyle='--', linewidth=2, label=f"Mean: {mean_esc:.1f}")
    axes[0].legend(facecolor=CARD_BG)
    axes[0].set_title("Escalation Score Distribution", color=ACCENT, fontsize=12)
    axes[0].set_xlabel("Escalation Score (0-100)")
    axes[0].set_ylabel("Article Count")
    axes[0].set_facecolor(CARD_BG)

    # Trend over time
    if not df_tl.empty:
        df_esc_tl = df.loc[df_tl.index].copy() if df_tl is not None else df.copy()
        df_esc_tl['parsed_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
        df_esc_tl = df_esc_tl.dropna(subset=['parsed_date']).copy()
        df_esc_tl['date_only'] = df_esc_tl['parsed_date'].dt.date
        daily_esc = df_esc_tl.groupby('date_only')['escalation_score'].mean().reset_index()
        if len(daily_esc) >= 3:
            daily_esc['rolling_avg'] = daily_esc['escalation_score'].rolling(3, min_periods=1).mean()
            axes[1].plot(daily_esc['date_only'], daily_esc['escalation_score'],
                         alpha=0.35, color=WARN, linewidth=1.2, label='Daily Avg')
            axes[1].plot(daily_esc['date_only'], daily_esc['rolling_avg'],
                         color=DANGER, linewidth=2.5, label='3-Day Rolling Avg')
            axes[1].legend(facecolor=CARD_BG)
        else:
            axes[1].plot(daily_esc['date_only'], daily_esc['escalation_score'], color=WARN, linewidth=2)
        axes[1].set_title("Escalation Score Trend Over Time", color=ACCENT, fontsize=12)
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Avg Escalation Score")
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        axes[1].text(0.5, 0.5, "Date parsing failed", transform=axes[1].transAxes,
                     ha='center', color=TEXT_COLOR)
    axes[1].set_facecolor(CARD_BG)

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "escalation_trend.png"), dpi=120)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SECTION 4 — SENTIMENT ANALYSIS
    # ═══════════════════════════════════════════════════════

    logger.info("[EDA 9/13] Sentiment analysis...")
    df['sentiment_polarity'] = (df['title'].fillna('') + " " + df['content'].fillna('')
                                ).apply(get_sentiment_polarity)
    df['sentiment_label'] = df['sentiment_polarity'].apply(label_sentiment)
    avg_sent = df['sentiment_polarity'].mean()
    sent_dist = df['sentiment_label'].value_counts(normalize=True).mul(100).round(1).to_dict()
    stats['avg_sentiment']          = float(avg_sent)
    stats['sentiment_distribution'] = sent_dist

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=DARK_BG)

    # Sentiment by source
    top10_src = source_counts.head(8).index.tolist()
    df_src_sent = df[df['source'].isin(top10_src)]
    if not df_src_sent.empty:
        order = df_src_sent.groupby('source')['sentiment_polarity'].median().sort_values().index.tolist()
        sent_palette = {s: (DANGER if df_src_sent[df_src_sent['source']==s]['sentiment_polarity'].median() < 0
                            else SUCCESS) for s in order}
        bplot = sns.boxplot(
            data=df_src_sent, x='sentiment_polarity', y='source',
            order=order, ax=axes[0],
            palette=sent_palette, linewidth=0.8,
            flierprops=dict(marker='o', markerfacecolor=WARN, markersize=3, alpha=0.5)
        )
        axes[0].axvline(0, color=TEXT_COLOR, linestyle='--', linewidth=0.9, alpha=0.6)
        axes[0].set_title("Sentiment Polarity Distribution by Source", color=ACCENT, fontsize=12)
        axes[0].set_xlabel("Polarity  (Negative ← 0 → Positive)")
        axes[0].set_ylabel("")

    # Sentiment trend over time
    if not df_tl.empty:
        df_st = df.copy()
        df_st['parsed_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
        df_st = df_st.dropna(subset=['parsed_date']).copy()
        df_st['date_only'] = df_st['parsed_date'].dt.date
        sent_time = df_st.groupby('date_only')['sentiment_polarity'].mean().reset_index()
        if len(sent_time) >= 3:
            sent_time['rolling'] = sent_time['sentiment_polarity'].rolling(3, min_periods=1).mean()
            axes[1].bar(sent_time['date_only'], sent_time['sentiment_polarity'],
                        color=[SUCCESS if v >= 0 else DANGER for v in sent_time['sentiment_polarity']],
                        alpha=0.45, width=0.8)
            axes[1].plot(sent_time['date_only'], sent_time['rolling'],
                         color=ACCENT, linewidth=2.2, label='3-Day Rolling Avg')
            axes[1].axhline(0, color=TEXT_COLOR, linewidth=0.8, linestyle='--', alpha=0.5)
            axes[1].legend(facecolor=CARD_BG)
        else:
            axes[1].plot(sent_time['date_only'], sent_time['sentiment_polarity'], color=ACCENT)
        axes[1].set_title("Sentiment Trend Over Time", color=ACCENT, fontsize=12)
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Avg Polarity")
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        axes[1].text(0.5, 0.5, "Date parsing failed", transform=axes[1].transAxes,
                     ha='center', color=TEXT_COLOR)
    for ax in axes:
        ax.set_facecolor(CARD_BG)

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "sentiment_analysis.png"), dpi=120)
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SECTION 5 — SOURCE PUBLICATION VELOCITY
    # ═══════════════════════════════════════════════════════

    logger.info("[EDA 10/13] Source publication velocity...")
    if not df_tl.empty:
        df_vel = df.copy()
        df_vel['parsed_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
        df_vel = df_vel.dropna(subset=['parsed_date']).copy()
        df_vel['date_only'] = df_vel['parsed_date'].dt.date

        top_src = source_counts.head(6).index.tolist()
        velocity = (df_vel[df_vel['source'].isin(top_src)]
                    .groupby(['date_only', 'source'])
                    .size()
                    .unstack(fill_value=0))

        if not velocity.empty:
            fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK_BG)
            vel_palette = sns.color_palette("tab10", len(velocity.columns))
            velocity.plot.area(ax=ax, color=vel_palette, alpha=0.75, linewidth=0)
            ax.set_title("Publication Velocity by Source (Stacked Area)", fontsize=13, color=ACCENT, pad=12)
            ax.set_xlabel("Date")
            ax.set_ylabel("Articles Published")
            ax.legend(title="Source", facecolor=CARD_BG, title_fontsize=9, fontsize=8,
                      loc='upper left')
            plt.xticks(rotation=45, ha='right')
            ax.set_facecolor(CARD_BG)
            fig.tight_layout()
            fig.savefig(os.path.join(EDA_DIR, "source_velocity.png"), dpi=120)
            plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SECTION 6 — N-GRAM / PHRASE INTELLIGENCE
    # ═══════════════════════════════════════════════════════

    logger.info("[EDA 11/13] N-gram analysis...")
    if _HAS_SKLEARN and len(df) >= 10:
        try:
            # Use raw (non-cleaned) text to preserve multi-word phrases better
            raw_texts = (df['title'].fillna('') + " " + df['content'].fillna('')).tolist()

            # Bigrams
            bi_vec = CountVectorizer(ngram_range=(2, 2), stop_words='english',
                                     max_features=2000, min_df=2)
            bi_mat = bi_vec.fit_transform(raw_texts)
            bi_counts = bi_mat.sum(axis=0).A1
            bi_features = bi_vec.get_feature_names_out()
            top_bi = sorted(zip(bi_features, bi_counts), key=lambda x: x[1], reverse=True)[:20]

            # Trigrams
            tri_vec = CountVectorizer(ngram_range=(3, 3), stop_words='english',
                                      max_features=2000, min_df=2)
            tri_mat = tri_vec.fit_transform(raw_texts)
            tri_counts = tri_mat.sum(axis=0).A1
            tri_features = tri_vec.get_feature_names_out()
            top_tri = sorted(zip(tri_features, tri_counts), key=lambda x: x[1], reverse=True)[:15]

            stats['top_bigrams']  = [(str(k), int(v)) for k, v in top_bi[:10]]
            stats['top_trigrams'] = [(str(k), int(v)) for k, v in top_tri[:8]]

            fig, axes = plt.subplots(1, 2, figsize=(17, 7), facecolor=DARK_BG)

            # Bigrams
            bi_labels = [b[0] for b in top_bi[:15]][::-1]
            bi_vals   = [b[1] for b in top_bi[:15]][::-1]
            axes[0].barh(bi_labels, bi_vals,
                         color=sns.color_palette("Blues_r", len(bi_labels)))
            axes[0].set_title("Top 15 Bigrams (Conflict Phrases)", color=ACCENT, fontsize=12)
            axes[0].set_xlabel("Occurrences")
            axes[0].set_facecolor(CARD_BG)

            # Trigrams
            tri_labels = [t[0] for t in top_tri[:12]][::-1]
            tri_vals   = [t[1] for t in top_tri[:12]][::-1]
            axes[1].barh(tri_labels, tri_vals,
                         color=sns.color_palette("Purples_r", len(tri_labels)))
            axes[1].set_title("Top 12 Trigrams (Key Phrases)", color=ACCENT, fontsize=12)
            axes[1].set_xlabel("Occurrences")
            axes[1].set_facecolor(CARD_BG)

            fig.tight_layout()
            fig.savefig(os.path.join(EDA_DIR, "top_ngrams.png"), dpi=120)
            plt.close(fig)

        except Exception as e:
            logger.warning(f"[EDA] N-gram analysis failed: {e}")

    # ═══════════════════════════════════════════════════════
    # SECTION 7 — WW3 THREAT LEVEL GAUGE
    # ═══════════════════════════════════════════════════════

    logger.info("[EDA 12/13] WW3 Threat Level...")

    # Composite score components (all normalised to 0-100)
    total_words = df['word_count'].sum() if df['word_count'].sum() > 0 else 1
    nuclear_density   = min((all_text.count("nuclear") / total_words) * 5000, 100)
    avg_esc_norm      = min(stats.get('avg_escalation_score', 0) * 1.5, 100)
    ww3_density       = min((all_text.count("ww3") + all_text.count("world war 3")) / max(len(df), 1) * 20, 100)
    missile_density   = min(all_text.count("missile") / total_words * 3000, 100)
    sentiment_neg     = sent_dist.get("Negative", 0)  # already a percentage

    threat_level = (
        avg_esc_norm    * 0.35 +
        nuclear_density * 0.25 +
        sentiment_neg   * 0.15 +
        ww3_density     * 0.15 +
        missile_density * 0.10
    )
    threat_level = min(round(float(threat_level), 1), 100.0)
    stats['ww3_threat_level'] = threat_level

    # Color zones
    def threat_color(tl):
        if tl < 25:  return "#3fb950"   # green
        if tl < 50:  return "#f0883e"   # orange
        if tl < 75:  return "#e05252"   # red
        return "#ff0000"                # critical

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Background track
    ax.barh(0.5, 100, height=0.35, color="#1c2128", left=0, zorder=1)
    # Zone bands
    for start, end, col in [(0, 25, "#3fb95033"), (25, 50, "#e0a02033"),
                              (50, 75, "#e0525233"), (75, 100, "#ff000033")]:
        ax.barh(0.5, end - start, height=0.35, color=col, left=start, zorder=2)
    # Filled value bar
    ax.barh(0.5, threat_level, height=0.35, color=threat_color(threat_level), left=0, zorder=3, alpha=0.9)

    # Tick labels
    for v, label in [(0, "0\nLow"), (25, "25"), (50, "50\nMedium"), (75, "75"), (100, "100\nCritical")]:
        ax.text(v, 0.18, label, ha='center', va='top', color=TEXT_COLOR, fontsize=8.5)

    # Big value
    ax.text(threat_level, 0.85, f"{threat_level}", ha='center', va='center',
            color=threat_color(threat_level), fontsize=26, fontweight='bold')
    ax.text(50, 0.05, "/ 100", ha='center', va='bottom', color="#555", fontsize=10)

    title_color = threat_color(threat_level)
    ax.set_title(f"WW3 THREAT LEVEL INDICATOR  —  {threat_level}/100",
                 fontsize=15, color=title_color, pad=10, fontweight='bold')

    comp_text = (
        f"Components:  Escalation={avg_esc_norm:.1f}  |  Nuclear Density={nuclear_density:.1f}  |  "
        f"Negative Sentiment={sentiment_neg:.1f}%  |  WW3 Keywords={ww3_density:.1f}  |  Missiles={missile_density:.1f}"
    )
    ax.text(50, -0.12, comp_text, ha='center', va='top', color="#888", fontsize=7.5,
            transform=ax.transData)

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "threat_level.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ═══════════════════════════════════════════════════════
    # SECTION 8 — TEMPORAL HEATMAP (Day-of-Week × Week)
    # ═══════════════════════════════════════════════════════

    logger.info("[EDA 13/13] Weekly coverage heatmap...")
    if not df_tl.empty:
        df_heat = df.copy()
        df_heat['parsed_date'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
        df_heat = df_heat.dropna(subset=['parsed_date']).copy()
        df_heat['day_of_week'] = df_heat['parsed_date'].dt.day_name()
        df_heat['week']        = df_heat['parsed_date'].dt.isocalendar().week.astype(str)

        pivot = df_heat.groupby(['day_of_week', 'week']).size().unstack(fill_value=0)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

        if not pivot.empty and pivot.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.8), 5), facecolor=DARK_BG)
            sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, linecolor=DARK_BG,
                        ax=ax, annot=(pivot.shape[1] <= 12),
                        fmt='d', annot_kws={'size': 8},
                        cbar_kws={'shrink': 0.7})
            ax.set_title("Article Coverage Heatmap  (Day of Week × Calendar Week)",
                         fontsize=13, color=ACCENT, pad=12)
            ax.set_xlabel("Calendar Week")
            ax.set_ylabel("")
            ax.tick_params(axis='x', labelrotation=45)
            fig.tight_layout()
            fig.savefig(os.path.join(EDA_DIR, "weekly_heatmap.png"), dpi=120)
            plt.close(fig)

    logger.info("EDA execution completed successfully.")
    return stats
