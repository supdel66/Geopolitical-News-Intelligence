"""
statistical_analysis.py
=======================
Inferential & descriptive statistics module for the Geopolitical News corpus.

Hypotheses verified
-------------------
H1 (Chi-Square)  : China and USA mentions are NOT independent in the news cycle.
H2 (ANOVA)       : Mean sentiment_polarity differs significantly across the three
                   major conflict theaters (Middle East / Russia-Ukraine / Asia-Pacific).

Sections
--------
1. Descriptive Statistics  — skewness, excess kurtosis, KDE-annotated histograms
2. Correlation & Covariance — Pearson matrix + covariance matrix (heatmaps)
3. Zipf's Law              — log-rank vs log-frequency + top-word bar chart
4. OLS Regression          — escalation_score ~ word_count + sentiment_polarity
5. Hypothesis Tests        — Chi-Square, one-way ANOVA

Dependencies: scipy, statsmodels, pandas, numpy, matplotlib, seaborn
"""

import os
import sys
import string
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Statistical libraries ────────────────────────────────────────────────────
try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import statsmodels.api as sm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import timer_logger, logger

try:
    from eda_code.eda import (
        compute_escalation_score, classify_theater,
        get_sentiment_polarity, clean_text,
        CONFLICT_ACTORS,
    )
except ImportError:
    from eda import (
        compute_escalation_score, classify_theater,
        get_sentiment_polarity, clean_text,
        CONFLICT_ACTORS,
    )

# ── Visual theme (mirrors eda.py) ────────────────────────────────────────────
EDA_DIR    = "eda_output"
DARK_BG    = "#0d1117"
CARD_BG    = "#161b22"
ACCENT     = "#58a6ff"
WARN       = "#f0883e"
DANGER     = "#ff4444"
SUCCESS    = "#3fb950"
TEXT_COLOR = "#c9d1d9"
MUTED      = "#8b949e"

_THEATER_COLORS = {
    "Middle East":     "#e05252",
    "Russia/Ukraine":  "#52b0e0",
    "Asia-Pacific":    "#8b52e0",
}

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "that", "this", "it", "as", "its", "he", "she",
    "they", "we", "you", "his", "her", "their", "our", "said", "also",
    "not", "no", "into", "than", "more", "after", "over", "about",
    "will", "would", "could", "should", "which", "when", "who", "what",
    "all", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "am", "if", "up", "out", "so", "do", "did", "does",
    "can", "may", "while", "there", "then", "them", "any", "some", "new",
    "many", "other", "mr", "ms", "dr", "per", "cent", "year", "day", "time",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _setup_theme():
    sns.set_theme(style="darkgrid")
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


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all numeric and categorical features needed for the analyses."""
    df = df.copy()
    df["combined_text"] = (
        df["title"].fillna("") + " " + df["content"].fillna("")
    ).apply(clean_text)

    # Full text (unclean) for sentiment — preserves capitalisation / punctuation
    df["full_text"] = df["title"].fillna("") + " " + df["content"].fillna("")

    df["word_count"]        = df["combined_text"].apply(lambda x: len(x.split()))
    df["escalation_score"]  = df["combined_text"].apply(compute_escalation_score)
    df["sentiment_polarity"] = df["full_text"].apply(get_sentiment_polarity)
    df["theater"]           = df["combined_text"].apply(classify_theater)

    df["has_china"] = df["combined_text"].apply(
        lambda x: int(any(t in x for t in CONFLICT_ACTORS["China"]))
    )
    df["has_usa"]   = df["combined_text"].apply(
        lambda x: int(any(t in x for t in CONFLICT_ACTORS["USA"]))
    )
    return df


def _annotation_box(ax, text, x=0.97, y=0.97, ha="right", va="top",
                    fontsize=8.5, color=TEXT_COLOR):
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fontsize,
            fontfamily="monospace", color=color,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=DARK_BG,
                      edgecolor="#30363d", alpha=0.92))


def _p_box(ax, text, p_val, x=0.5, y=0.97):
    """Annotate a subplot with test result and colour-code by significance."""
    border = DANGER if p_val < 0.05 else SUCCESS
    ax.text(x, y, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.8, fontweight="bold",
            color=border,
            bbox=dict(facecolor=DARK_BG, edgecolor=border,
                      boxstyle="round,pad=0.4", alpha=0.95))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def _plot_distributions(df: pd.DataFrame, desc: dict, stats: dict):
    """Histogram + KDE + Normal reference for word_count, escalation, sentiment."""
    var_meta = [
        ("word_count",         "Word Count",          ACCENT),
        ("escalation_score",   "Escalation Score",    WARN),
        ("sentiment_polarity", "Sentiment Polarity",  DANGER),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK_BG)
    fig.suptitle(
        "Distribution Shapes — Skewness & Excess Kurtosis",
        fontsize=15, color=ACCENT, y=1.01,
    )

    for ax, (col, label, color) in zip(axes, var_meta):
        data = df[col].dropna().values
        d    = desc[col]

        # Histogram (density)
        ax.hist(data, bins=45, color=color, alpha=0.45,
                edgecolor=DARK_BG, density=True, label="Observed")

        xs = np.linspace(data.min(), data.max(), 400)

        # KDE
        kde = sp_stats.gaussian_kde(data, bw_method="scott")
        ax.plot(xs, kde(xs), color="white", linewidth=2.0, label="KDE")

        # Normal reference
        norm_pdf = sp_stats.norm.pdf(xs, data.mean(), data.std())
        ax.plot(xs, norm_pdf, color=SUCCESS, linewidth=1.5,
                linestyle="--", alpha=0.7, label="Normal (ref)")

        ax.set_facecolor(CARD_BG)
        ax.set_title(label, color=ACCENT, fontsize=12)
        ax.set_xlabel("Value", color=TEXT_COLOR)
        ax.set_ylabel("Density", color=TEXT_COLOR)
        ax.legend(fontsize=8, facecolor=CARD_BG)

        # Interpretation of skew direction
        skew_dir = "right-tailed" if d["skewness"] > 0 else "left-tailed"
        kurt_type = "leptokurtic (heavy tails)" if d["excess_kurtosis"] > 0 else "platykurtic (thin tails)"

        ann = (
            f"Skewness:      {d['skewness']:+.3f}  ({skew_dir})\n"
            f"Excess Kurt:   {d['excess_kurtosis']:+.3f}  ({kurt_type})\n"
            f"Mean:          {d['mean']:.2f}\n"
            f"Median:        {d['median']:.2f}\n"
            f"Std Dev:       {d['std']:.2f}\n"
            f"N:             {d['n']:,}"
        )
        _annotation_box(ax, ann)

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "stats_distributions.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [STATS] Saved: stats_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CORRELATION & COVARIANCE
# ─────────────────────────────────────────────────────────────────────────────

def _plot_correlation_covariance(df: pd.DataFrame, stats: dict):
    numeric_cols = ["word_count", "escalation_score", "sentiment_polarity"]
    col_labels   = ["Word Count", "Escalation\nScore", "Sentiment\nPolarity"]

    corr = df[numeric_cols].corr(method="pearson")
    cov  = df[numeric_cols].cov()

    stats["pearson_correlation"] = corr.round(4).to_dict()
    stats["covariance"]          = cov.round(4).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG)
    fig.suptitle(
        "Numeric Variable Relationships — Pearson Correlation & Covariance",
        fontsize=14, color=TEXT_COLOR, y=1.01,
    )

    # ── Correlation matrix ──
    corr_display = corr.copy()
    corr_display.index = corr_display.columns = col_labels

    sns.heatmap(
        corr_display, annot=True, fmt=".3f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=1.2, linecolor=DARK_BG,
        cbar_kws={"shrink": 0.82, "label": "Pearson r"},
        ax=axes[0], annot_kws={"size": 13, "fontweight": "bold", "color": "white"},
    )
    axes[0].set_title("Pearson Correlation Matrix", color=ACCENT, fontsize=13, pad=10)
    axes[0].tick_params(axis="x", labelrotation=0)
    axes[0].tick_params(axis="y", labelrotation=0)
    axes[0].set_facecolor(CARD_BG)

    # ── Covariance matrix ──
    cov_display = cov.copy()
    cov_display.index = cov_display.columns = col_labels

    sns.heatmap(
        cov_display, annot=True, fmt=".2f", cmap="magma",
        linewidths=1.2, linecolor=DARK_BG,
        cbar_kws={"shrink": 0.82, "label": "Covariance"},
        ax=axes[1], annot_kws={"size": 11},
    )
    axes[1].set_title("Covariance Matrix", color=ACCENT, fontsize=13, pad=10)
    axes[1].tick_params(axis="x", labelrotation=0)
    axes[1].tick_params(axis="y", labelrotation=0)
    axes[1].set_facecolor(CARD_BG)

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "stats_correlation_matrices.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [STATS] Saved: stats_correlation_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ZIPF'S LAW
# ─────────────────────────────────────────────────────────────────────────────

def _plot_zipf(df: pd.DataFrame, stats: dict):
    """
    Demonstrates Zipf's law in the text corpus:
    word frequency decays as a power law of rank.
    The fitted exponent is expected to be ≈ -1.0 for natural language.
    """
    all_text = " ".join(df["combined_text"].dropna().values)
    words    = [w for w in all_text.split() if len(w) > 2 and w not in STOP_WORDS]
    freq_ctr = Counter(words)

    top_N    = min(600, len(freq_ctr))
    top_data = freq_ctr.most_common(top_N)
    ranks    = np.arange(1, len(top_data) + 1, dtype=float)
    freqs    = np.array([f for _, f in top_data], dtype=float)

    # Power-law fit via log-log regression
    slope, intercept, r, _, _ = sp_stats.linregress(np.log(ranks), np.log(freqs))
    fitted  = np.exp(intercept) * ranks ** slope
    r_sq    = r ** 2

    stats["zipf_slope"]     = round(float(slope), 4)
    stats["zipf_r_squared"] = round(float(r_sq), 4)
    stats["top_global_keywords"] = [(w, int(f)) for w, f in freq_ctr.most_common(20)]

    logger.info(f"  Zipf slope={slope:.4f}, R²={r_sq:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=DARK_BG)
    fig.suptitle(
        "Corpus Lexical Analysis — Zipf's Law Power-Law Distribution",
        fontsize=14, color=ACCENT,
    )

    # ── Log-log scatter ──
    ax = axes[0]
    ax.scatter(ranks, freqs, color=ACCENT, alpha=0.35, s=6, label="Observed")
    ax.plot(ranks, fitted, color=DANGER, linewidth=2.5,
            label=f"Power-law fit  α={slope:.3f}  R²={r_sq:.3f}")
    ax.plot(ranks, freqs[0] / ranks, color=SUCCESS, linewidth=1.5,
            linestyle="--", alpha=0.7, label="Theoretical Zipf  α=−1.0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor(CARD_BG)
    ax.set_title("Word Frequency vs. Rank  (Log-Log Scale)", color=ACCENT, fontsize=12)
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.legend(facecolor=CARD_BG, fontsize=9)

    ann = (
        f"Power-law exponent α = {slope:.4f}\n"
        f"R² (log-log fit)     = {r_sq:.4f}\n"
        f"Theoretical Zipf α   = −1.000\n"
        f"Deviation from Zipf  = {abs(slope + 1):.4f}\n\n"
        f"A slope near −1.0 confirms\n"
        f"Zipf's law behavior in this\n"
        f"conflict news corpus."
    )
    _annotation_box(ax, ann, x=0.97, y=0.05, va="bottom")

    # ── Top 25 words bar chart ──
    ax2 = axes[1]
    top_25 = freq_ctr.most_common(25)
    wds  = [w for w, _ in top_25][::-1]
    fqs  = [f for _, f in top_25][::-1]
    palette = sns.color_palette("magma", len(wds))
    bars = ax2.barh(wds, fqs, color=palette, alpha=0.88)
    for bar, val in zip(bars, fqs):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", color=TEXT_COLOR, fontsize=7.5)
    ax2.set_facecolor(CARD_BG)
    ax2.set_title("Top 25 Most Frequent Words  (stopwords removed)",
                  color=ACCENT, fontsize=12)
    ax2.set_xlabel("Frequency")

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "stats_zipf_distribution.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [STATS] Saved: stats_zipf_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OLS REGRESSION (statsmodels)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ols(df: pd.DataFrame, stats: dict):
    """
    OLS: escalation_score ~ word_count + sentiment_polarity
    Tests whether article length and negative tone jointly predict
    how escalation-heavy the coverage is.
    """
    if not _HAS_STATSMODELS:
        logger.warning("  [STATS] statsmodels not available — OLS skipped.")
        return

    feat_df = df[["escalation_score", "word_count", "sentiment_polarity"]].dropna()
    X = sm.add_constant(feat_df[["word_count", "sentiment_polarity"]])
    y = feat_df["escalation_score"]

    model = sm.OLS(y, X).fit()

    stats["ols_regression"] = {
        "r_squared":        round(float(model.rsquared), 4),
        "adj_r_squared":    round(float(model.rsquared_adj), 4),
        "f_statistic":      round(float(model.fvalue), 4),
        "f_p_value":        round(float(model.f_pvalue), 6),
        "const_coef":       round(float(model.params["const"]), 4),
        "word_count_coef":  round(float(model.params["word_count"]), 6),
        "sentiment_coef":   round(float(model.params["sentiment_polarity"]), 4),
        "word_count_p":     round(float(model.pvalues["word_count"]), 6),
        "sentiment_p":      round(float(model.pvalues["sentiment_polarity"]), 6),
        "n_obs":            int(model.nobs),
    }

    logger.info(
        f"  OLS: R²={model.rsquared:.4f}  F={model.fvalue:.3f}  p={model.f_pvalue:.4f}"
    )

    # ── Residual diagnostic plot ──
    residuals  = model.resid
    fitted_vals = model.fittedvalues

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=DARK_BG)
    fig.suptitle(
        "OLS Regression: Escalation Score ~ Word Count + Sentiment Polarity",
        fontsize=13, color=ACCENT, y=1.01,
    )

    # Fitted vs Residuals
    axes[0].scatter(fitted_vals, residuals, alpha=0.35, s=12, color=ACCENT)
    axes[0].axhline(0, color=DANGER, linewidth=1.5, linestyle="--")
    axes[0].set_facecolor(CARD_BG)
    axes[0].set_title("Fitted Values vs. Residuals", color=ACCENT, fontsize=11)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")

    # Residual histogram
    axes[1].hist(residuals, bins=40, color=WARN, alpha=0.7, edgecolor=DARK_BG, density=True)
    xs = np.linspace(residuals.min(), residuals.max(), 300)
    axes[1].plot(xs, sp_stats.norm.pdf(xs, residuals.mean(), residuals.std()),
                 color=SUCCESS, linewidth=2, linestyle="--", label="Normal ref")
    axes[1].set_facecolor(CARD_BG)
    axes[1].set_title("Residual Distribution", color=ACCENT, fontsize=11)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    axes[1].legend(facecolor=CARD_BG, fontsize=9)

    # Coefficient plot
    coef_names = ["Intercept", "Word Count", "Sentiment\nPolarity"]
    coef_vals  = [model.params["const"],
                  model.params["word_count"],
                  model.params["sentiment_polarity"]]
    coef_errs  = [model.bse["const"],
                  model.bse["word_count"],
                  model.bse["sentiment_polarity"]]
    coef_ps    = [model.pvalues["const"],
                  model.pvalues["word_count"],
                  model.pvalues["sentiment_polarity"]]

    colors_coef = [DANGER if v < 0 else SUCCESS for v in coef_vals]
    bars = axes[2].barh(coef_names, coef_vals, xerr=coef_errs,
                        color=colors_coef, alpha=0.80,
                        error_kw={"ecolor": TEXT_COLOR, "linewidth": 1.2},
                        edgecolor=DARK_BG)
    axes[2].axvline(0, color=TEXT_COLOR, linewidth=1, linestyle="--", alpha=0.5)
    for bar, p_v in zip(bars, coef_ps):
        sig = "***" if p_v < 0.001 else "**" if p_v < 0.01 else "*" if p_v < 0.05 else "ns"
        axes[2].text(bar.get_width() + (0.001 if bar.get_width() >= 0 else -0.001),
                     bar.get_y() + bar.get_height() / 2,
                     f" {sig}  p={p_v:.4f}", va="center", color=TEXT_COLOR, fontsize=8.5)
    axes[2].set_facecolor(CARD_BG)
    axes[2].set_title(
        f"OLS Coefficients  (R²={model.rsquared:.3f}, adj R²={model.rsquared_adj:.3f})",
        color=ACCENT, fontsize=11,
    )
    axes[2].set_xlabel("Coefficient Value")

    fig.tight_layout()
    fig.savefig(os.path.join(EDA_DIR, "stats_ols_regression.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [STATS] Saved: stats_ols_regression.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — HYPOTHESIS TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_hypothesis_tests(df: pd.DataFrame, stats: dict):
    test_results = {}

    # ── TEST 1: Chi-Square — China ↔ USA co-occurrence ──────────────────────
    logger.info("  Running Test 1: Chi-Square (China vs. USA)...")
    contingency = pd.crosstab(
        pd.Series(df["has_china"].values, name="China Mentioned (1=Yes, 0=No)"),
        pd.Series(df["has_usa"].values,   name="USA Mentioned (1=Yes, 0=No)"),
    )
    for v in [0, 1]:
        if v not in contingency.index:
            contingency.loc[v] = 0
        if v not in contingency.columns:
            contingency[v] = 0
    contingency = contingency.sort_index().sort_index(axis=1)

    chi2, chi2_p, chi2_dof, chi2_exp = sp_stats.chi2_contingency(contingency)

    # Cramér's V (effect size)
    n    = contingency.values.sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    cramers_v = float(np.sqrt(phi2 / min(k - 1, r - 1))) if min(k - 1, r - 1) > 0 else 0.0

    test_results["chi_square"] = {
        "test":            "Chi-Square Test of Independence (China vs. USA co-occurrence)",
        "chi2_statistic":  round(float(chi2), 4),
        "p_value":         round(float(chi2_p), 6),
        "degrees_of_freedom": int(chi2_dof),
        "cramers_v":       round(cramers_v, 4),
        "contingency_table": contingency.to_dict(),
        "conclusion": (
            "REJECT H0 — China and USA mentions are NOT independent in the news cycle"
            if chi2_p < 0.05 else
            "FAIL TO REJECT H0 — No significant dependency between China and USA mentions"
        ),
        "interpretation": (
            f"The Chi-Square test (χ²={chi2:.3f}, df={chi2_dof}, p={chi2_p:.4f}) shows a "
            f"statistically {'significant' if chi2_p < 0.05 else 'non-significant'} association "
            f"between China and USA mentions (Cramér's V = {cramers_v:.3f}). "
            + ("Articles tend to mention both nations together more often than chance "
               "would predict, suggesting they are framed as co-protagonists in the same geopolitical storylines."
               if chi2_p < 0.05 else
               "No systematic co-occurrence pattern was detected — the two actors are covered independently.")
        ),
    }
    logger.info(
        f"    χ²={chi2:.4f}  p={chi2_p:.6f}  V={cramers_v:.4f}  "
        f"→ {test_results['chi_square']['conclusion']}"
    )

    # ── TEST 2: One-Way ANOVA — Sentiment across theaters ───────────────────
    logger.info("  Running Test 2: One-Way ANOVA (theater sentiment)...")
    theater_names = ["Middle East", "Russia/Ukraine", "Asia-Pacific"]
    theater_groups = {
        t: df[df["theater"] == t]["sentiment_polarity"].dropna()
        for t in theater_names
    }
    theater_groups = {t: g for t, g in theater_groups.items() if len(g) >= 2}

    if len(theater_groups) >= 2:
        f_stat, anova_p = sp_stats.f_oneway(*theater_groups.values())
    else:
        f_stat, anova_p = 0.0, 1.0

    group_means = {t: round(float(g.mean()), 4) for t, g in theater_groups.items()}
    group_stds  = {t: round(float(g.std()), 4)  for t, g in theater_groups.items()}
    group_ns    = {t: int(len(g))               for t, g in theater_groups.items()}

    # Eta-squared (effect size): SS_between / SS_total
    if len(theater_groups) >= 2:
        grand_mean = df[df["theater"].isin(theater_groups.keys())]["sentiment_polarity"].mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for g in theater_groups.values()
        )
        all_vals = pd.concat(list(theater_groups.values()))
        ss_total = float(((all_vals - grand_mean) ** 2).sum())
        eta_sq   = float(ss_between / ss_total) if ss_total > 0 else 0.0
    else:
        eta_sq = 0.0

    test_results["anova"] = {
        "test":         "One-Way ANOVA — Sentiment Polarity across Conflict Theaters",
        "f_statistic":  round(float(f_stat), 4),
        "p_value":      round(float(anova_p), 6),
        "eta_squared":  round(eta_sq, 4),
        "group_means":  group_means,
        "group_stds":   group_stds,
        "group_ns":     group_ns,
        "conclusion": (
            "REJECT H0 — Sentiment polarity differs significantly across conflict theaters"
            if anova_p < 0.05 else
            "FAIL TO REJECT H0 — No significant sentiment difference across theaters"
        ),
        "interpretation": (
            f"One-way ANOVA (F={f_stat:.3f}, p={anova_p:.4f}, η²={eta_sq:.3f}) "
            + ("reveals" if anova_p < 0.05 else "does not reveal") +
            " statistically significant differences in news sentiment across theaters. "
            + (f"The variance explained by theater grouping is {eta_sq*100:.1f}% (η²)."
               if anova_p < 0.05 else
               "All three theaters receive similarly toned coverage in this corpus.")
        ),
    }
    logger.info(
        f"    F={f_stat:.4f}  p={anova_p:.6f}  η²={eta_sq:.4f}  "
        f"→ {test_results['anova']['conclusion']}"
    )

    stats["hypothesis_tests"] = test_results

    # ── Combined Hypothesis Test Visualization ───────────────────────────────
    _plot_hypothesis_panel(df, contingency, theater_groups, test_results)


def _plot_hypothesis_panel(df, contingency, theater_groups, test_results):
    """Four-panel figure: visualisation + interpretation card for each test."""
    chi_res   = test_results["chi_square"]
    anova_res = test_results["anova"]

    fig = plt.figure(figsize=(14, 14), facecolor=DARK_BG)
    fig.suptitle(
        "Statistical Hypothesis Testing  —  Geopolitical News Corpus",
        fontsize=16, color=ACCENT, y=0.99,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.36)

    # ── Row 0: Visualisations ────────────────────────────────────────────────

    # Col 0: Chi-Square stacked bar
    ax_chi = fig.add_subplot(gs[0, 0])
    chi_df = pd.DataFrame(
        contingency.values,
        index=["No China", "China\nMentioned"],
        columns=["No USA", "USA\nMentioned"],
    )
    chi_df.plot(kind="bar", ax=ax_chi, color=[ACCENT, DANGER],
                alpha=0.82, edgecolor=DARK_BG, width=0.6)
    ax_chi.set_facecolor(CARD_BG)
    ax_chi.set_title("Test 1 — Chi-Square\nChina × USA Co-occurrence",
                     color=ACCENT, fontsize=11)
    ax_chi.set_xlabel("China Mention", color=TEXT_COLOR)
    ax_chi.set_ylabel("Article Count", color=TEXT_COLOR)
    ax_chi.tick_params(axis="x", rotation=0)
    ax_chi.legend(facecolor=CARD_BG, fontsize=8)
    _p_box(ax_chi,
           f"χ²={chi_res['chi2_statistic']:.3f}   p={chi_res['p_value']:.4f}\n"
           f"Cramér's V={chi_res['cramers_v']:.3f}",
           chi_res["p_value"])

    # Col 1: ANOVA boxplot
    ax_an = fig.add_subplot(gs[0, 1])
    if theater_groups:
        an_recs = [
            {"Theater": t, "Sentiment": float(v)}
            for t, grp in theater_groups.items()
            for v in grp
        ]
        an_df = pd.DataFrame(an_recs)
        sns.boxplot(
            data=an_df, x="Theater", y="Sentiment", ax=ax_an,
            hue="Theater",
            palette=_THEATER_COLORS, linewidth=1.2, legend=False,
            flierprops=dict(marker="o", markersize=3, alpha=0.45),
        )
        ax_an.axhline(0, color=TEXT_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
    ax_an.set_facecolor(CARD_BG)
    ax_an.set_title("Test 2 — One-Way ANOVA\nSentiment by Conflict Theater",
                    color=ACCENT, fontsize=11)
    ax_an.set_xlabel("")
    ax_an.set_ylabel("Sentiment Polarity", color=TEXT_COLOR)
    ax_an.tick_params(axis="x", labelrotation=12)
    _p_box(ax_an,
           f"F={anova_res['f_statistic']:.3f}   p={anova_res['p_value']:.4f}\n"
           f"η²={anova_res['eta_squared']:.3f}",
           anova_res["p_value"])

    # ── Row 1: Interpretation cards ─────────────────────────────────────────

    for col_idx, (title, res) in enumerate([
        ("Test 1 — Chi-Square  (China ↔ USA)", chi_res),
        ("Test 2 — One-Way ANOVA  (Theater Sentiment)", anova_res),
    ]):
        ax_c = fig.add_subplot(gs[1, col_idx])
        ax_c.set_facecolor(CARD_BG)
        ax_c.axis("off")

        p_val  = res["p_value"]
        border = DANGER if p_val < 0.05 else SUCCESS
        sig_txt = (
            "★  STATISTICALLY SIGNIFICANT  (p < 0.05)"
            if p_val < 0.05 else
            "○  NOT SIGNIFICANT  (p ≥ 0.05)"
        )
        card = (
            f"{title}\n"
            f"{'─' * 62}\n"
            f"{res['conclusion']}\n\n"
            f"{res['interpretation']}\n\n"
            f"{sig_txt}"
        )
        ax_c.text(
            0.04, 0.97, card,
            transform=ax_c.transAxes,
            va="top", ha="left", fontsize=8.4,
            color=TEXT_COLOR, wrap=True,
            bbox=dict(facecolor=DARK_BG, edgecolor=border,
                      boxstyle="round,pad=0.55", alpha=0.93),
        )

    fig.savefig(os.path.join(EDA_DIR, "stats_hypothesis_tests.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  [STATS] Saved: stats_hypothesis_tests.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

@timer_logger
def run_statistical_analysis(df: pd.DataFrame) -> dict:
    """
    Run the complete statistical analysis suite.

    Parameters
    ----------
    df : pd.DataFrame
        The raw articles DataFrame (same as passed to run_eda).

    Returns
    -------
    dict
        Computed statistics and test results, suitable for embedding in
        the HTML report.
    """
    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)

    if df is None or df.empty:
        logger.warning("[STATS] Empty dataframe — statistical analysis skipped.")
        return {}

    if not _HAS_SCIPY:
        logger.error(
            "[STATS] scipy is not installed.  "
            "Run: pip install scipy statsmodels  then retry."
        )
        return {}

    _setup_theme()
    stats: dict = {}

    logger.info("[STATS] Feature engineering...")
    df = _prepare_features(df)

    # ── Section 1: Descriptive statistics ───────────────────────────────────
    logger.info("[STATS 1/4] Descriptive statistics (skewness, kurtosis)...")
    var_cols = ["word_count", "escalation_score", "sentiment_polarity"]
    desc: dict = {}
    for col in var_cols:
        data = df[col].dropna().values
        desc[col] = {
            "n":               int(len(data)),
            "mean":            round(float(data.mean()), 4),
            "median":          round(float(np.median(data)), 4),
            "std":             round(float(data.std(ddof=1)), 4),
            "min":             round(float(data.min()), 4),
            "max":             round(float(data.max()), 4),
            "skewness":        round(float(sp_stats.skew(data)), 4),
            "excess_kurtosis": round(float(sp_stats.kurtosis(data, fisher=True)), 4),
        }
        logger.info(
            f"  {col}: skew={desc[col]['skewness']:.4f}  "
            f"ex_kurt={desc[col]['excess_kurtosis']:.4f}"
        )
    stats["descriptive"] = desc
    _plot_distributions(df, desc, stats)

    # ── Section 2: Correlation & covariance ─────────────────────────────────
    logger.info("[STATS 2/4] Correlation and covariance matrices...")
    _plot_correlation_covariance(df, stats)

    # ── Section 3: Zipf's law ────────────────────────────────────────────────
    logger.info("[STATS 3/4] Zipf's Law analysis...")
    _plot_zipf(df, stats)

    # ── Section 4: OLS regression ────────────────────────────────────────────
    logger.info("[STATS 4a] OLS regression (statsmodels)...")
    _run_ols(df, stats)

    # ── Section 5: Hypothesis tests ──────────────────────────────────────────
    logger.info("[STATS 4b/4] Hypothesis testing (Chi-Square, ANOVA)...")
    _run_hypothesis_tests(df, stats)

    logger.info("[STATS] Statistical analysis complete. 5 charts saved to eda_output/.")
    return stats
