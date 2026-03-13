import os
from datetime import datetime
from utils import timer_logger, logger

EDA_DIR = "eda_output"

_CHART = lambda name: f'<img src="{name}" alt="{name}" onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'block\'"><p class="warn-missing">[Chart unavailable: {name}]</p>'


@timer_logger
def generate_html_report(sqlite_stats, vector_stats, statistical_stats=None):
    """Generates a dark-themed HTML intelligence dashboard for all EDA outputs."""
    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)

    report_path = os.path.join(EDA_DIR, "report.html")

    # ── Stats
    total_articles  = sqlite_stats.get("total_articles", 0)
    top_sources     = sqlite_stats.get("top_sources", {})
    mean_words      = sqlite_stats.get("mean_words", 0.0)
    median_words    = sqlite_stats.get("median_words", 0.0)
    threat_level    = sqlite_stats.get("ww3_threat_level", 0.0)
    avg_sent        = sqlite_stats.get("avg_sentiment", 0.0)
    dom_theater     = sqlite_stats.get("dominant_theater", "N/A")
    actor_mentions  = sqlite_stats.get("actor_mentions", {})
    sent_dist       = sqlite_stats.get("sentiment_distribution", {})
    theaters        = sqlite_stats.get("conflict_theaters", {})
    top_bigrams     = sqlite_stats.get("top_bigrams", [])
    top_trigrams    = sqlite_stats.get("top_trigrams", [])
    avg_esc         = sqlite_stats.get("avg_escalation_score", 0.0)

    # Vector stats
    theme_counts    = (vector_stats or {}).get("theme_counts", {})
    theme_dists     = (vector_stats or {}).get("theme_distances", {})

    # Statistical analysis stats
    st              = statistical_stats or {}
    desc_stats      = st.get("descriptive", {})
    ols             = st.get("ols_regression", {})
    hyp             = st.get("hypothesis_tests", {})
    chi_res         = hyp.get("chi_square", {})
    anova_res       = hyp.get("anova", {})

    # ── Dynamic threat colour
    if threat_level < 25:   threat_color = "#3fb950"
    elif threat_level < 50: threat_color = "#f0883e"
    elif threat_level < 75: threat_color = "#e05252"
    else:                   threat_color = "#ff4444"
    threat_label = (
        "LOW"      if threat_level < 25 else
        "MODERATE" if threat_level < 50 else
        "HIGH"     if threat_level < 75 else
        "CRITICAL"
    )

    # ── Top Sources Table
    sources_rows = "".join(
        f"<tr><td>{src}</td><td>{cnt}</td></tr>"
        for src, cnt in list(top_sources.items())[:10]
    )

    # ── Actor Mentions Table
    actor_rows = "".join(
        f"<tr><td>{actor}</td><td>{count:,}</td></tr>"
        for actor, count in list(actor_mentions.items())[:12]
    )

    # ── Theater Distribution Table
    theater_rows = "".join(
        f"<tr><td>{theater}</td><td>{count}</td></tr>"
        for theater, count in theaters.items()
    )

    # ── Sentiment Badge HTML
    pos_pct  = sent_dist.get("Positive", 0)
    neu_pct  = sent_dist.get("Neutral",  0)
    neg_pct  = sent_dist.get("Negative", 0)

    # ── N-gram tables
    bigram_rows = "".join(
        f"<tr><td>{phrase}</td><td>{cnt:,}</td></tr>"
        for phrase, cnt in top_bigrams
    )
    trigram_rows = "".join(
        f"<tr><td>{phrase}</td><td>{cnt:,}</td></tr>"
        for phrase, cnt in top_trigrams
    )

    # ── Vector theme table
    theme_rows = "".join(
        f"<tr><td>{theme}</td>"
        f"<td>{theme_counts.get(theme, 0)}</td>"
        f"<td>{theme_dists.get(theme, 0):.4f}</td></tr>"
        for theme in theme_counts
    )

    # ── Statistical analysis — descriptive table
    def _fmt(val, decimals=4):
        try:
            return f"{float(val):.{decimals}f}"
        except Exception:
            return str(val)

    desc_rows = "".join(
        f"<tr>"
        f"<td>{col.replace('_', ' ').title()}</td>"
        f"<td>{_fmt(d.get('n', 0), 0)}</td>"
        f"<td>{_fmt(d.get('mean', 0))}</td>"
        f"<td>{_fmt(d.get('median', 0))}</td>"
        f"<td>{_fmt(d.get('std', 0))}</td>"
        f"<td>{_fmt(d.get('skewness', 0))}</td>"
        f"<td>{_fmt(d.get('excess_kurtosis', 0))}</td>"
        f"</tr>"
        for col, d in desc_stats.items()
    ) if desc_stats else '<tr><td colspan="7" style="color:var(--muted)">Stats not computed</td></tr>'

    def _sig_badge(p):
        if p < 0.001:
            return '<span style="color:#ff4444;font-weight:700">★★★ p&lt;0.001</span>'
        if p < 0.01:
            return '<span style="color:#ff4444;font-weight:700">★★ p&lt;0.01</span>'
        if p < 0.05:
            return '<span style="color:#f0883e;font-weight:700">★ p&lt;0.05</span>'
        return '<span style="color:#3fb950">ns  p≥0.05</span>'

    chi_p    = chi_res.get("p_value", 1.0)
    anova_p  = anova_res.get("p_value", 1.0)

    ols_r2        = _fmt(ols.get("r_squared", 0))
    ols_adj_r2    = _fmt(ols.get("adj_r_squared", 0))
    ols_f         = _fmt(ols.get("f_statistic", 0), 3)
    ols_fp        = _fmt(ols.get("f_p_value", 1), 4)
    ols_wc_coef   = _fmt(ols.get("word_count_coef", 0), 6)
    ols_sent_coef = _fmt(ols.get("sentiment_coef", 0), 4)
    ols_n         = ols.get("n_obs", "N/A")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WW3 Geopolitical Intelligence Dashboard</title>
<style>
  :root {{
    --bg:      #0d1117;
    --card:    #161b22;
    --border:  #30363d;
    --text:    #c9d1d9;
    --muted:   #8b949e;
    --accent:  #58a6ff;
    --green:   #3fb950;
    --orange:  #f0883e;
    --red:     #ff4444;
    --purple:  #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px; line-height: 1.6; padding: 20px;
  }}

  /* ── Header */
  .header {{
    text-align: center; padding: 28px 0 20px;
    border-bottom: 1px solid var(--border); margin-bottom: 28px;
  }}
  .header h1 {{ font-size: 26px; color: var(--accent); letter-spacing: 1px; }}
  .header p  {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}

  /* ── Threat banner */
  .threat-banner {{
    border: 2px solid {threat_color};
    border-radius: 10px; padding: 16px 24px;
    margin-bottom: 28px; text-align: center;
    background: {threat_color}18;
  }}
  .threat-value {{ font-size: 52px; font-weight: 900; color: {threat_color}; line-height: 1; }}
  .threat-label {{ font-size: 18px; font-weight: 700; color: {threat_color}; letter-spacing: 3px; }}
  .threat-sub   {{ color: var(--muted); font-size: 12px; margin-top: 6px; }}

  /* ── Metrics row */
  .metrics {{ display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 28px; }}
  .metric  {{
    flex: 1 1 160px; background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; text-align: center;
  }}
  .metric-value {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
  .metric-label {{ font-size: 11px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}

  /* ── Sentiment bar */
  .sent-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 8px 0; }}
  .sent-pos {{ background: var(--green);  flex: {pos_pct}; }}
  .sent-neu {{ background: var(--orange); flex: {neu_pct}; }}
  .sent-neg {{ background: var(--red);    flex: {neg_pct}; }}

  /* ── Section headings */
  h2 {{
    font-size: 16px; color: #03dac6; border-bottom: 1px solid var(--border);
    padding-bottom: 6px; margin: 30px 0 16px;
    text-transform: uppercase; letter-spacing: 1px;
  }}
  h3 {{ font-size: 13px; color: var(--accent); margin-bottom: 10px; }}

  /* ── Grid layouts */
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; margin-bottom: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin-bottom: 20px; }}
  .grid-full {{ margin-bottom: 20px; }}

  /* ── Cards */
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 18px; }}
  .card img {{ max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto; }}
  .warn-missing {{ display: none; color: var(--orange); font-size: 12px; margin-top: 6px; }}

  /* ── Tables */
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--accent); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }}
  tr:last-child td {{ border-bottom: none; }}

  /* ── Tags */
  .tag {{
    display: inline-block; padding: 3px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin: 2px;
  }}
  .tag-pos {{ background: #3fb95030; color: var(--green); border: 1px solid var(--green); }}
  .tag-neg {{ background: #ff444430; color: var(--red);   border: 1px solid var(--red);   }}
  .tag-neu {{ background: #f0883e30; color: var(--orange);border: 1px solid var(--orange);}}

  .footer {{ text-align: center; margin-top: 40px; color: var(--muted); font-size: 11px;
             border-top: 1px solid var(--border); padding-top: 14px; }}
  code {{ background: #21262d; padding: 2px 5px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>

<!-- ═══ HEADER ═══ -->
<div class="header">
  <h1>WW3 GEOPOLITICAL INTELLIGENCE DASHBOARD</h1>
  <p>Automated ETL + EDA Pipeline  |  {total_articles:,} articles analysed  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>

<!-- ═══ THREAT LEVEL BANNER ═══ -->
<div class="threat-banner">
  <div class="threat-value">{threat_level}</div>
  <div class="threat-label">WW3 THREAT LEVEL — {threat_label}</div>
  <div class="threat-sub">
    Composite score from: escalation keywords, nuclear density, negative sentiment, WW3 phrase frequency, missile mentions.
    Scale: 0 (no threat) → 100 (critical).
  </div>
</div>

<!-- ═══ TOP METRICS ═══ -->
<div class="metrics">
  <div class="metric">
    <div class="metric-value">{total_articles:,}</div>
    <div class="metric-label">Articles Analysed</div>
  </div>
  <div class="metric">
    <div class="metric-value">{len(top_sources)}</div>
    <div class="metric-label">Unique Sources</div>
  </div>
  <div class="metric">
    <div class="metric-value">{mean_words:.0f}</div>
    <div class="metric-label">Avg Word Count</div>
  </div>
  <div class="metric">
    <div class="metric-value">{avg_esc:.1f}</div>
    <div class="metric-label">Avg Escalation Score</div>
  </div>
  <div class="metric">
    <div class="metric-value" style="color: {'var(--green)' if avg_sent >= 0 else 'var(--red)'}">
      {avg_sent:+.3f}
    </div>
    <div class="metric-label">Avg Sentiment Polarity</div>
  </div>
  <div class="metric">
    <div class="metric-value" style="color: var(--purple)">{dom_theater}</div>
    <div class="metric-label">Dominant Theater</div>
  </div>
</div>

<!-- Sentiment bar -->
<div class="card" style="margin-bottom:20px;">
  <h3>Sentiment Distribution</h3>
  <div class="sent-bar">
    <div class="sent-pos"></div>
    <div class="sent-neu"></div>
    <div class="sent-neg"></div>
  </div>
  <div style="display:flex; gap:14px; margin-top:6px; font-size:12px;">
    <span class="tag tag-pos">Positive {pos_pct:.1f}%</span>
    <span class="tag tag-neu">Neutral {neu_pct:.1f}%</span>
    <span class="tag tag-neg">Negative {neg_pct:.1f}%</span>
  </div>
</div>

<!-- ═══ SECTION 1 — BASIC OVERVIEW ═══ -->
<h2>Section 1 — Data Overview</h2>
<div class="grid-full card">
  {_CHART("threat_level.png")}
</div>

<div class="grid-2">
  <div class="card">
    <h3>Top 10 News Sources</h3>
    {_CHART("top_sources.png")}
    <table>
      <tr><th>Source</th><th>Articles</th></tr>
      {sources_rows}
    </table>
  </div>
  <div class="card">
    <h3>Conflict Keyword Frequencies</h3>
    {_CHART("top_keywords.png")}
  </div>
</div>

<div class="grid-2">
  <div class="card">
    <h3>Article Word Count Distribution</h3>
    <p style="color:var(--muted); font-size:12px; margin-bottom:8px;">
      Mean: <b>{mean_words:.0f}</b> words  |  Median: <b>{median_words:.0f}</b> words
    </p>
    {_CHART("article_length_dist.png")}
  </div>
  <div class="card">
    <h3>Publication Timeline</h3>
    {_CHART("articles_over_time.png")}
  </div>
</div>

<!-- ═══ SECTION 2 — GEOPOLITICAL ACTOR INTELLIGENCE ═══ -->
<h2>Section 2 — Geopolitical Actor Intelligence</h2>

<div class="grid-2">
  <div class="card">
    <h3>Actor Mention Frequency</h3>
    {_CHART("country_actor_mentions.png")}
    <table>
      <tr><th>Actor / Country</th><th>Total Mentions</th></tr>
      {actor_rows}
    </table>
  </div>
  <div class="card">
    <h3>Conflict Theater Distribution</h3>
    {_CHART("conflict_theaters.png")}
    <table>
      <tr><th>Theater</th><th>Articles</th></tr>
      {theater_rows}
    </table>
  </div>
</div>

<div class="grid-full card">
  <h3>Actor Co-Occurrence Heatmap</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Counts how many articles mention two actors simultaneously — reveals key conflict pairs and alliances.
  </p>
  {_CHART("actor_cooccurrence_heatmap.png")}
</div>

<!-- ═══ SECTION 3 — ESCALATION & SENTIMENT ═══ -->
<h2>Section 3 — Escalation Scoring &amp; Sentiment Analysis</h2>

<div class="grid-full card">
  <h3>Escalation Score — Distribution &amp; Trend</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Each article receives a weighted keyword score (nuclear=10, missile=5, ceasefire=−3 …).
    High scores indicate more intense conflict language. Rolling 3-day average shows trajectory.
    Avg score: <b>{avg_esc:.2f} / 100</b>
  </p>
  {_CHART("escalation_trend.png")}
</div>

<div class="grid-full card">
  <h3>Sentiment Analysis — By Source &amp; Over Time</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Hybrid conflict-domain sentiment scorer: primary score from a curated geopolitical lexicon
    (conflict words weighted −1→−5; ceasefire/peace +2→+5), blended 70:30 with TextBlob.
    Polarity: −1 (strongly negative) → +1 (strongly positive).
    Left: Median polarity per source. Right: Daily rolling sentiment trend.
    Overall avg: <b>{avg_sent:+.3f}</b>
  </p>
  {_CHART("sentiment_analysis.png")}
</div>

<!-- ═══ SECTION 4 — PUBLICATION VELOCITY ═══ -->
<h2>Section 4 — Source Publication Velocity</h2>

<div class="grid-full card">
  <h3>Stacked Area — Articles per Source per Day</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Shows which sources are most active on any given day and how overall volume evolves.
  </p>
  {_CHART("source_velocity.png")}
</div>

<div class="grid-full card">
  <h3>Weekly Coverage Heatmap  (Day of Week × Calendar Week)</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Reveals editorial cycles — which days see the most conflict reporting and week-by-week intensity shifts.
  </p>
  {_CHART("weekly_heatmap.png")}
</div>

<!-- ═══ SECTION 5 — NLP PHRASE INTELLIGENCE ═══ -->
<h2>Section 5 — NLP Phrase Intelligence (N-grams)</h2>

<div class="grid-full card">
  <h3>Top Bigrams &amp; Trigrams</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Most frequent multi-word phrases across all articles after English stopword removal.
    Reveals recurring events, weapon systems, and diplomatic terms.
  </p>
  {_CHART("top_ngrams.png")}
</div>

<div class="grid-2">
  <div class="card">
    <h3>Top Bigrams</h3>
    <table>
      <tr><th>Phrase</th><th>Count</th></tr>
      {bigram_rows if bigram_rows else '<tr><td colspan="2" style="color:var(--muted)">N/A</td></tr>'}
    </table>
  </div>
  <div class="card">
    <h3>Top Trigrams</h3>
    <table>
      <tr><th>Phrase</th><th>Count</th></tr>
      {trigram_rows if trigram_rows else '<tr><td colspan="2" style="color:var(--muted)">N/A</td></tr>'}
    </table>
  </div>
</div>

<!-- ═══ SECTION 6 — STATISTICAL ANALYSIS ═══ -->
<h2>Section 6 — Statistical Analysis  (Descriptive + Inferential)</h2>
<p style="color:var(--muted); font-size:12px; margin-bottom:14px;">
  Rigorous statistical analysis using <code>scipy.stats</code>, <code>statsmodels</code>, and <code>pandas</code>.
  Verifies two key hypotheses about the conflict news corpus.
</p>

<!-- Descriptive stats table -->
<div class="grid-full card">
  <h3>1A — Descriptive Statistics  (Skewness &amp; Excess Kurtosis)</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Excess kurtosis &gt; 0 = leptokurtic (heavy tails); &lt; 0 = platykurtic.
    Skewness &gt; 0 = right-tailed; &lt; 0 = left-tailed.
  </p>
  {_CHART("stats_distributions.png")}
  <table style="margin-top:14px;">
    <tr>
      <th>Variable</th><th>N</th><th>Mean</th><th>Median</th>
      <th>Std Dev</th><th>Skewness</th><th>Excess Kurtosis</th>
    </tr>
    {desc_rows}
  </table>
</div>

<!-- Correlation / Covariance -->
<div class="grid-full card">
  <h3>1B — Pearson Correlation &amp; Covariance Matrices</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Pearson r measures linear dependency between numeric variables.
    Covariance captures the raw joint variability (scale-dependent).
  </p>
  {_CHART("stats_correlation_matrices.png")}
</div>

<!-- Zipf's Law -->
<div class="grid-full card">
  <h3>1C — Zipf's Law  (Global Keyword Frequency Distribution)</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Natural language corpora follow Zipf's law: frequency ∝ rank<sup>−1</sup>.
    A fitted slope near −1.0 on the log-log plot confirms this corpus obeys the power law,
    validating that the text is representative natural language rather than fabricated or
    heavily templated content.
  </p>
  {_CHART("stats_zipf_distribution.png")}
</div>

<!-- OLS Regression -->
<div class="grid-full card">
  <h3>1D — OLS Regression  (Escalation Score ~ Word Count + Sentiment)</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    Tests whether article length and negativity jointly predict escalation intensity.
    R² = <b>{ols_r2}</b>  |  Adj R² = <b>{ols_adj_r2}</b>  |
    F = <b>{ols_f}</b>  (p = {ols_fp})  |  N = {ols_n}
  </p>
  {_CHART("stats_ols_regression.png")}
  <table style="margin-top:12px; max-width:500px;">
    <tr><th>Term</th><th>Coefficient</th><th>Interpretation</th></tr>
    <tr><td>Word Count</td><td>{ols_wc_coef}</td>
        <td>Change in escalation per additional word</td></tr>
    <tr><td>Sentiment Polarity</td><td>{ols_sent_coef}</td>
        <td>More negative sentiment → higher escalation score</td></tr>
  </table>
</div>

<!-- Hypothesis Tests -->
<div class="grid-full card">
  <h3>2 — Hypothesis Testing  (Chi-Square · One-Way ANOVA)</h3>
  <p style="color:var(--muted); font-size:12px; margin-bottom:10px;">
    All tests use α = 0.05. Two-tailed where applicable.
  </p>
  {_CHART("stats_hypothesis_tests.png")}
  <table style="margin-top:14px;">
    <tr><th>Hypothesis</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Effect Size</th><th>Conclusion</th></tr>
    <tr>
      <td>H1: China &amp; USA mentions are independent</td>
      <td>Chi-Square</td>
      <td>χ² = {_fmt(chi_res.get('chi2_statistic', 0), 3)}</td>
      <td>{_sig_badge(chi_p)}</td>
      <td>Cramér's V = {_fmt(chi_res.get('cramers_v', 0), 3)}</td>
      <td>{chi_res.get('conclusion', 'N/A')}</td>
    </tr>
    <tr>
      <td>H2: Sentiment differs across conflict theaters</td>
      <td>One-Way ANOVA</td>
      <td>F = {_fmt(anova_res.get('f_statistic', 0), 3)}</td>
      <td>{_sig_badge(anova_p)}</td>
      <td>η² = {_fmt(anova_res.get('eta_squared', 0), 3)}</td>
      <td>{anova_res.get('conclusion', 'N/A')}</td>
    </tr>
  </table>
</div>

<!-- ═══ SECTION 7 — CHROMADB SEMANTIC ANALYSIS ═══ -->
<h2>Section 7 — Semantic Vector Analysis  (ChromaDB + Ollama Embeddings)</h2>
<p style="color:var(--muted); font-size:12px; margin-bottom:14px;">
  Articles were chunked and embedded via <code>nomic-embed-text</code>  (local Ollama).
  Themes are queried by semantic similarity — distance threshold ≤ 1.0.
</p>

<div class="grid-2">
  <div class="card">
    <h3>Thematic Volume — Unique Articles per Theme</h3>
    {_CHART("vector_themes_volume.png")}
  </div>
  <div class="card">
    <h3>Average Semantic Distance per Theme</h3>
    <p style="color:var(--muted); font-size:12px; margin-bottom:8px;">Lower distance = closer semantic match.</p>
    {_CHART("vector_themes_relevance.png")}
  </div>
</div>

{'<div class="card"><h3>Semantic Theme Summary</h3><table><tr><th>Theme</th><th>Matching Articles</th><th>Avg Distance</th></tr>' + theme_rows + '</table></div>' if theme_rows else ''}

<!-- ═══ FOOTER ═══ -->
<div class="footer">
  WW3 Geopolitical Intelligence Pipeline  |  Sources: RSS Feeds + NewsAPI
  |  NLP: Domain Lexicon + scikit-learn  |  Stats: scipy + statsmodels  |  Vectors: ChromaDB + Ollama
  |  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>

</body>
</html>"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"[Report] HTML dashboard generated at: {report_path}")
