# eda_sqlite_piyush outputs

These files are generated from `sqlite_databases/news.db` (table: `articles`).

## Charts

- `bar_articles_per_source.png`: Bar chart of top sources by article count.
- `pie_source_distribution.png`: Pie chart of source distribution (labels moved to legend to avoid overlap; tiny slices hide % text).
- `line_articles_over_time.png`: Daily article counts over time.
- `hist_publishing_frequen9cy.png`: Histogram of daily article counts (publishing frequency).
- `bar_peak_hours.png`: Articles by hour-of-day (UTC).

## CSV files

- `eda_summary.csv`: Key summary metrics (rows, uniques, duplicates, parseable dates) and optional ANOVA/forecast status.
- `similar_titles_sample.csv`: Sample of near-duplicate title pairs (indexes `i`/`j` refer to row positions in the loaded dataframe).
- `forecast_next_7_days.csv`: Simple 7-day forecast (moving-average + weekday baseline).
