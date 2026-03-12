import os
import sys

# Ensure proper path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db import init_db, save_articles, get_all_articles
from scraper.scraper import scrape_all_sources
from eda_code.eda import run_eda
from eda_code.statistical_analysis import run_statistical_analysis
from database.vector_db import store_articles_in_vector_db
from eda_code.vector_eda import run_vector_eda
from eda_code.report_generator import generate_html_report
from utils import timer_logger, logger

@timer_logger
def main():
    logger.info("=============================================")
    logger.info(" Geopolitical News Pipeline / WW3 Context Analyzer ")
    logger.info("=============================================")

    # Initialize SQLite Database
    logger.info("Initializing SQLite database...")
    init_db()

    # Define API keys and settings
    newsapi_key = os.environ.get("NEWSAPI_KEY", "a752bdf354ee4625b3ed58d906eec969")
    
    # # Scrape News
    logger.info("Scraping recent geopolitical articles...")
    articles = scrape_all_sources(api_key=newsapi_key)
    logger.info(f"Extracted {len(articles)} relevant articles.")

    # Save to Database
    logger.info("Saving articles to the database...")
    save_articles(articles)

    # Fetch from DB to run EDA
    logger.info("Extracting data from database for analysis...")
    df = get_all_articles()

    # Generate EDA
    logger.info("Firing up the SQLite EDA module...")
    sqlite_stats = run_eda(df)

    # Run Statistical Analysis (descriptive stats + hypothesis testing)
    logger.info("Running Statistical Analysis module...")
    statistical_stats = run_statistical_analysis(df)

    # Embed and Store in Vector Database
    logger.info("Launching Vector Database Embedding Sequence...")
    store_articles_in_vector_db(df)
    
    # Generate Vector DB EDA
    logger.info("Generating Vector Search EDA Metrics...")
    vector_stats = run_vector_eda()
    
    # Generate Final Report
    logger.info("Building HTML Dashboard...")
    generate_html_report(sqlite_stats, vector_stats, statistical_stats)

    logger.info("=============================================")
    logger.info("           PIPELINE EXECUTION COMPLETE       ")
    logger.info("  Please open 'eda_output/report.html' to view it  ")
    logger.info("=============================================")

if __name__ == "__main__":
    main()
