import os
import sys
import argparse
import time
import schedule

# Ensure proper path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db import SQLiteDatabase
from database.vector_db import VectorDatabase
from scraper.scraper import NewsScraper
from eda_code.eda import run_eda
from eda_code.statistical_analysis import run_statistical_analysis
from eda_code.vector_eda import run_vector_eda
from eda_code.report_generator import generate_html_report
from utils import timer_logger, logger

class ETLPipeline:
    """
    Object-Oriented Manager for the Geopolitical News Extract, Transform, Load (ETL) Pipeline.
    """
    def __init__(self):
        self.newsapi_key = os.environ.get("NEWSAPI_KEY", "a752bdf354ee4625b3ed58d906eec969")
        self.sql_db = SQLiteDatabase()
        self.vector_db = VectorDatabase()
        self.scraper = NewsScraper(api_key=self.newsapi_key, db=self.sql_db)

    @timer_logger
    def run(self):
        logger.info("=============================================")
        logger.info(" Geopolitical News Pipeline / WW3 Context Analyzer ")
        logger.info("=============================================")

        # 1. EXTRACT AND LOAD (Generator pipeline)
        # Yield articles natively from scraper straight into SQLite DB
        logger.info("Scraping geopolotical articles & saving chunks via generator...")
        article_generator = self.scraper.scrape_all_sources()
        self.sql_db.save_articles(article_generator)

        # 2. VECTOR DATBASE CHUNKING (Generator pipeline)
        logger.info("Launching Vector Database Embedding Sequence via Iterator...")
        # Get iterator from SQLite DB yielding batches, feed it into vector DB which yields chunks
        articles_iterator = self.sql_db.get_all_articles_iterator()
        self.vector_db.store_articles_in_vector_db(articles_iterator)

        # 3. TRANSFORM & EXPLORATORY DATA ANALYSIS (EDA)
        logger.info("Extracting data dataframe for EDA Analysis...")
        # Fetch standard dataframe specifically for Pandas-based operations down the line
        df = self.sql_db.get_all_articles()

        logger.info("Firing up the SQLite EDA module...")
        sqlite_stats = run_eda(df)

        logger.info("Running Statistical Analysis module...")
        statistical_stats = run_statistical_analysis(df)

        logger.info("Generating Vector Search EDA Metrics...")
        vector_stats = run_vector_eda()
        
        logger.info("Building HTML Dashboard...")
        generate_html_report(sqlite_stats, vector_stats, statistical_stats)

        logger.info("=============================================")
        logger.info("           PIPELINE EXECUTION COMPLETE       ")
        logger.info("  Please open 'eda_output/report.html' to view it  ")
        logger.info("=============================================")

def run_scheduler():
    """Starts the endless loop scheduler to run the pipeline periodically."""
    pipeline = ETLPipeline()
    logger.info("Scheduling pipeline to run every 1 hour...")
    
    # Run once immediately
    pipeline.run()
    
    # Schedule repeating interval
    schedule.every(1).hours.do(pipeline.run)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Geopolitical News ETL Pipeline")
    parser.add_argument("--hourly", action="store_true", help="Run the pipeline endlessly every hour via scheduler.")
    args = parser.parse_args()

    if args.hourly:
        run_scheduler()
    else:
        # Run exactly once and exit
        pipeline = ETLPipeline()
        pipeline.run()
