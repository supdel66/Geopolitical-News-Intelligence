import os
import sys

# Ensure proper path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db import init_db, save_articles, get_all_articles
from scraper.scraper import scrape_all_sources
from eda_code.eda import run_eda
from database.vector_db import store_articles_in_vector_db

def main():
    print("=====================================================")
    print(" Geopolitical News Pipeline / WW3 Context Analyzer ")
    print("=====================================================")

    # Initialize SQLite Database
    print("\n[INFO] Initializing SQLite database...")
    init_db()

    # Define API keys and settings
    newsapi_key = os.environ.get("NEWSAPI_KEY", "a752bdf354ee4625b3ed58d906eec969") # Adding the default provided in scripts for convenience
    
    # Scrape News
    print("\n[INFO] Scraping recent geopolitical articles...")
    articles = scrape_all_sources(api_key=newsapi_key)
    print(f"-> Extracted {len(articles)} relevant articles.")

    # Save to Database
    print("\n[INFO] Saving articles to the database...")
    save_articles(articles)

    # Fetch from DB to run EDA
    print("\n[INFO] Extracting data from database for analysis...")
    df = get_all_articles()

    # Generate EDA
    print("\n[INFO] Firing up the EDA module...")
    run_eda(df)

    # Embed and Store in Vector Database
    print("\n[INFO] Launching Vector Database Embedding Sequence...")
    # store_articles_in_vector_db(df)

    print("\n=====================================================")
    print("            PIPELINE EXECUTION COMPLETE              ")
    print("   Please check 'eda_output' and 'chromadb' folder   ")
    print("=====================================================")

if __name__ == "__main__":
    main()

