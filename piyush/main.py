from database import create_database
from extract import extract_all_articles
from transform import transform_articles
from load import load_articles, get_recent_articles
import time
from datetime import datetime

def run_etl_pipeline():
    """Run the complete ETL pipeline"""
    print("Starting ETL Pipeline...")
    print(f"Execution time: {datetime.now()}")
    
    # Step 1: Create database
    print("\n1. Creating database...")
    create_database()
    
    # Step 2: Extract
    print("\n2. Extracting articles...")
    raw_articles = extract_all_articles()
    print(f"Total articles extracted: {len(raw_articles)}")
    
    # Step 3: Transform
    print("\n3. Transforming and filtering articles...")
    transformed_articles = transform_articles(raw_articles)
    
    # Step 4: Load
    print("\n4. Loading articles to database...")
    loaded_count = load_articles(transformed_articles)
    
    # Display results
    print(f"\nETL Pipeline completed!")
    print(f"Extracted: {len(raw_articles)} articles")
    print(f"After filtering: {len(transformed_articles)} articles")
    print(f"Loaded: {loaded_count} new articles")
    
    # Show recent articles
    print(f"\nRecent articles in database:")
    recent_articles = get_recent_articles(5)
    for i, article in enumerate(recent_articles, 1):
        print(f"{i}. [{article[0]}] {article[1]}")

if __name__ == "__main__":
    run_etl_pipeline()
