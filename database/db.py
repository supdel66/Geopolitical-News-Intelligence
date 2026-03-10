import sqlite3
import pandas as pd
import os
from utils import timer_logger

DB_DIR = "sqlite_databases"
DB_PATH = os.path.join(DB_DIR, "news.db")

@timer_logger
def init_db():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT,
            content TEXT,
            published_at TEXT,
            link TEXT,
            scraped_at TEXT,
            img_link TEXT,
            UNIQUE(link)
        )
    ''')
    
    # Try to add img_link column if the database already existed without it
    try:
        cursor.execute('ALTER TABLE articles ADD COLUMN img_link TEXT')
    except sqlite3.OperationalError:
        # Column already exists
        pass
        
    conn.commit()
    conn.close()
    print("Database initialized.")


@timer_logger
def save_articles(articles):
    if not articles:
        print("No articles to save.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    saved_count = 0
    for article in articles:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO articles 
                (source, title, content, published_at, link, scraped_at, img_link)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.get("source", ""),
                article.get("title", ""),
                article.get("content", ""),
                article.get("published_at", ""),
                article.get("link", ""),
                article.get("scraped_at", ""),
                article.get("img_link", "")
            ))
            if cursor.rowcount > 0:
                saved_count += 1
        except Exception as e:
            print(f"Error saving article {article.get('title', '')}: {e}")
            
    conn.commit()
    conn.close()
    print(f"Saved {saved_count} new articles to database.")


@timer_logger
def get_all_articles():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()
    return df

@timer_logger
def is_article_saved(url: str) -> bool:
    """
    Checks if an article with exactly this `link` already exists in the SQLite database.
    """
    if not os.path.exists(DB_PATH):
        return False
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM articles WHERE link = ?", (url,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists
