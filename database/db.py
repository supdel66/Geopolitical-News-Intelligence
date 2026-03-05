import sqlite3
import pandas as pd
import os

DB_DIR = "sqlite_databases"
DB_PATH = os.path.join(DB_DIR, "news.db")

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


def get_all_articles():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM articles", conn)
    conn.close()
    return df


