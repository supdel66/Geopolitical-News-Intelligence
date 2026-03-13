import sqlite3
import pandas as pd
import os
from utils import logger
import logging

class SQLiteDatabase:
    def __init__(self, db_dir="sqlite_databases", db_name="news.db"):
        self.db_dir = db_dir
        self.db_path = os.path.join(self.db_dir, db_name)
        self.init_db()
        
    def init_db(self):
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            
        conn = sqlite3.connect(self.db_path)
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
        logger.info("Database initialized.")

    def save_articles(self, articles_iterator):
        """Consume an iterator of articles and save them to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for article in articles_iterator:
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
                    logger.info(f"Saved to DB: {article.get('title', '')[:50]}...")
            except Exception as e:
                logger.error(f"Error saving article {article.get('title', '')}: {e}")
                
        conn.commit()
        conn.close()
        logger.info(f"Total newly saved articles: {saved_count}")

    def get_all_articles(self):
        """Returns a Pandas DataFrame for EDA tasks."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM articles", conn)
        conn.close()
        return df

    def get_all_articles_iterator(self, batch_size=500):
        """Yield articles in batches to prevent memory overflow."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM articles")
        while True:
            records = cursor.fetchmany(batch_size)
            if not records:
                break
            for row in records:
                yield dict(row)
                
        conn.close()

    def is_article_saved(self, url: str) -> bool:
        """Checks if an article with exactly this link already exists."""
        if not os.path.exists(self.db_path):
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM articles WHERE link = ?", (url,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
