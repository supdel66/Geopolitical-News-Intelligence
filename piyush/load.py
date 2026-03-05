from database import get_db_connection

def load_articles(articles):
    """Load articles into SQLite database"""
    if not articles:
        print("No articles to load")
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    loaded_count = 0
    
    for article in articles:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO news_articles 
                (source, title, content, published_at, link)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                article['source'],
                article['title'],
                article['content'],
                article['published_at'],
                article['link']
            ))
            
            if cursor.rowcount > 0:
                loaded_count += 1
                
        except Exception as e:
            print(f"Error loading article {article['title']}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    print(f"Successfully loaded {loaded_count} new articles")
    return loaded_count

def get_recent_articles(limit=10):
    """Retrieve recent articles from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT source, title, content, published_at, link 
        FROM news_articles 
        ORDER BY published_at DESC 
        LIMIT ?
    ''', (limit,))
    
    articles = cursor.fetchall()
    conn.close()
    
    return articles
