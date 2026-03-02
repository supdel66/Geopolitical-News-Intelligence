import os

# API Configuration
NEWS_API_KEY = "238f85291fc94ea5a7c6c6bed099d352"
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml", 
    "https://www.theguardian.com/world/rss"
]

# Keywords for filtering
CONFLICT_KEYWORDS = [
    'ww3', 'world war', 'military', 'nuclear', 'missile', 'missiles',
    'israel', 'iran', 'usa', 'united states', 'conflict', 'war',
    'tension', 'attack', 'defense', 'weapon', 'troops', 'army'
]

# Database configuration
DB_PATH = 'news.db'
