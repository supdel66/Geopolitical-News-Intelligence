import requests
import feedparser
from datetime import datetime
from config import NEWS_API_KEY, RSS_FEEDS, CONFLICT_KEYWORDS

def extract_from_newsapi():
    """Extract news from NewsAPI"""
    articles = []
    url = 'https://newsapi.org/v2/everything'
    
    for keyword in CONFLICT_KEYWORDS:
        params = {
            'q': keyword,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', []):
                    articles.append({
                        'source': article['source']['name'],
                        'title': article['title'],
                        'content': article.get('description', ''),
                        'published_at': article['publishedAt'],
                        'link': article['url']
                    })
        except Exception as e:
            print(f"Error fetching from NewsAPI for keyword {keyword}: {e}")
    
    return articles

def extract_from_rss(feed_url):
    """Extract news from RSS feed"""
    articles = []
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            articles.append({
                'source': feed.feed.get('title', 'Unknown Source'),
                'title': entry.title,
                'content': entry.get('description', entry.get('summary', '')),
                'published_at': entry.get('published', entry.get('updated', '')),
                'link': entry.link
            })
    except Exception as e:
        print(f"Error parsing RSS feed {feed_url}: {e}")
    
    return articles

def extract_all_articles():
    """Extract articles from all sources"""
    all_articles = []
    
    # Extract from NewsAPI
    print("Extracting from NewsAPI...")
    newsapi_articles = extract_from_newsapi()
    all_articles.extend(newsapi_articles)
    print(f"Extracted {len(newsapi_articles)} articles from NewsAPI")
    
    # Extract from RSS feeds
    for rss_feed in RSS_FEEDS:
        print(f"Extracting from RSS: {rss_feed}")
        rss_articles = extract_from_rss(rss_feed)
        all_articles.extend(rss_articles)
        print(f"Extracted {len(rss_articles)} articles from {rss_feed}")
    
    return all_articles
