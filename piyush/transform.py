import re
from datetime import datetime
from config import CONFLICT_KEYWORDS

def normalize_date(date_string):
    """Normalize different date formats to ISO format"""
    try:
        # Try parsing common date formats
        for fmt in [
            '%a, %d %b %Y %H:%M:%S %Z',
            '%a, %d %b %Y %H:%M:%S %z',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%d %H:%M:%S'
        ]:
            try:
                dt = datetime.strptime(date_string, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # If all parsing fails, return current time
        return datetime.now().isoformat()
    except:
        return datetime.now().isoformat()

def contains_conflict_keywords(text):
    """Check if text contains any conflict-related keywords"""
    if not text:
        return False
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CONFLICT_KEYWORDS)

def transform_articles(articles):
    """Transform and filter articles"""
    transformed = []
    
    for article in articles:
        # Filter articles containing conflict keywords
        article_text = f"{article['title']} {article['content']}".lower()
        
        if contains_conflict_keywords(article_text):
            # Clean and transform data
            transformed_article = {
                'source': article['source'].strip() if article['source'] else 'Unknown',
                'title': article['title'].strip() if article['title'] else 'No Title',
                'content': article['content'].strip() if article['content'] else '',
                'published_at': normalize_date(article['published_at']),
                'link': article['link'].strip()
            }
            
            # Basic validation
            if (transformed_article['title'] != 'No Title' and 
                transformed_article['link'].startswith('http')):
                transformed.append(transformed_article)
    
    print(f"Transformed {len(transformed)} articles after filtering")
    return transformed
