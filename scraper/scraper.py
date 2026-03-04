import os
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

RSS_SOURCES = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
#     "Associated Press": "https://feedx.net/rss/ap.xml",
#     "UN News": "https://news.un.org/en/rss-feeds",
#     "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
#     "Reuters": "https://www.reutersagency.com/feed/?best-topics=top-news",
#     "Guardian": "https://www.theguardian.com/world/rss",
#     "CNN": "http://rss.cnn.com/rss/edition_world.rss",
#     "POLITICO": "https://www.politico.com/rss",
#     "NYTimes": "https://www.nytimes.com/rss",
#     "NDTV": "https://www.ndtv.com/rss",
#     "The Hindu": "https://www.thehindu.com/rssfeeds/",
#     "Newswise": "https://www.newswise.com/channels/rss",
#     "Fox News": "http://moxie.foxnews.com/google-publisher/latest.xml",
#     "New York Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
#     "ABC News": "https://abcnews.go.com/abcnews/topstories",
#     "Washington Post": "https://feeds.washingtonpost.com/rss/national",
#     "Yahoo News": "https://news.yahoo.com/rss/topstories"
}


KEYWORDS = [
        "war", "iran", "israel", "united states",
        "us", "missile", "attack", "military",
        "retaliation", "conflict", "gaza",
        "tehran", "hezbollah",
        "world war 3", "ww3", "nuclear", "escalation", "world war iii", 
        "global conflict", "third world war", "thermonuclear", "armageddon",
        "doomsday", "nato article 5", "mutual assured destruction", "us israel", "united states israel", "biden israel", "trump israel",
        "american israel", "iron dome", "us aid israel", "f-35 israel",
        "idf us", "military aid israel", "israel lobby","israel iran", "iran attack", "iran strike", "tehran israel",
        "idf iran", "iranian drone", "iranian missile", "mossad iran",
        "nuclear iran", "iran nuclear deal", "jcpoa", "khamenei israel", "iran us", "iran america", "iran sanction", "us strike iran",
        "iran proxy", "hezbollah", "hamas", "houthi", "irgc", "strait of hormuz", "persian gulf us",
        "gaza", "west bank", "beirut", "damascus", "iraq militia", "red sea attack", "tanker attack", "drone swarm", "ballistic missile"]


def is_conflict_related(text):
    if not text:
        return False
    text = text.lower()
    return any(k in text for k in KEYWORDS)

def extract_full_article(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join([p.get_text() for p in paragraphs])
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def fetch_rss_news():
    articles = []
    for source_name, rss_url in RSS_SOURCES.items():
        print(f"Scraping RSS: {source_name}...")
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries: 
                #save entry to txt file 
                print(f"Entry: {entry}")
                title = entry.get("title", "")
                link = entry.get("link", "")
                published = entry.get("published", "")
                
                # The summary is just a 1-sentence teaser from the RSS feed
                summary_teaser = entry.get("summary", "")
                
                # Only if the teaser is relevant, we spend time downloading the FULL article
                if is_conflict_related(title + " " + summary_teaser):
                    print(f" -> Relevant article found, downloading full text: {title}")
                    full_content = extract_full_article(link)
                    
                    articles.append({
                        "source": source_name,
                        "title": title,
                        "content": full_content if full_content else summary_teaser,
                        "published_at": published,
                        "link": link,
                        "scraped_at": datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            
    return articles

def fetch_newsapi_news(api_key, queries=None):
    if queries is None:

        queries = [
             "war", "iran", "israel", "united states",
    "us", "missile", "attack", "military",
    "retaliation", "conflict", "gaza",
    "tehran", "hezbollah",
            "world war 3", "israel iran", "us iran escalation",
            "russia ukraine", "china taiwan", "middle east conflict",
            "nato", "geopolitics", "nuclear threat", "red sea crisis",     "world war 3", "ww3", "nuclear", "escalation", "world war iii",
        "global conflict", "third world war", "thermonuclear", "armageddon",
        "doomsday", "nato article 5", "mutual assured destruction", "us israel", "united states israel", "biden israel", "trump israel",
        "american israel", "iron dome", "us aid israel", "f-35 israel",
        "idf us", "military aid israel", "israel lobby","israel iran", "iran attack", "iran strike", "tehran israel",
        "idf iran", "iranian drone", "iranian missile", "mossad iran",
        "nuclear iran", "iran nuclear deal", "jcpoa", "khamenei israel", "iran us", "iran america", "iran sanction", "us strike iran",
        "iran proxy", "hezbollah", "hamas", "houthi", "irgc", "strait of hormuz", "persian gulf us",
        "gaza", "west bank", "beirut", "damascus", "iraq militia", "red sea attack", "tanker attack", "drone swarm", "ballistic missile"
        ]

    articles = []
    url = "https://newsapi.org/v2/everything"
    for query in queries:
        print(f"Scraping NewsAPI for query: {query}...")
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 10,
            "apiKey": api_key,
        }
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("articles", []):
                    source_name = item.get("source", {}).get("name", "NewsAPI")
                    title = item.get("title", "")
                    content = item.get("description", "") or item.get("content", "")
                    published_at = item.get("publishedAt", "")
                    link = item.get("url", "")
                    
                    if is_conflict_related(title + " " + content):
                        articles.append({
                            "source": source_name,
                            "title": title,
                            "content": content,
                            "published_at": published_at,
                            "link": link,
                            "scraped_at": datetime.now().isoformat()
                        })
            else:
                print(f"NewsAPI error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            
    return articles

def scrape_all_sources(api_key=None):
    all_articles = []
    print("Starting RSS scraping...")
    rss_articles = fetch_rss_news()
    all_articles.extend(rss_articles)
    
    if api_key:
        print("Starting NewsAPI scraping...")
        # newsapi_articles = fetch_newsapi_news(api_key)
        # all_articles.extend(newsapi_articles)
    else:
        print("Skipping NewsAPI (No API Key provided).")
    print(f"Total articles scraped: {   len(all_articles)}")
    return all_articles
