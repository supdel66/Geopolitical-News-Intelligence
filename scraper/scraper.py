import os
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from utils import timer_logger
from database.db import is_article_saved

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

RSS_SOURCES = {
    "BBC": "https://feeds.bbci.co.uk/news/rss.xml",
    "Associated_Press": "https://feedx.net/rss/ap.xml",
    "Al_Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "CNN": "http://rss.cnn.com/rss/edition_world.rss",
    "Fox_News": "https://moxie.foxnews.com/google-publisher/world.xml",
    "New_York_Times": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "ABC_News": "https://abcnews.go.com/abcnews/topstories",
    "Washington_Post": "https://feeds.washingtonpost.com/rss/national",
    "Yahoo_News": "https://news.yahoo.com/rss/topstories"
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

def extract_full_article_BBC(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("article")
        img_link = article.find("img").get("src") if article and article.find("img") else ""
        ps = article.find_all("p") if article else []
        fulltext = "".join([p.text for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping BBC {url}: {e}")
        return "", ""
    
def extract_full_article_Associated_Press(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("main")
        img_link = article.find("img").get("src") if article and article.find("img") else ""
        ps = article.find_all("p") if article else []
        fulltext = "".join([p.text for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping Associated_Press {url}: {e}")
        return "", ""

def extract_full_article_Al_Jazeera(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("main")
        img_link = article.find("img").get("src") if article and article.find("img") else ""
        ps = article.find_all("p") if article else []
        fulltext = "".join([p.text for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping Al_Jazeera {url}: {e}")
        return "", ""

def extract_full_article_Guardian(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("main")
        img_link = article.find("img").get("src") if article and article.find("img") else ""
        ps = article.find_all("p") if article else []
        fulltext = "".join([p.text for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping Guardian {url}: {e}")
        return "", ""

def extract_full_article_CNN(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        img_link = soup.find("img").get("src") if soup.find("img") else ""
        ps = soup.find_all("p")
        fulltext = "".join([p.text.strip() for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping CNN {url}: {e}")
        return "", ""

def extract_full_article_Fox_News(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("div", class_="article-content")
        img_link = article.find("img").get("src") if article and article.find("img") else ""
        ps = article.find_all("p") if article else []
        fulltext = "".join([p.text.strip() for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping Fox_News {url}: {e}")
        return "", ""

def extract_full_article_ABC_News(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        img_link = soup.find("img").get("src") if soup.find("img") else ""
        ps = soup.find_all("p")
        fulltext = "".join([p.text.strip() for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping ABC_News {url}: {e}")
        return "", ""

def extract_full_article_Yahoo_News(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("article")
        img_link = article.find("img").get("src") if article and article.find("img") else ""
        ps = article.find_all("p") if article else []
        fulltext = "".join([p.text.strip() for p in ps])
        return img_link, fulltext
    except Exception as e:
        print(f"Error scraping Yahoo_News {url}: {e}")
        return "", ""

def extract_full_article_newsapi(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # try article tag first, fallback to main, then body
        container = (
            soup.find("article") or
            soup.find("main") or
            soup.find("body")
        )
        
        img_link = container.find("img").get("src") if container and container.find("img") else ""
        ps = container.find_all("p") if container else []
        fulltext = "".join([p.text.strip() for p in ps])
        
        return img_link, fulltext

    except Exception as e:
        print(f"Error scraping NewsAPI article {url}: {e}")
        return "", ""

@timer_logger
def fetch_rss_news():
    articles = []
    for source_name, rss_url in RSS_SOURCES.items():
        print(f"Scraping RSS: {source_name}...")
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:15]: 
                #save entry to txt file 
                title = entry.get("title", "")
                link = entry.get("link", "")
                published = entry.get("published", "")
                img_link, full_text = "", ""
                
                if is_article_saved(link):
                    print(f"Skipping RSS article already in DB: {link}")
                    continue

                if source_name in ["Washington_Post", "New_York_Times"]:
                    full_text = entry.get('value', entry.get('summary', ''))
                    img_link = entry.get('media_content', [{}])[0].get('url', '')
                else:
                    func_name = f"extract_full_article_{source_name}"
                    extract_func = globals().get(func_name)
                    if extract_func:
                        try:
                            img_link, full_text = extract_func(link)
                        except Exception as e:
                            print(f"Error scraping {source_name} at {link}: {e}")
                
                # Only if the teaser is relevant, we spend time downloading the FULL article
                # In Jupyter they filtered after downloading all text, here we do it after downloading text if applicable
                
                if is_conflict_related(title + " " + full_text):
                    print(f" -> Relevant article found: {title}")
                    
                    articles.append({
                        "source": source_name,
                        "title": title,
                        "content": full_text if full_text else entry.get("summary", ""),
                        "published_at": published,
                        "link": link,
                        "img_link": img_link,
                        "scraped_at": datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error fetching from {source_name}: {e}")
            
    return articles

@timer_logger
def fetch_newsapi_news(api_key, queries=None):
    if queries is None:

        queries = [
             "war", "iran", "israel", "united states",
    "us", "missile", "attack", "military",
    "retaliation", "conflict", "gaza",
    "tehran", "hezbollah"]
    
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
                    
                    if is_article_saved(link):
                        print(f"Skipping NewsAPI article already in DB: {link}")
                        continue

                    img_link, full_text = extract_full_article_newsapi(link)
                    
                    if is_conflict_related(title + " " + content + " " + full_text):
                        articles.append({
                            "source": source_name,
                            "title": title,
                            "content": full_text if full_text else content,
                            "published_at": published_at,
                            "link": link,
                            "img_link": img_link,
                            "scraped_at": datetime.now().isoformat()
                        })
            else:
                print(f"NewsAPI error {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            
    return articles

@timer_logger
def scrape_all_sources(api_key=None):
    all_articles = []
    print("Starting RSS scraping...")
    rss_articles = fetch_rss_news()
    all_articles.extend(rss_articles)
    
    if api_key:
        print("Starting NewsAPI scraping...")
        newsapi_articles = fetch_newsapi_news(api_key)
        all_articles.extend(newsapi_articles)
    else:
        print("Skipping NewsAPI (No API Key provided).")
    print(f"Total articles scraped: {   len(all_articles)}")
    return all_articles
