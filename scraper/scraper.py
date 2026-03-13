import os
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime
from utils import timer_logger, logger
from database.db import SQLiteDatabase

class NewsScraper:
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
        "gaza", "west bank", "beirut", "damascus", "iraq militia", "red sea attack", "tanker attack", "drone swarm", "ballistic missile"
    ]

    def __init__(self, api_key=None, db: SQLiteDatabase = None):
        self.api_key = api_key
        self.db = db

    def is_conflict_related(self, text):
        if not text:
            return False
        text = text.lower()
        return any(k in text for k in self.KEYWORDS)

    @staticmethod
    def extract_full_article_BBC(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article")
            img_link = article.find("img").get("src") if article and article.find("img") else ""
            ps = article.find_all("p") if article else []
            fulltext = "".join([p.text for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping BBC {url}: {e}")
            return "", ""
        
    @staticmethod
    def extract_full_article_Associated_Press(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("main")
            img_link = article.find("img").get("src") if article and article.find("img") else ""
            ps = article.find_all("p") if article else []
            fulltext = "".join([p.text for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping Associated_Press {url}: {e}")
            return "", ""

    @staticmethod
    def extract_full_article_Al_Jazeera(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("main")
            img_link = article.find("img").get("src") if article and article.find("img") else ""
            ps = article.find_all("p") if article else []
            fulltext = "".join([p.text for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping Al_Jazeera {url}: {e}")
            return "", ""

    @staticmethod
    def extract_full_article_Guardian(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("main")
            img_link = article.find("img").get("src") if article and article.find("img") else ""
            ps = article.find_all("p") if article else []
            fulltext = "".join([p.text for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping Guardian {url}: {e}")
            return "", ""

    @staticmethod
    def extract_full_article_CNN(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            img_link = soup.find("img").get("src") if soup.find("img") else ""
            ps = soup.find_all("p")
            fulltext = "".join([p.text.strip() for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping CNN {url}: {e}")
            return "", ""

    @staticmethod
    def extract_full_article_Fox_News(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("div", class_="article-content")
            img_link = article.find("img").get("src") if article and article.find("img") else ""
            ps = article.find_all("p") if article else []
            fulltext = "".join([p.text.strip() for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping Fox_News {url}: {e}")
            return "", ""

    @staticmethod
    def extract_full_article_ABC_News(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            img_link = soup.find("img").get("src") if soup.find("img") else ""
            ps = soup.find_all("p")
            fulltext = "".join([p.text.strip() for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping ABC_News {url}: {e}")
            return "", ""

    @staticmethod
    def extract_full_article_Yahoo_News(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            article = soup.find("article")
            img_link = article.find("img").get("src") if article and article.find("img") else ""
            ps = article.find_all("p") if article else []
            fulltext = "".join([p.text.strip() for p in ps])
            return img_link, fulltext
        except Exception as e:
            logger.error(f"Error scraping Yahoo_News {url}: {e}")
            return "", ""

    @staticmethod
    def extract_full_article_newsapi(url):
        try:
            response = requests.get(url, headers=NewsScraper.HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
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
            logger.error(f"Error scraping NewsAPI article {url}: {e}")
            return "", ""

    def fetch_rss_news(self):
        for source_name, rss_url in self.RSS_SOURCES.items():
            logger.info(f"Scraping RSS: {source_name}...")
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries[:15]: 
                    title = entry.get("title", "")
                    link = entry.get("link", "")
                    published = entry.get("published", "")
                    img_link, full_text = "", ""
                    
                    if self.db and self.db.is_article_saved(link):
                        logger.info(f"Skipping RSS article already in DB: {link}")
                        continue

                    if source_name in ["Washington_Post", "New_York_Times"]:
                        full_text = entry.get('value', entry.get('summary', ''))
                        img_link = entry.get('media_content', [{}])[0].get('url', '')
                    else:
                        func_name = f"extract_full_article_{source_name}"
                        if hasattr(self, func_name):
                            extract_func = getattr(self, func_name)
                            try:
                                img_link, full_text = extract_func(link)
                            except Exception as e:
                                logger.error(f"Error scraping {source_name} at {link}: {e}")
                    
                    if self.is_conflict_related(title + " " + full_text):
                        logger.info(f" -> Relevant article found: {title}")
                        yield {
                            "source": source_name,
                            "title": title,
                            "content": full_text if full_text else entry.get("summary", ""),
                            "published_at": published,
                            "link": link,
                            "img_link": img_link,
                            "scraped_at": datetime.now().isoformat()
                        }
            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")

    def fetch_newsapi_news(self):
        queries = [
             "war", "iran", "israel", "united states",
             "us", "missile", "attack", "military",
             "retaliation", "conflict", "gaza",
             "tehran", "hezbollah"
        ]
        
        url = "https://newsapi.org/v2/everything"
        for query in queries:
            logger.info(f"Scraping NewsAPI for query: {query}...")
            params = {
                "q": query,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 10,
                "apiKey": self.api_key,
            }
            try:
                resp = requests.get(url, params=params, headers=self.HEADERS, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("articles", []):
                        source_name = item.get("source", {}).get("name", "NewsAPI")
                        title = item.get("title", "")
                        content = item.get("description", "") or item.get("content", "")
                        published_at = item.get("publishedAt", "")
                        link = item.get("url", "")
                        
                        if self.db and self.db.is_article_saved(link):
                            logger.info(f"Skipping NewsAPI article already in DB: {link}")
                            continue

                        img_link, full_text = self.extract_full_article_newsapi(link)
                        
                        if self.is_conflict_related(title + " " + content + " " + full_text):
                            logger.info(f" -> Relevant article found: {title}")
                            yield {
                                "source": source_name,
                                "title": title,
                                "content": full_text if full_text else content,
                                "published_at": published_at,
                                "link": link,
                                "img_link": img_link,
                                "scraped_at": datetime.now().isoformat()
                            }
                else:
                    logger.error(f"NewsAPI error {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.error(f"Error fetching from NewsAPI: {e}")

    def scrape_all_sources(self):
        logger.info("Starting RSS scraping...")
        yield from self.fetch_rss_news()
        
        if self.api_key:
            logger.info("Starting NewsAPI scraping...")
            yield from self.fetch_newsapi_news()
        else:
            logger.info("Skipping NewsAPI (No API Key provided).")
