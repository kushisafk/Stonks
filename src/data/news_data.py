import os
import time
import json
import requests
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import yfinance as yf
from src.config.settings import settings
from src.logging.logger import logger

class NewsDataCollector(ABC):
    """Abstract base class for financial news collection providers."""
    
    @abstractmethod
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetches news articles for a stock symbol over a trailing window.
        
        Args:
            symbol: Ticker symbol (e.g., AAPL)
            days: Trailing lookback window in days
            
        Returns:
            List[Dict[str, Any]]: Standardized articles list:
                                  [{"headline": str, "summary": str, "published_at": str, "source": str}]
        """
        pass

class FinnhubNewsCollector(NewsDataCollector):
    """Fetches stock news using Finnhub REST API."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or settings.FINNHUB_API_KEY
        
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        symbol = symbol.strip().upper()
        if not self.token:
            logger.warning("Finnhub API token is missing. Bypassing FinnhubNewsCollector.")
            return []
            
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
            "token": self.token
        }
        
        try:
            logger.info(f"Finnhub: Querying news for {symbol} from {start_date} to {end_date}...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json()
            
            standardized = []
            for art in articles:
                ts = art.get("datetime", time.time())
                published_at = datetime.fromtimestamp(ts).isoformat()
                
                standardized.append({
                    "headline": art.get("headline", ""),
                    "summary": art.get("summary", "") or art.get("headline", ""),
                    "published_at": published_at,
                    "source": art.get("source", "Finnhub")
                })
            logger.info(f"Finnhub: Successfully fetched {len(standardized)} articles for {symbol}.")
            return standardized[:settings.NEWS_MAX_ARTICLES]
        except Exception as e:
            logger.error(f"Finnhub API query failed for {symbol}: {e}")
            return []

class NewsAPINewsCollector(NewsDataCollector):
    """Fetches stock news using NewsAPI v2 Everything endpoint."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.NEWSAPI_API_KEY
        
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        symbol = symbol.strip().upper()
        if not self.api_key:
            logger.warning("NewsAPI key is missing. Bypassing NewsAPINewsCollector.")
            return []
            
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{symbol}" stock OR "{symbol}" finance',
            "from": start_date,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.api_key
        }
        
        try:
            logger.info(f"NewsAPI: Querying news for {symbol} starting {start_date}...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            
            articles = payload.get("articles", [])
            standardized = []
            for art in articles:
                standardized.append({
                    "headline": art.get("title", ""),
                    "summary": art.get("description", "") or art.get("content", "") or art.get("title", ""),
                    "published_at": art.get("publishedAt", ""),
                    "source": art.get("source", {}).get("name", "NewsAPI")
                })
            logger.info(f"NewsAPI: Successfully fetched {len(standardized)} articles for {symbol}.")
            return standardized[:settings.NEWS_MAX_ARTICLES]
        except Exception as e:
            logger.error(f"NewsAPI query failed for {symbol}: {e}")
            return []

class YFinanceNewsCollector(NewsDataCollector):
    """Fetches stock news using yfinance Ticker.news property. Serves as our default, zero-dependency free fallback."""
    
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        symbol = symbol.strip().upper()
        try:
            logger.info(f"YFinance News: Extracting news for {symbol}...")
            ticker = yf.Ticker(symbol)
            articles = ticker.news
            
            if not articles:
                logger.warning(f"YFinance News: No articles returned for {symbol}.")
                return []
                
            standardized = []
            lookback_cutoff = datetime.now() - timedelta(days=days)
            
            for art in articles:
                ts = art.get("providerPublishTime", time.time())
                pub_date = datetime.fromtimestamp(ts)
                
                # Apply lookback filter
                if pub_date < lookback_cutoff:
                    continue
                    
                standardized.append({
                    "headline": art.get("title", ""),
                    "summary": art.get("summary", "") or art.get("title", ""),
                    "published_at": pub_date.isoformat(),
                    "source": art.get("publisher", "YFinance")
                })
            logger.info(f"YFinance News: Successfully extracted {len(standardized)} articles for {symbol}.")
            return standardized[:settings.NEWS_MAX_ARTICLES]
        except Exception as e:
            logger.error(f"YFinance News extraction failed for {symbol}: {e}")
            return []

class MockNewsDataCollector(NewsDataCollector):
    """Mock news data collector from Phase 1, kept as fallback/testing mock."""
    
    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        symbol = symbol.strip().upper()
        return [
            {
                "headline": f"{symbol} earnings report exceeds analyst consensus expectations.",
                "summary": f"{symbol} shares rose as quarterly earnings beat estimates.",
                "published_at": datetime.now().isoformat(),
                "source": "Mock Capital"
            }
        ]

class CachedNewsCollector:
    """Orchestrates news collection by instantiating providers and caching JSON outputs daily."""
    
    def __init__(self, provider_name: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.provider_name = provider_name or settings.NEWS_PROVIDER
        self.cache_dir = cache_dir or settings.NEWS_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_provider()
        
    def _setup_provider(self):
        name = self.provider_name.strip().lower()
        if name == "finnhub":
            self.provider = FinnhubNewsCollector()
        elif name == "newsapi":
            self.provider = NewsAPINewsCollector()
        elif name == "yfinance":
            self.provider = YFinanceNewsCollector()
        elif name == "mock":
            self.provider = MockNewsDataCollector()
        else:
            logger.warning(f"Unknown news provider '{self.provider_name}'. Falling back to YFinanceNewsCollector.")
            self.provider = YFinanceNewsCollector()
            
    def get_news(self, symbol: str, days: Optional[int] = None, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieves news articles for a symbol, checking cache first.
        
        Args:
            symbol: Ticker symbol
            days: Trailing days to lookup (defaults to settings.NEWS_LOOKBACK_DAYS)
            force_refresh: If True, bypasses cache and queries API directly
        """
        symbol = symbol.strip().upper()
        days = days if days is not None else settings.NEWS_LOOKBACK_DAYS
        
        # Caching strategy: cache news daily per symbol to capture moving sentiments
        date_str = datetime.now().strftime("%Y-%m-%d")
        cache_path = self.cache_dir / f"{symbol}_{date_str}.json"
        
        if not force_refresh and cache_path.exists():
            try:
                logger.info(f"News Cache hit: Loading news for {symbol} from {cache_path}")
                with open(cache_path, mode="r", encoding="utf-8") as f:
                    articles = json.load(f)
                return articles
            except Exception as e:
                logger.warning(f"Error loading news cache for {symbol}: {e}")
                
        # Cache miss: fetch news from active provider
        articles = self.provider.fetch_news(symbol, days=days)
        
        # If API failed and we have no articles, check if a previous day's cache exists as an emergency fallback
        if not articles:
            logger.warning(f"News provider failed to return articles for {symbol}. Checking historical cache fallbacks...")
            for i in range(1, 4):
                fallback_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                fallback_path = self.cache_dir / f"{symbol}_{fallback_date}.json"
                if fallback_path.exists():
                    try:
                        logger.warning(f"Emergency recovery: Loading news cache from {fallback_path}")
                        with open(fallback_path, mode="r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        pass
                        
        # Save successfully fetched articles to cache
        if articles:
            try:
                with open(cache_path, mode="w", encoding="utf-8") as f:
                    json.dump(articles, f, indent=4, ensure_ascii=False)
                logger.info(f"News Cache created for {symbol} at {cache_path}")
            except Exception as e:
                logger.error(f"Failed to write news cache for {symbol}: {e}")
                
        return articles

# Global orchestrator instance
news_collector = CachedNewsCollector()
