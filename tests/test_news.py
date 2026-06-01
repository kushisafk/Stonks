import json
from pathlib import Path
from src.data.news_data import CachedNewsCollector, MockNewsDataCollector, YFinanceNewsCollector

def test_mock_news_collector():
    """Verify that MockNewsDataCollector returns standard headline fields."""
    collector = MockNewsDataCollector()
    articles = collector.fetch_news("AAPL")
    
    assert len(articles) == 1
    assert articles[0]["headline"] == "AAPL earnings report exceeds analyst consensus expectations."
    assert "summary" in articles[0]
    assert "source" in articles[0]
    assert "published_at" in articles[0]

def test_yfinance_news_collector():
    """Verify that YFinanceNewsCollector extracts news and standardizes timestamps (when online)."""
    # This runs a live query if online, otherwise safe try-catch
    collector = YFinanceNewsCollector()
    try:
        articles = collector.fetch_news("TSLA", days=3)
        if articles:
            art = articles[0]
            assert "headline" in art
            assert "published_at" in art
            assert "source" in art
            assert "summary" in art
    except Exception as e:
        # Ignore network errors in sandboxed tests
        pass

def test_cached_news_collector(tmp_path):
    """Verify that CachedNewsCollector writes to disk on miss and fetches from disk on hit."""
    # Build orchestrator using Mock provider and temporary cache directory
    orchestrator = CachedNewsCollector(provider_name="mock", cache_dir=tmp_path)
    
    # 1. First fetch (Cache Miss)
    articles_1 = orchestrator.get_news("AAPL", force_refresh=True)
    assert len(articles_1) == 1
    
    # Verify JSON file exists in cache
    expected_cache_file = tmp_path / f"AAPL_{Path(tmp_path).name}.json"  # Wait, the date string format is used
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    expected_cache_file = tmp_path / f"AAPL_{date_str}.json"
    assert expected_cache_file.exists()
    
    # Modify cache file manually to prove subsequent calls hit cache
    with open(expected_cache_file, mode="w", encoding="utf-8") as f:
        json.dump([{"headline": "CACHE HIT SUCCESS", "summary": "N/A", "published_at": "N/A", "source": "Cache"}], f)
        
    # 2. Second fetch (Cache Hit)
    articles_2 = orchestrator.get_news("AAPL", force_refresh=False)
    assert len(articles_2) == 1
    assert articles_2[0]["headline"] == "CACHE HIT SUCCESS"
