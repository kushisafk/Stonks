import hashlib
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config.settings import settings
from src.logging.logger import logger

class FinBERTLoader:
    """Singleton-style lazy loader for HuggingFace FinBERT tokenizer and classification model."""
    _tokenizer = None
    _model = None
    
    @classmethod
    def get_tokenizer_and_model(cls):
        if cls._tokenizer is None or cls._model is None:
            logger.info("FinBERT: Lazily loading 'ProsusAI/finbert' tokenizer and classification model on CPU...")
            cls._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            cls._model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            # Default to CPU execution to guarantee standard platform compatibility
            cls._model.to("cpu")
            cls._model.eval()
            logger.info("FinBERT: Successfully loaded ProsusAI/finbert model.")
        return cls._tokenizer, cls._model

class SentimentAnalyzer:
    """Analyzes financial news articles using FinBERT and manages an inference cache to save CPU cycles."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or settings.SENTIMENT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_text_hash(self, headline: str, summary: str) -> str:
        """Generates a unique MD5 hex digest for article content."""
        combined = f"{headline.strip()}::{summary.strip()}"
        return hashlib.md5(combined.encode("utf-8")).hexdigest()
        
    def _get_cache_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.json"
        
    def analyze_article(self, headline: str, summary: str) -> Dict[str, float]:
        """
        Calculates sentiment classification probabilities (positive, neutral, negative).
        Checks local inference cache first.
        
        Args:
            headline: Article title
            summary: Article description/summary
            
        Returns:
            Dict[str, float]: Softmax normalized probabilities for positive, neutral, and negative
        """
        headline = headline.strip()
        summary = summary.strip()
        
        if not headline and not summary:
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
            
        text_hash = self._get_text_hash(headline, summary)
        cache_path = self._get_cache_path(text_hash)
        
        # Check local inference cache
        if cache_path.exists():
            try:
                with open(cache_path, mode="r", encoding="utf-8") as f:
                    probs = json.load(f)
                return probs
            except Exception as e:
                logger.warning(f"Error loading sentiment inference cache: {e}")
                
        # Cache miss, run PyTorch model inference
        combined_text = f"{headline} {summary}".strip()
        logger.debug(f"FinBERT Inference: Analyzing text (Hash: {text_hash})...")
        
        try:
            tokenizer, model = FinBERTLoader.get_tokenizer_and_model()
            inputs = tokenizer(combined_text, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                # Apply softmax to get normalized probabilities
                probs = F.softmax(logits, dim=-1).squeeze().tolist()
                
            # Read label mapping dynamically from model config
            id2label = model.config.id2label
            sentiment = {
                id2label[i].lower(): float(probs[i])
                for i in range(len(probs))
            }
            
            # Save to inference cache
            with open(cache_path, mode="w", encoding="utf-8") as f:
                json.dump(sentiment, f, indent=4)
            return sentiment
            
        except Exception as e:
            logger.error(f"FinBERT Sentiment Inference failed: {e}")
            # Fallback to standard neutral on errors
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    def analyze_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes a batch of standardized articles, appending 'sentiment' dictionary to each article.
        
        Args:
            articles: List of standardized article dictionaries
            
        Returns:
            List[Dict[str, Any]]: Updated article lists with 'sentiment' fields
        """
        logger.info(f"FinBERT: Batch analyzing {len(articles)} articles...")
        analyzed = []
        for art in articles:
            headline = art.get("headline", "")
            summary = art.get("summary", "")
            
            sentiment = self.analyze_article(headline, summary)
            
            art_copy = art.copy()
            art_copy["sentiment"] = sentiment
            analyzed.append(art_copy)
        return analyzed

# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()
