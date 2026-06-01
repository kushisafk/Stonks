from pydantic import BaseModel, Field

class SentimentResponse(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    sentiment_score: float = Field(..., description="Aggregated news sentiment score ranging from -1.0 to +1.0")
    articles_analyzed: int = Field(..., description="Total number of articles evaluated")
    positive_ratio: float = Field(..., description="Fraction of articles classified as positive")
    negative_ratio: float = Field(..., description="Fraction of articles classified as negative")
