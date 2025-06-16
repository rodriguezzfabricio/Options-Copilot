from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from enum import Enum

class SentimentScore(BaseModel):
    """Sentiment analysis results"""
    score: float = Field(..., ge=-1, le=1, description="Sentiment score between -1 (bearish) and 1 (bullish)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level of sentiment analysis")
    source: str = Field(..., description="Data source (news, social, earnings_call)")

class VolatilityForecast(BaseModel):
    """Volatility prediction model"""
    implied_volatility: float = Field(..., ge=0, description="Current implied volatility")
    predicted_volatility: float = Field(..., ge=0, description="AI-predicted volatility")
    confidence_interval: tuple[float, float] = Field(..., description="95% confidence interval")
    time_horizon: int = Field(..., gt=0, description="Forecast horizon in days")

class OptionsRecommendation(str, Enum):
    """Options strategy recommendations"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"

class RiskMetrics(BaseModel):
    """Risk assessment metrics"""
    max_loss: float = Field(..., description="Maximum potential loss")
    max_gain: Optional[float] = Field(None, description="Maximum potential gain (None for unlimited)")
    probability_of_profit: float = Field(..., ge=0, le=1, description="Probability of profitable outcome")
    risk_reward_ratio: float = Field(..., description="Risk to reward ratio")

class MarketAnalysis(BaseModel):
    """Comprehensive market analysis for options trading"""
    symbol: str = Field(..., description="Stock ticker symbol")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When analysis was performed")
    
    # Core analysis components
    sentiment_scores: List[SentimentScore] = Field(default_factory=list, description="Sentiment analysis from multiple sources")
    volatility_forecast: Optional[VolatilityForecast] = Field(None, description="Volatility predictions")
    recommendation: OptionsRecommendation = Field(..., description="AI-generated trading recommendation")
    
    # Financial metrics
    current_price: float = Field(..., gt=0, description="Current stock price")
    price_target: Optional[float] = Field(None, description="AI-predicted price target")
    support_levels: List[float] = Field(default_factory=list, description="Technical support levels")
    resistance_levels: List[float] = Field(default_factory=list, description="Technical resistance levels")
    
    # Risk assessment
    risk_metrics: Optional[RiskMetrics] = Field(None, description="Risk/reward analysis")
    
    # AI insights
    key_insights: List[str] = Field(default_factory=list, description="AI-generated key insights")
    earnings_impact: Optional[str] = Field(None, description="Expected earnings impact analysis")
    unusual_activity: List[str] = Field(default_factory=list, description="Unusual options flow detected")
    
    # Metadata
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence in analysis")
    data_sources_used: List[str] = Field(default_factory=list, description="Data sources utilized")
    
    class Config:
        validate_assignment = True
        use_enum_values = True

class DocumentAnalysis(BaseModel):
    """Analysis results from SEC filings and documents"""
    document_type: Literal["10-K", "10-Q", "8-K", "earnings_call", "news"] = Field(..., description="Type of document analyzed")
    document_url: Optional[str] = Field(None, description="Source document URL")
    summary: str = Field(..., description="AI-generated summary")
    key_points: List[str] = Field(default_factory=list, description="Key points extracted")
    sentiment: SentimentScore = Field(..., description="Document sentiment analysis")
    risk_factors: List[str] = Field(default_factory=list, description="Risk factors identified")
    financial_highlights: Dict[str, Any] = Field(default_factory=dict, description="Important financial metrics")

class EarningsCallAnalysis(BaseModel):
    """Real-time earnings call analysis"""
    company: str = Field(..., description="Company name")
    call_date: datetime = Field(..., description="Earnings call date")
    transcript_summary: str = Field(..., description="AI-generated call summary")
    management_sentiment: SentimentScore = Field(..., description="Management tone analysis")
    analyst_sentiment: SentimentScore = Field(..., description="Analyst Q&A sentiment")
    key_topics: List[str] = Field(default_factory=list, description="Main topics discussed")
    guidance_changes: List[str] = Field(default_factory=list, description="Guidance updates")
    surprise_factors: List[str] = Field(default_factory=list, description="Unexpected announcements")

class ChartPatternAnalysis(BaseModel):
    """Technical chart pattern recognition"""
    pattern_type: str = Field(..., description="Identified chart pattern")
    pattern_confidence: float = Field(..., ge=0, le=1, description="Pattern recognition confidence")
    breakout_probability: float = Field(..., ge=0, le=1, description="Probability of pattern completion")
    price_targets: List[float] = Field(default_factory=list, description="Pattern-based price targets")
    timeframe: str = Field(..., description="Chart timeframe analyzed")
    similar_historical_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Similar past patterns")

# Factory function for creating comprehensive analysis
def create_market_analysis(
    symbol: str,
    current_price: float,
    recommendation: OptionsRecommendation,
    confidence_score: float
) -> MarketAnalysis:
    """Factory function to create a market analysis with defaults"""
    return MarketAnalysis(
        symbol=symbol,
        current_price=current_price,
        recommendation=recommendation,
        confidence_score=confidence_score,
        data_sources_used=["market_data", "ai_analysis"]
    )