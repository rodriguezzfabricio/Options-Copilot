# backend/app/api/v1/ticker_analysis_v2.py
"""
Production-ready ticker analysis endpoint for OptionsAI Copilot
"""

from fastapi import APIRouter, HTTPException
import traceback
from app.models.analysis import MarketAnalysis

router = APIRouter()

@router.post("/analyze/ticker/{ticker}", response_model=MarketAnalysis)
async def analyze_ticker_v2(ticker: str):
    """
    Production-ready ticker analysis using async queue processing
    Returns cached results immediately or queues analysis for background processing
    """
    from app.services.market_data_service import market_data_service
    
    try:
        # Initialize service if not already done
        if not market_data_service.redis_client:
            await market_data_service.initialize()
        
        # Get analysis (returns immediately with cached data or basic analysis)
        analysis = await market_data_service.get_ticker_analysis(ticker)
        return analysis
        
    except Exception as e:
        print(f"Error in ticker analysis: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis service error for {ticker}: {str(e)}"
        )