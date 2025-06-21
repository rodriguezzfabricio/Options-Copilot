# backend/app/api/v1/analysis.py
"""
First real endpoint - SEC Filing Analysis with Q&A
This is your MVP feature to get started!
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
import io
from datetime import datetime, timedelta

import yfinance as yf
import traceback
from fastapi import HTTPException
from functools import lru_cache
import threading

# Fix common yfinance issues
import requests
requests.packages.urllib3.disable_warnings()

# ADD THIS SECTION HERE - RIGHT AFTER YOUR IMPORTS
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure yfinance to handle rate limiting
def setup_yfinance_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    return session

# Import and fix yfinance
yf._session = setup_yfinance_session()

# Cache to store ticker data and reduce API calls
ticker_cache = {}
cache_lock = threading.Lock()
last_api_call = None
API_CALL_DELAY = 30  # Minimum seconds between API calls - very conservative
CACHE_DURATION = timedelta(hours=1)  # Cache data for 1 hour to minimize API calls

def get_cached_ticker_data(ticker: str):
    """Get cached ticker data if available, otherwise fetch from yfinance"""
    global last_api_call
    
    with cache_lock:
        cache_key = ticker.upper()
        now = datetime.now()
        
        # Check if we have valid cached data
        if cache_key in ticker_cache:
            cached_data, timestamp = ticker_cache[cache_key]
            if now - timestamp < CACHE_DURATION:
                print(f"Using cached data for {ticker}")
                return cached_data
        
        # Enforce rate limiting between API calls
        if last_api_call is not None:
            time_since_last = (now - last_api_call).total_seconds()
            if time_since_last < API_CALL_DELAY:
                sleep_time = API_CALL_DELAY - time_since_last
                print(f"Rate limiting: sleeping {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        last_api_call = datetime.now()
        
        # Fetch fresh data
        print(f"Fetching fresh data for {ticker}")
        stock = yf.Ticker(ticker.upper())
        
        # Collect all data in one go with delays
        ticker_data = {
            'current_price': None,
            'info': None,
            'news': []
        }
        
        # Single API call strategy - get what we can from stock.info only
        print("Making single API call to stock.info...")
        try:
            ticker_data['info'] = stock.info
            
            # Extract price from info
            if ticker_data['info']:
                ticker_data['current_price'] = (
                    ticker_data['info'].get('currentPrice') or
                    ticker_data['info'].get('regularMarketPrice') or
                    ticker_data['info'].get('previousClose')
                )
                print(f"Got price from info: {ticker_data['current_price']}")
            
            # Try to get news (this is often less reliable, so we skip it to avoid rate limits)
            # ticker_data['news'] = stock.news[:3] if hasattr(stock, 'news') else []
            
        except Exception as e:
            print(f"Stock info failed: {e}")
            # If stock.info fails, try historical data as fallback
            print("Trying historical data as fallback...")
            try:
                time.sleep(3)  # Wait before fallback
                hist = stock.history(period="1d")
                if not hist.empty:
                    ticker_data['current_price'] = float(hist['Close'].iloc[-1])
                    print(f"Got price from history: {ticker_data['current_price']}")
                    # Create minimal info dict for analysis
                    ticker_data['info'] = {
                        'currentPrice': ticker_data['current_price'],
                        'regularMarketPrice': ticker_data['current_price'],
                        'fiftyTwoWeekHigh': ticker_data['current_price'] * 1.2,
                        'fiftyTwoWeekLow': ticker_data['current_price'] * 0.8,
                        'trailingPE': 20  # Default PE ratio
                    }
            except Exception as e2:
                print(f"Historical data fallback also failed: {e2}")
        
        # Cache the data
        ticker_cache[cache_key] = (ticker_data, now)
        return ticker_data

# Try to import required packages with error handling
try:
    import PyPDF2
except ImportError:
    raise ImportError("PyPDF2 is required. Install with: pip install PyPDF2==3.0.1")

try:
    from transformers import pipeline
except ImportError:
    raise ImportError("transformers is required. Install with: pip install transformers==4.36.0")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance==0.2.18")

# Import models with relative imports to avoid path issues
try:
    from app.models.analysis import (
        DocumentAnalysis,
        MarketAnalysis,
        SentimentScore,
        OptionsRecommendation,
        create_market_analysis
    )
except ImportError:
    # Fallback to absolute imports
    try:
        from app.models.analysis import (
            DocumentAnalysis, 
            SentimentScore,
            MarketAnalysis,
            OptionsRecommendation,
            create_market_analysis
        )
    except ImportError:
        # If both fail, we'll need to fix the import structure
        print("Warning: Could not import analysis models. Check your project structure.")
        from models.analysis import (
            DocumentAnalysis, 
            SentimentScore,
            MarketAnalysis,
            OptionsRecommendation,
            create_market_analysis
        )

router = APIRouter()

# Global variables for models (will be loaded on first use)
_summarizer = None
_sentiment_analyzer = None
_qa_model = None

def get_summarizer():
    """Lazy load summarizer to avoid startup delays"""
    global _summarizer
    if _summarizer is None:
        try:
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            print(f"Warning: Could not load summarizer: {e}")
            _summarizer = "unavailable"
    return _summarizer

def get_sentiment_analyzer():
    """Lazy load sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            _sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            print(f"Warning: Could not load sentiment analyzer: {e}")
            # Fallback to a simpler model
            try:
                _sentiment_analyzer = pipeline("sentiment-analysis")
            except Exception as e2:
                print(f"Warning: Could not load any sentiment analyzer: {e2}")
                _sentiment_analyzer = "unavailable"
    return _sentiment_analyzer

def get_qa_model():
    """Lazy load Q&A model"""
    global _qa_model
    if _qa_model is None:
        try:
            _qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
        except Exception as e:
            print(f"Warning: Could not load Q&A model: {e}")
            _qa_model = "unavailable"
    return _qa_model

@router.post("/analyze/document", response_model=DocumentAnalysis)
async def analyze_document(
    file: UploadFile = File(...),
    ticker: Optional[str] = None
):
    """
    Analyze an uploaded document (PDF) - typically SEC filings
    
    This endpoint:
    1. Extracts text from PDF
    2. Generates AI summary
    3. Performs sentiment analysis
    4. Extracts key points
    5. Identifies risk factors
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read PDF content
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages[:10]:  # Limit to first 10 pages for speed
            text += page.extract_text()
        
        # Clean text
        text = text.replace('\n', ' ').strip()
        
        if len(text) < 100:
            raise HTTPException(status_code=400, detail="Could not extract sufficient text from PDF")
        
        # Initialize response data with defaults
        summary_text = "Document processed successfully"
        sentiment_score = 0.0
        sentiment_confidence = 0.5
        key_points = []
        risk_factors = []
        
        # 1. Generate summary (with error handling)
        summarizer = get_summarizer()
        if summarizer != "unavailable":
            try:
                summary_text_input = text[:1024]
                summary = summarizer(summary_text_input, max_length=150, min_length=50, do_sample=False)
                summary_text = summary[0]['summary_text']
            except Exception as e:
                print(f"Summary generation failed: {e}")
                summary_text = f"Summary generation unavailable. Document contains {len(text)} characters."
        
        # 2. Sentiment analysis (with error handling)
        sentiment_analyzer = get_sentiment_analyzer()
        if sentiment_analyzer != "unavailable":
            try:
                sentiment_text = text[:512]
                sentiment_result = sentiment_analyzer(sentiment_text)[0]
                sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else -sentiment_result['score']
                sentiment_confidence = sentiment_result['score']
            except Exception as e:
                print(f"Sentiment analysis failed: {e}")
        
        # 3. Extract key points using Q&A (with error handling)
        qa_model = get_qa_model()
        if qa_model != "unavailable":
            try:
                questions = [
                    "What is the company's revenue?",
                    "What are the main risks?",
                    "What is the company's outlook?"
                ]
                
                for question in questions:
                    try:
                        answer = qa_model(question=question, context=text[:2048])
                        if answer['score'] > 0.1:
                            key_points.append(f"{question} {answer['answer']}")
                    except:
                        continue
            except Exception as e:
                print(f"Q&A extraction failed: {e}")
        
        # 4. Simple risk factor extraction (look for risk-related sentences)
        try:
            sentences = text.split('.')
            risk_keywords = ['risk', 'uncertain', 'adverse', 'decline', 'loss', 'litigation']
            
            for sentence in sentences[:50]:
                if any(keyword in sentence.lower() for keyword in risk_keywords):
                    if len(sentence) > 20 and len(sentence) < 200:
                        risk_factors.append(sentence.strip())
                        if len(risk_factors) >= 3:
                            break
        except Exception as e:
            print(f"Risk factor extraction failed: {e}")
        
        # 5. If ticker provided, get current market data
        financial_highlights = {}
        if ticker:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                financial_highlights = {
                    "market_cap": info.get("marketCap", "N/A"),
                    "pe_ratio": info.get("trailingPE", "N/A"),
                    "revenue": info.get("totalRevenue", "N/A"),
                    "profit_margin": info.get("profitMargins", "N/A")
                }
            except Exception as e:
                print(f"Financial data retrieval failed: {e}")
        
        # Create response
        return DocumentAnalysis(
            document_type="10-K",
            summary=summary_text,
            key_points=key_points[:5] if key_points else ["Document analysis completed"],
            sentiment=SentimentScore(
                score=sentiment_score,
                confidence=sentiment_confidence,
                source="document"
            ),
            risk_factors=risk_factors[:3] if risk_factors else ["No specific risk factors extracted"],
            financial_highlights=financial_highlights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")



@router.post("/analyze/ticker/{ticker}", response_model=MarketAnalysis)
async def analyze_ticker(ticker: str):
    """
    Quick market analysis for a ticker with options recommendations.
    This endpoint combines robust data fetching with detailed analysis.
    """
    try:
        print(f"Starting analysis for ticker: {ticker}")
        
        # Add delay to avoid rate limiting
        time.sleep(1) # Wait 1 second before making request
        
        # Use the new multi-API service with failover
        from app.services.market_data_service import market_data_service
        
        # Initialize service if needed
        if not hasattr(market_data_service, 'initialized'):
            await market_data_service.initialize()
            market_data_service.initialized = True
        
        # Get data using multi-API failover system
        try:
            comprehensive_analysis = await market_data_service.get_ticker_analysis(ticker)
            # Extract data for legacy format
            current_price = comprehensive_analysis.current_price
            info = {
                'currentPrice': current_price,
                'trailingPE': 20,  # Default PE for now
                'fiftyTwoWeekHigh': current_price * 1.2,
                'fiftyTwoWeekLow': current_price * 0.8,
                'fiftyDayAverage': current_price * 0.95
            }
            news = []  # Will be enhanced later
        except Exception as e:
            print(f"Multi-API service failed, falling back to old method: {e}")
            # Fallback to old cached method
            ticker_data = get_cached_ticker_data(ticker)
            current_price = ticker_data['current_price']
            info = ticker_data['info']
            news = ticker_data['news']
        
        # Final check for price
        if current_price is None or current_price == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not retrieve price data for {ticker}. This might be due to rate limiting or invalid ticker."
            )
            
        # If info wasn't fetched successfully before, we can't do deep analysis
        if not info:
             raise HTTPException(
                status_code=503, 
                detail=f"Could get price for {ticker}, but failed to get detailed info for full analysis."
            )

        # --- Complex Analysis Logic (from Code 1) ---

        # 1. Get recent news from cached data
        news = ticker_data['news']

        # 2. Analyze news sentiment
        sentiment_scores = []
        sentiment_analyzer = get_sentiment_analyzer()
        
        if sentiment_analyzer != "unavailable" and news:
            for article in news:
                try:
                    title = article.get('title', '')
                    if title:
                        sentiment = sentiment_analyzer(title)[0]
                        score = sentiment['score'] if sentiment['label'] in ['POSITIVE', 'positive'] else -sentiment['score']
                        sentiment_scores.append(
                            SentimentScore(
                                score=score,
                                confidence=sentiment['score'],
                                source="news"
                            )
                        )
                except Exception as e:
                    print(f"Error analyzing a news sentiment: {e}")
                    continue
        
        # 3. Calculate average sentiment
        avg_sentiment = sum(s.score for s in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # 4. Use complex recommendation logic based on sentiment and PE ratio
        pe_ratio = info.get('trailingPE', 20) # Default to 20 if PE is not available
        
        recommendation = OptionsRecommendation.HOLD # Default recommendation
        if avg_sentiment > 0.3 and pe_ratio < 25:
            recommendation = OptionsRecommendation.BUY
        elif avg_sentiment < -0.3 or pe_ratio > 40:
            recommendation = OptionsRecommendation.SELL
        elif abs(avg_sentiment) < 0.1:
            recommendation = OptionsRecommendation.IRON_CONDOR
        
        # 5. Get support/resistance from 52-week high/low
        support_levels = [info.get('fiftyTwoWeekLow', current_price * 0.9)]
        resistance_levels = [info.get('fiftyTwoWeekHigh', current_price * 1.1)]
        
        # 6. Generate key insights based on data
        insights = []
        if avg_sentiment > 0.2:
            insights.append("Positive news sentiment detected")
        if pe_ratio and pe_ratio < 15:
            insights.append(f"Attractive valuation with P/E of {pe_ratio:.1f}")
        if current_price < info.get('fiftyDayAverage', current_price):
            insights.append("Trading below 50-day moving average")
        if not insights:
            insights.append(f"Current price: ${current_price:.2f}")

        # --- Create Final Response ---
        analysis = create_market_analysis(
            symbol=ticker.upper(),
            current_price=current_price,
            recommendation=recommendation,
            confidence_score=0.75 # Higher confidence due to more data points
        )
        
        analysis.sentiment_scores = sentiment_scores
        analysis.support_levels = support_levels
        analysis.resistance_levels = resistance_levels
        analysis.key_insights = insights[:3]
        analysis.data_sources_used = ["yahoo_finance", "news_sentiment"] if news else ["yahoo_finance"]
        
        return analysis
        
    except HTTPException:
        raise # Re-raise HTTPException so FastAPI handles it
    except Exception as e:
        print(f"Critical error in analyze_ticker: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while analyzing ticker: {str(e)}")

@router.post("/qa/document")
async def question_answer(
    question: str,
    file: UploadFile = File(...)
):
    """
    Answer questions about an uploaded document
    
    Example questions:
    - "What is the company's revenue growth?"
    - "What are the main risks mentioned?"
    - "What is management's outlook?"
    """
    try:
        # Extract text from PDF
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages[:5]:
            text += page.extract_text()
        
        text = text.replace('\n', ' ').strip()
        
        # Use Q&A model
        qa_model = get_qa_model()
        if qa_model == "unavailable":
            return {
                "question": question,
                "answer": "Q&A model is currently unavailable",
                "confidence": 0.0,
                "source": file.filename
            }
        
        answer = qa_model(question=question, context=text[:2048])
        
        return {
            "question": question,
            "answer": answer['answer'],
            "confidence": answer['score'],
            "source": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Q&A: {str(e)}")

# Add a simple test endpoint
@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify the router is working"""
    return {"message": "Analysis router is working!", "timestamp": datetime.now()}