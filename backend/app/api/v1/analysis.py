# backend/app/api/v1/analysis.py
"""
First real endpoint - SEC Filing Analysis with Q&A
This is your MVP feature to get started!
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
import PyPDF2
import io
from transformers import pipeline
import yfinance as yf
from datetime import datetime

from app.models.analysis import (
    DocumentAnalysis, 
    SentimentScore,
    MarketAnalysis,
    OptionsRecommendation,
    create_market_analysis
)

router = APIRouter()

# Initialize AI models (these load on startup)
# Using models that work well on CPU for development
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

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
        
        # 1. Generate summary (limit text to avoid token limits)
        summary_text = text[:1024]  # First 1024 chars for summary
        summary = summarizer(summary_text, max_length=150, min_length=50, do_sample=False)
        
        # 2. Sentiment analysis
        sentiment_text = text[:512]  # First 512 chars for sentiment
        sentiment_result = sentiment_analyzer(sentiment_text)[0]
        
        # Convert sentiment to our scale (-1 to 1)
        sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'positive' else -sentiment_result['score']
        
        # 3. Extract key points using Q&A
        key_points = []
        questions = [
            "What is the company's revenue?",
            "What are the main risks?",
            "What is the company's outlook?"
        ]
        
        for question in questions:
            try:
                answer = qa_model(question=question, context=text[:2048])
                if answer['score'] > 0.1:  # Confidence threshold
                    key_points.append(f"{question} {answer['answer']}")
            except:
                continue
        
        # 4. Simple risk factor extraction (look for risk-related sentences)
        risk_factors = []
        sentences = text.split('.')
        risk_keywords = ['risk', 'uncertain', 'adverse', 'decline', 'loss', 'litigation']
        
        for sentence in sentences[:50]:  # Check first 50 sentences
            if any(keyword in sentence.lower() for keyword in risk_keywords):
                if len(sentence) > 20 and len(sentence) < 200:
                    risk_factors.append(sentence.strip())
                    if len(risk_factors) >= 3:
                        break
        
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
            except:
                pass
        
        # Create response
        return DocumentAnalysis(
            document_type="10-K",  # You could detect this from content
            summary=summary[0]['summary_text'],
            key_points=key_points[:5],  # Limit to 5 key points
            sentiment=SentimentScore(
                score=sentiment_score,
                confidence=sentiment_result['score'],
                source="document"
            ),
            risk_factors=risk_factors[:3],  # Top 3 risk factors
            financial_highlights=financial_highlights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/analyze/ticker/{ticker}", response_model=MarketAnalysis)
async def analyze_ticker(ticker: str):
    """
    Quick market analysis for a ticker with options recommendations
    
    This endpoint:
    1. Gets current market data
    2. Analyzes recent news sentiment
    3. Provides AI-driven options recommendation
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        if current_price == 0:
            raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
        
        # Get recent news
        news = stock.news[:3] if hasattr(stock, 'news') else []
        
        # Analyze news sentiment
        sentiment_scores = []
        for article in news:
            title = article.get('title', '')
            if title:
                sentiment = sentiment_analyzer(title)[0]
                score = sentiment['score'] if sentiment['label'] == 'positive' else -sentiment['score']
                sentiment_scores.append(
                    SentimentScore(
                        score=score,
                        confidence=sentiment['score'],
                        source="news"
                    )
                )
        
        # Calculate average sentiment
        avg_sentiment = sum(s.score for s in sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        # Simple recommendation logic based on sentiment and PE ratio
        pe_ratio = info.get('trailingPE', 20)
        
        if avg_sentiment > 0.3 and pe_ratio < 25:
            recommendation = OptionsRecommendation.BUY
        elif avg_sentiment < -0.3 or pe_ratio > 40:
            recommendation = OptionsRecommendation.SELL
        elif abs(avg_sentiment) < 0.1:
            recommendation = OptionsRecommendation.IRON_CONDOR  # Neutral strategy
        else:
            recommendation = OptionsRecommendation.HOLD
        
        # Get support/resistance (simplified - using 52 week high/low)
        support_levels = [info.get('fiftyTwoWeekLow', current_price * 0.9)]
        resistance_levels = [info.get('fiftyTwoWeekHigh', current_price * 1.1)]
        
        # Key insights
        insights = []
        if avg_sentiment > 0.2:
            insights.append("Positive news sentiment detected")
        if pe_ratio < 15:
            insights.append(f"Attractive valuation with P/E of {pe_ratio:.1f}")
        if current_price < info.get('fiftyDayAverage', current_price):
            insights.append("Trading below 50-day moving average")
        
        # Create analysis
        analysis = create_market_analysis(
            symbol=ticker.upper(),
            current_price=current_price,
            recommendation=recommendation,
            confidence_score=0.7  # Moderate confidence for now
        )
        
        # Add additional data
        analysis.sentiment_scores = sentiment_scores
        analysis.support_levels = support_levels
        analysis.resistance_levels = resistance_levels
        analysis.key_insights = insights[:3]
        analysis.data_sources_used = ["yahoo_finance", "news_sentiment"]
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing ticker: {str(e)}")

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
        # Extract text from PDF (same as above)
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages[:5]:  # First 5 pages for Q&A
            text += page.extract_text()
        
        text = text.replace('\n', ' ').strip()
        
        # Use Q&A model
        answer = qa_model(question=question, context=text[:2048])  # Token limit
        
        return {
            "question": question,
            "answer": answer['answer'],
            "confidence": answer['score'],
            "source": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Q&A: {str(e)}")