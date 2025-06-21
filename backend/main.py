from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import routers
from app.api.v1.analysis import router as analysis_router
from app.api.v1.ticker_analysis_v2 import router as ticker_v2_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MVP: Simple in-memory cache instead of Redis
mvp_cache = {}
mvp_queue = asyncio.Queue()
mvp_workers = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting OptionsAI Copilot MVP...")
    
    # Start MVP background workers (3 workers)
    global mvp_workers
    for i in range(3):
        worker = asyncio.create_task(mvp_background_worker(f"worker-{i}"))
        mvp_workers.append(worker)
    
    logger.info("MVP workers started")
    logger.info("AI models ready (using HuggingFace free tier)")
    logger.info("OptionsAI Copilot MVP ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    for worker in mvp_workers:
        worker.cancel()
    logger.info("Shutdown complete")

async def mvp_background_worker(worker_id: str):
    """MVP background worker using in-memory queue"""
    logger.info(f"Starting worker {worker_id}")
    
    while True:
        try:
            # Get next job
            job = await mvp_queue.get()
            ticker = job['ticker']
            
            logger.info(f"{worker_id} processing {ticker}")
            
            # Simulate comprehensive analysis
            await asyncio.sleep(2)  # Simulate work
            
            # Cache result
            mvp_cache[f"analysis:{ticker}"] = {
                "symbol": ticker,
                "price": 150.0,  # MVP mock price
                "status": "completed",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info(f"{worker_id} completed {ticker}")
            mvp_queue.task_done()
            
        except Exception as e:
            logger.error(f"{worker_id} error: {e}")
            await asyncio.sleep(1)

app = FastAPI(
    title="OptionsAI Copilot MVP",
    description="MVP of AI-powered options trading platform - ready for production scaling",
    version="1.0.0-MVP",
    lifespan=lifespan
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP: Allow all origins for local testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include existing routers
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Document Analysis"])
app.include_router(ticker_v2_router, prefix="/api/v2/analysis", tags=["Ticker Analysis v2"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OptionsAI Copilot MVP",
        "version": "1.0.0-MVP",
        "status": "production-ready-architecture",
        "docs": "/docs",
        "features": [
            "Document analysis with AI",
            "Ticker analysis with caching",
            "Background job processing",
            "Production-ready patterns"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check - works without Redis/DB"""
    return {
        "status": "healthy",
        "version": "1.0.0-MVP", 
        "cache_items": len(mvp_cache),
        "queue_size": mvp_queue.qsize(),
        "workers": len(mvp_workers),
        "mode": "MVP"
    }

@app.get("/api/v2/ticker/{ticker}/quick")
async def quick_ticker_mvp(ticker: str):
    """MVP quick ticker endpoint"""
    
    # Check cache first
    cache_key = f"analysis:{ticker.upper()}"
    if cache_key in mvp_cache:
        cached = mvp_cache[cache_key]
        return {
            "symbol": ticker.upper(),
            "price": cached["price"],
            "status": "cached",
            "recommendation": "HOLD",
            "confidence": 0.75
        }
    
    # Queue for background processing
    await mvp_queue.put({"ticker": ticker.upper()})
    
    # Return immediate response
    return {
        "symbol": ticker.upper(),
        "price": 0.0,
        "status": "processing",
        "recommendation": "PENDING",
        "confidence": 0.5,
        "message": "Analysis queued - check back in a few seconds"
    }

@app.get("/api/v2/cache/stats")
async def mvp_cache_stats():
    """MVP cache stats"""
    return {
        "cache_items": len(mvp_cache),
        "queue_size": mvp_queue.qsize(),
        "active_workers": len([w for w in mvp_workers if not w.done()]),
        "mode": "MVP - in-memory cache",
        "ready_for_production": "Replace with Redis + PostgreSQL"
    }

@app.post("/api/v2/demo/load-test")
async def demo_load_test():
    """Demo endpoint to show queue processing"""
    
    demo_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
    
    # Queue multiple tickers
    for ticker in demo_tickers:
        await mvp_queue.put({"ticker": ticker})
    
    return {
        "message": f"Queued {len(demo_tickers)} tickers for analysis",
        "tickers": demo_tickers,
        "queue_size": mvp_queue.qsize(),
        "check_status": "/api/v2/cache/stats"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )