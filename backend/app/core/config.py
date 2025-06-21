# backend/app/core/config.py
"""
Configuration settings for OptionsAI Copilot
"""

import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/optionsai")
    
    # Redis for caching and queues
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Market Data APIs
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    POLYGON_API_KEY: Optional[str] = os.getenv("POLYGON_API_KEY")
    FINNHUB_API_KEY: Optional[str] = os.getenv("FINNHUB_API_KEY")
    
    # News APIs
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")
    
    # HuggingFace
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    # Application settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    class Config:
        env_file = ".env"

settings = Settings()