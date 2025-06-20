from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your analysis router
from app.api.v1.analysis import router as analysis_router

app = FastAPI(
    title="OptionsAI API",
    description="API for OptionsAI Copilot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the analysis router
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["analysis"])

@app.on_event("startup")
async def startup_event():
    print("Starting OptionsAI Copilot...")
    print("Loading AI models...")
    print("Ready to serve requests!")

@app.get("/")
async def root():
    return {"message": "Welcome to OptionsAI API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}!"}