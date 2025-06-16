from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
async def root():
    return {"message": "Welcome to OptionsAI API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}!"} 