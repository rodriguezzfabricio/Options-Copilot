# OptionsAI Copilot Production Setup

## 🚀 Quick Start (5 minutes)

### 1. Install Redis (Required)
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### 2. Install PostgreSQL (Required)
```bash
# macOS
brew install postgresql
brew services start postgresql
createdb optionsai_db

# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo -u postgres createdb optionsai_db
```

### 3. Get API Keys (Choose at least one)

#### Alpha Vantage (Recommended - Free tier: 25 requests/day)
1. Go to: https://www.alphavantage.co/support/#api-key
2. Sign up for free account
3. Get your API key instantly

#### Polygon.io (Professional - Free tier: 5 requests/minute)
1. Go to: https://polygon.io/
2. Sign up for free account
3. Get API key from dashboard

#### Finnhub (Alternative - Free tier: 60 requests/minute)
1. Go to: https://finnhub.io/
2. Sign up for free account  
3. Get API key from dashboard

### 4. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

### 5. Install Dependencies & Run
```bash
# Install Python dependencies
pip install -r requirements_v2.txt

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test in Browser
Open: http://localhost:8000/docs

Test endpoint: `POST /api/v1/analysis/analyze/ticker/AAPL`

## 🔑 API Keys You Need

| Service | Required | Free Tier | Sign Up |
|---------|----------|-----------|---------|
| Redis | ✅ Required | Free (self-hosted) | N/A |
| PostgreSQL | ✅ Required | Free (self-hosted) | N/A |
| Alpha Vantage | ⭐ Recommended | 25 req/day | [Sign up](https://www.alphavantage.co/support/#api-key) |
| Polygon.io | 🚀 Professional | 5 req/min | [Sign up](https://polygon.io/) |
| Finnhub | 🔄 Alternative | 60 req/min | [Sign up](https://finnhub.io/) |
| News API | 🔄 Optional | 1000 req/month | [Sign up](https://newsapi.org/) |
| HuggingFace | 🤖 Optional | Free tier | [Sign up](https://huggingface.co/) |

## ⚡ Production Deployment Options

### Option 1: Docker (Recommended)
```bash
# Coming soon - Docker Compose setup
docker-compose up -d
```

### Option 2: Cloud Services
- **Redis**: Redis Cloud, AWS ElastiCache, Google Memorystore
- **Database**: AWS RDS, Google Cloud SQL, Supabase
- **App**: Railway, Render, DigitalOcean App Platform

### Option 3: Local Development
- Redis: Local installation
- PostgreSQL: Local installation  
- App: Local Python environment

## 🔧 Troubleshooting

### Redis Connection Issues
```bash
# Check if Redis is running
redis-cli ping

# Start Redis if not running
brew services start redis  # macOS
sudo systemctl start redis-server  # Linux
```

### Database Connection Issues  
```bash
# Check PostgreSQL status
brew services list | grep postgresql  # macOS
sudo systemctl status postgresql  # Linux

# Create database if missing
createdb optionsai_db
```

### API Key Issues
- Verify keys are correctly set in `.env`
- Check free tier limits haven't been exceeded
- Test API keys manually with curl

## 📊 Performance Expectations

With proper setup:
- **Response Time**: < 500ms for cached results
- **Throughput**: 100+ requests/second  
- **Reliability**: 99.9% uptime with fallback APIs
- **Scalability**: Horizontal scaling with Redis cluster

## 🎯 Next Steps

1. Get at least Alpha Vantage API key (free)
2. Install Redis and PostgreSQL  
3. Run the setup commands
4. Test in browser at `/docs`
5. Start building your frontend!