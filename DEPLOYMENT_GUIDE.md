# 🚀 Panduan Deployment FastAPI - Batik Classification Service

## 📋 Overview

Panduan lengkap untuk mendeploy FastAPI service untuk klasifikasi citra batik Indonesia menggunakan model yang telah dioptimalkan dengan Genetic Algorithm (98.33% accuracy).

## 🏗️ Struktur Project

```
src/api/fastapi/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── final_tuned_model.keras  # Model terbaik (Genetic Algorithm)
└── labels.txt          # Label kelas batik (60 classes)
```

## 🎯 Metode Deployment

### 1. Local Development Deployment

#### Langkah 1: Setup Environment
```bash
# Navigasi ke direktori FastAPI
cd src/api/fastapi

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Langkah 2: Run FastAPI Server
```bash
# Development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production server (tanpa reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Langkah 3: Test API
```bash
# Test endpoint root
curl http://localhost:8000/

# Test prediction (dengan gambar)
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/batik_image.jpg"
```

### 2. Docker Deployment

#### Langkah 1: Build Docker Image
```bash
# Navigasi ke direktori FastAPI
cd src/api/fastapi

# Build Docker image
docker build -t batik-classifier-api .

# Verifikasi image
docker images | grep batik-classifier-api
```

#### Langkah 2: Run Docker Container
```bash
# Run container
docker run -d -p 8000:8000 --name batik-api batik-classifier-api

# Check container status
docker ps

# View logs
docker logs batik-api
```

#### Langkah 3: Stop dan Remove Container
```bash
# Stop container
docker stop batik-api

# Remove container
docker rm batik-api
```

### 3. Cloud Deployment

#### A. Heroku Deployment

##### Langkah 1: Install Heroku CLI
```bash
# Download dan install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli
```

##### Langkah 2: Prepare for Heroku
```bash
# Buat file Procfile
echo "web: uvicorn main:app --host=0.0.0.0 --port=\$PORT" > Procfile

# Buat runtime.txt
echo "python-3.11.0" > runtime.txt
```

##### Langkah 3: Deploy to Heroku
```bash
# Login ke Heroku
heroku login

# Buat app baru
heroku create batik-classifier-api

# Add git remote
heroku git:remote -a batik-classifier-api

# Deploy
git add .
git commit -m "Deploy batik classifier API"
git push heroku main

# Check logs
heroku logs --tail
```

#### B. Railway Deployment

##### Langkah 1: Setup Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login ke Railway
railway login
```

##### Langkah 2: Deploy
```bash
# Initialize project
railway init

# Deploy
railway up

# Get deployment URL
railway domain
```

#### C. Render Deployment

##### Langkah 1: Create Render Account
- Sign up di https://render.com
- Connect GitHub repository

##### Langkah 2: Create Web Service
- Service Type: Web Service
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

#### D. Google Cloud Run

##### Langkah 1: Setup Google Cloud
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

##### Langkah 2: Deploy to Cloud Run
```bash
# Build dan deploy
gcloud run deploy batik-classifier-api \
  --source . \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

#### E. AWS Elastic Beanstalk

##### Langkah 1: Install EB CLI
```bash
pip install awsebcli
```

##### Langkah 2: Initialize EB
```bash
# Initialize EB application
eb init -p python-3.11 batik-classifier-api

# Create environment
eb create batik-classifier-env
```

##### Langkah 3: Deploy
```bash
# Deploy
eb deploy

# Open application
eb open
```

## 🔧 Configuration

### Environment Variables
```bash
# Production settings
export MODEL_PATH="final_tuned_model.keras"
export LABELS_PATH="labels.txt"
export IMAGE_SIZE="224,224"
export MAX_FILE_SIZE="10485760"  # 10MB
export ALLOWED_EXTENSIONS="jpg,jpeg,png"
```

### Production Settings
```python
# main.py - Production configuration
import os

# Load from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "final_tuned_model.keras")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
IMAGE_SIZE = tuple(map(int, os.getenv("IMAGE_SIZE", "224,224").split(",")))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png").split(",")
```

### Security Configuration
```python
# CORS settings untuk production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Ganti dengan domain Anda
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📊 API Endpoints

### 1. Health Check
```bash
GET /
```
**Response:**
```json
{
  "message": "FastAPI Batik Classifier is running!"
}
```

### 2. Prediction
```bash
POST /predict
Content-Type: multipart/form-data
```
**Request:**
- `file`: Image file (JPG, PNG, JPEG)

**Response:**
```json
{
  "success": true,
  "data": {
    "class_name": "Sekar Pijetan",
    "confidence": 0.9833,
    "probabilities": {
      "Sekar Pijetan": 0.9833,
      "Sekar Pacar": 0.0089,
      "Gedhangan": 0.0045,
      "Sekar Keben": 0.0021,
      "Sekar Jali": 0.0012
    }
  }
}
```

## 🧪 Testing

### 1. Manual Testing dengan curl
```bash
# Test health check
curl http://localhost:8000/

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_batik.jpg"
```

### 2. Testing dengan Python
```python
import requests

# Test health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Test prediction
with open("test_batik.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

### 3. Testing dengan Postman
1. Import collection ke Postman
2. Set base URL: `http://localhost:8000`
3. Test endpoints:
   - GET `/`
   - POST `/predict` dengan file upload

## 📈 Monitoring & Logging

### 1. Add Logging
```python
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request for file: {file.filename}")
    # ... prediction logic
    logger.info(f"Prediction completed: {top_prediction['label']}")
    return result
```

### 2. Add Metrics
```python
from fastapi import Request
import time

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

### 3. Health Check dengan Model Status
```python
@app.get("/health")
async def health_check():
    try:
        # Test model prediction
        dummy_input = np.random.random((1, 224, 224, 3))
        _ = model.predict(dummy_input, verbose=0)
        return {
            "status": "healthy",
            "model_loaded": True,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
```

## 🔒 Security Best Practices

### 1. File Validation
```python
def validate_file(file: UploadFile):
    # Check file size
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Check file extension
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return True
```

### 2. Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile = File(...)):
    # ... prediction logic
```

### 3. API Key Authentication
```python
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_api_key(credentials: HTTPBearer = Depends(security)):
    if credentials.credentials != "your-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    # ... prediction logic
```

## 🚀 Performance Optimization

### 1. Model Caching
```python
import joblib

# Cache model predictions
@lru_cache(maxsize=1000)
def cached_predict(image_hash: str):
    # ... prediction logic
```

### 2. Async Processing
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Process in thread pool
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, predict_sync, file)
    return result
```

### 3. Response Compression
```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## 📱 Frontend Integration

### 1. HTML Form
```html
<!DOCTYPE html>
<html>
<head>
    <title>Batik Classifier</title>
</head>
<body>
    <form id="uploadForm">
        <input type="file" id="imageFile" accept="image/*" required>
        <button type="submit">Classify Batik</button>
    </form>
    <div id="result"></div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('imageFile');
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                document.getElementById('result').innerHTML = 
                    `<h3>Result: ${result.data.class_name}</h3>
                     <p>Confidence: ${(result.data.confidence * 100).toFixed(2)}%</p>`;
            } catch (error) {
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
```

### 2. React Component
```jsx
import React, { useState } from 'react';

function BatikClassifier() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input 
                    type="file" 
                    onChange={(e) => setFile(e.target.files[0])}
                    accept="image/*"
                    required
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Classifying...' : 'Classify Batik'}
                </button>
            </form>
            
            {result && (
                <div>
                    <h3>Result: {result.data.class_name}</h3>
                    <p>Confidence: {(result.data.confidence * 100).toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}

export default BatikClassifier;
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Error
```bash
# Error: Model file not found
# Solution: Check model path
ls -la final_tuned_model.keras
```

#### 2. Memory Issues
```bash
# Error: Out of memory
# Solution: Increase memory limit
docker run -d -p 8000:8000 --memory=4g batik-classifier-api
```

#### 3. Port Already in Use
```bash
# Error: Port 8000 already in use
# Solution: Use different port
uvicorn main:app --host 0.0.0.0 --port 8001
```

#### 4. CORS Issues
```python
# Error: CORS policy violation
# Solution: Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📊 Performance Monitoring

### 1. Add Prometheus Metrics
```python
from prometheus_fastapi_instrumentator import Instrumentator

# Add metrics
Instrumentator().instrument(app).expose(app)
```

### 2. Add Custom Metrics
```python
from prometheus_client import Counter, Histogram

# Custom metrics
prediction_counter = Counter('batik_predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # ... prediction logic
    
    duration = time.time() - start_time
    prediction_counter.inc()
    prediction_duration.observe(duration)
    
    return result
```

## 🎯 Deployment Checklist

### Pre-Deployment
- [ ] Model file (`final_tuned_model.keras`) ada dan valid
- [ ] Labels file (`labels.txt`) sesuai dengan model
- [ ] Dependencies di `requirements.txt` lengkap
- [ ] Environment variables dikonfigurasi
- [ ] Security settings diatur

### Deployment
- [ ] Build Docker image (jika menggunakan Docker)
- [ ] Test locally
- [ ] Deploy ke cloud platform
- [ ] Verify endpoints berfungsi
- [ ] Test dengan gambar batik

### Post-Deployment
- [ ] Monitor logs
- [ ] Check performance metrics
- [ ] Test error handling
- [ ] Verify security settings
- [ ] Setup monitoring dan alerting

## 🚀 Quick Start Commands

### Local Development
```bash
cd src/api/fastapi
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
cd src/api/fastapi
docker build -t batik-classifier-api .
docker run -d -p 8000:8000 batik-classifier-api
```

### Heroku Deployment
```bash
cd src/api/fastapi
heroku create batik-classifier-api
git add .
git commit -m "Deploy batik classifier"
git push heroku main
```

---

**Model Performance**: 98.33% accuracy (Genetic Algorithm)
**API Response Time**: ~500ms average
**Supported Formats**: JPG, PNG, JPEG
**Max File Size**: 10MB
**Rate Limit**: 10 requests/minute

**Ready to Deploy! 🚀** 