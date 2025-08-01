# 🚀 Quick Deployment Guide - FastAPI Batik Classifier

## ⚡ Quick Start (5 Minutes)

### 1. Local Development (Recommended for Testing)

```bash
# Navigasi ke direktori FastAPI
cd src/api/fastapi

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access API:**
- Health Check: http://localhost:8000/
- API Docs: http://localhost:8000/docs
- Prediction: POST http://localhost:8000/predict

### 2. Docker Deployment (Recommended for Production)

```bash
# Navigasi ke direktori FastAPI
cd src/api/fastapi

# Build Docker image
docker build -t batik-classifier-api .

# Run container
docker run -d -p 8000:8000 --name batik-api batik-classifier-api

# Check status
docker ps
docker logs batik-api
```

### 3. Automated Deployment Script

```bash
# Run deployment script
python deploy_fastapi.py --mode local
python deploy_fastapi.py --mode docker
python deploy_fastapi.py --mode heroku --app-name your-app-name
```

## 🧪 Testing API

### Quick Test
```bash
# Test health check
curl http://localhost:8000/

# Test prediction (dengan gambar)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/batik_image.jpg"
```

### Comprehensive Testing
```bash
# Run comprehensive test suite
python test_api.py --comprehensive

# Test specific image
python test_api.py --image your_batik_image.jpg

# Test performance
python test_api.py --performance 10
```

## 📊 API Endpoints

### Health Check
```bash
GET http://localhost:8000/
```
**Response:**
```json
{
  "message": "FastAPI Batik Classifier is running!"
}
```

### Prediction
```bash
POST http://localhost:8000/predict
Content-Type: multipart/form-data
```
**Request Body:**
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
      "Gedhangan": 0.0045
    }
  }
}
```

## 🎯 Model Information

- **Model**: Genetic Algorithm Optimized MobileNetV2
- **Accuracy**: 98.33%
- **Classes**: 60 Indonesian batik patterns
- **Input Size**: 224x224 pixels
- **Supported Formats**: JPG, PNG, JPEG
- **Max File Size**: 10MB
- **Response Time**: ~500ms average

## 🔧 Configuration

### Environment Variables
```bash
export MODEL_PATH="final_tuned_model.keras"
export LABELS_PATH="labels.txt"
export IMAGE_SIZE="224,224"
export MAX_FILE_SIZE="10485760"
export ALLOWED_EXTENSIONS="jpg,jpeg,png"
```

### Production Settings
```python
# main.py - Production configuration
import os

MODEL_PATH = os.getenv("MODEL_PATH", "final_tuned_model.keras")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
IMAGE_SIZE = tuple(map(int, os.getenv("IMAGE_SIZE", "224,224").split(",")))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png").split(",")
```

## 🚀 Cloud Deployment

### Heroku
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login and deploy
heroku login
heroku create batik-classifier-api
git add .
git commit -m "Deploy batik classifier"
git push heroku main
```

### Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### Render
1. Sign up at https://render.com
2. Connect GitHub repository
3. Create Web Service
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Google Cloud Run
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Deploy
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud run deploy batik-classifier-api \
  --source . \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated
```

## 📱 Frontend Integration

### HTML Form
```html
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
```

### React Component
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

## 🔒 Security

### CORS Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(file: UploadFile = File(...)):
    # ... prediction logic
```

### File Validation
```python
def validate_file(file: UploadFile):
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    ext = file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return True
```

## 📊 Monitoring

### Health Check
```python
@app.get("/health")
async def health_check():
    try:
        dummy_input = np.random.random((1, 224, 224, 3))
        _ = model.predict(dummy_input, verbose=0)
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Performance Metrics
```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

## 🎯 Troubleshooting

### Common Issues

#### 1. Model Loading Error
```bash
# Check model file exists
ls -la final_tuned_model.keras

# Check file permissions
chmod 644 final_tuned_model.keras
```

#### 2. Port Already in Use
```bash
# Use different port
uvicorn main:app --host 0.0.0.0 --port 8001

# Or kill existing process
lsof -ti:8000 | xargs kill -9
```

#### 3. Memory Issues
```bash
# Increase Docker memory
docker run -d -p 8000:8000 --memory=4g batik-classifier-api

# Or use production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 4. CORS Issues
```python
# Update CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Model file exists and valid
- [ ] Labels file matches model
- [ ] Dependencies installed
- [ ] Environment variables set
- [ ] Security settings configured

### Deployment
- [ ] Build Docker image (if using Docker)
- [ ] Test locally
- [ ] Deploy to cloud platform
- [ ] Verify endpoints work
- [ ] Test with batik images

### Post-Deployment
- [ ] Monitor logs
- [ ] Check performance metrics
- [ ] Test error handling
- [ ] Verify security settings
- [ ] Setup monitoring and alerting

## 🚀 Quick Commands

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

### Testing
```bash
# Quick test
curl http://localhost:8000/

# Comprehensive test
python test_api.py --comprehensive

# Performance test
python test_api.py --performance 10
```

---

**Model Performance**: 98.33% accuracy (Genetic Algorithm)
**API Response Time**: ~500ms average
**Supported Formats**: JPG, PNG, JPEG
**Max File Size**: 10MB
**Rate Limit**: 10 requests/minute

**Ready to Deploy! 🚀** 