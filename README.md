# Text Analysis Microservices Architecture

Arsitektur microservices untuk character screening dengan pemisahan antara API service dan Worker service untuk better scalability dan maintainability.

## 🏗️ Arsitektur

```
┌─────────────────────────────────────────────────────────────────┐
│                         GITHUB                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │ text-analysis-api        │  │ text-analysis-worker     │   │
│  │ - app.py                 │  │ - worker.py              │   │
│  │ - Dockerfile.api         │  │ - Dockerfile.worker      │   │
│  │ - api_service.yaml       │  │ - worker_service.yaml    │   │
│  │ - auth.py                │  │ - text_analysis_metrics  │   │
│  │ - security.py            │  │ - requirements.txt       │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────┬────────────────────────────┬────────────────────┘
              │ git push (trigger)         │ git push (trigger)
              ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD BUILD                                  │
│  cloudbuild.yaml - builds both services simultaneously         │
└─────────────┬────────────────────────────┬────────────────────┘
              │ deploy                     │ deploy
              ▼                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CLOUD RUN                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │ text-analysis-api        │  │ text-analysis-worker     │   │
│  │ - Public endpoint        │  │ - Internal service       │   │
│  │ - 2Gi RAM, 1 CPU         │  │ - 4Gi RAM, 2 CPU         │   │
│  │ - 0-10 instances         │  │ - 1-10 instances         │   │
│  │ - HTTP requests          │  │ - Pub/Sub processing     │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└─────────────┬────────────────────────────┬────────────────────┘
              │                            │
              │ ┌──────────────────────────┴──────────┐
              │ │     GOOGLE CLOUD SERVICES           │
              ├─┤ - Firestore (database)              │
              ├─┤ - Pub/Sub (messaging)                │
              ├─┤ - Secret Manager (credentials)       │
              └─┤ - Custom Search API (optional)       │
                └─────────────────────────────────────┘
```

## 🚀 Services

### 1. API Service (`text-analysis-api-service`)
- **Purpose**: Menerima HTTP requests untuk character screening
- **Features**:
  - Authentication & authorization
  - Input validation & sanitization
  - Rate limiting
  - Job creation & status tracking
  - Comprehensive audit logging
- **Endpoints**:
  - `POST /api/analyze-text/{analysis_type}` - Submit analysis
  - `GET /api/status/{job_id}` - Check job status
  - `GET /health` - Health check
  - `GET /metrics` - Performance metrics

### 2. Worker Service (`text-analysis-worker-service`)
- **Purpose**: Process character screening jobs asynchronously
- **Features**:
  - Pub/Sub message processing
  - OpenWebUI API integration
  - Custom search API integration
  - Fallback model support
  - Comprehensive metrics tracking
- **Capabilities**:
  - PEP (Politically Exposed Person) analysis
  - Negative news screening
  - Legal involvement analysis
  - Corporate screening

## 🔧 Setup & Deployment

### Prerequisites
- Google Cloud Project dengan billing enabled
- `gcloud` CLI installed dan authenticated
- Docker installed
- Required APIs enabled (Cloud Run, Cloud Build, Pub/Sub, Firestore)

### Quick Start

1. **Clone repository dan setup files**:
```bash
# Pastikan semua files ada di directory yang sama
ls -la
# Harus ada: app.py, worker.py, cloudbuild.yaml, dll.
```

2. **Run deployment script**:
```bash
chmod +x deploy.sh
./deploy.sh
```

3. **Atau manual deployment**:
```bash
# Set project
gcloud config set project your-project-id

# Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com pubsub.googleapis.com

# Build & deploy
gcloud builds submit --config cloudbuild.yaml
```

### Manual Configuration

#### 1. Pub/Sub Setup
```bash
# Create topic
gcloud pubsub topics create character-screening-request

# Create subscription
gcloud pubsub subscriptions create character-screening-worker-sub \
    --topic=character-screening-request \
    --ack-deadline=600
```

#### 2. Deploy API Service
```bash
gcloud run services replace api_service.yaml --region=asia-southeast2
```

#### 3. Deploy Worker Service
```bash
gcloud run services replace worker_service.yaml --region=asia-southeast2
```

## ⚙️ Configuration

### Environment Variables

#### API Service
- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `PUBSUB_TOPIC`: Pub/Sub topic name
- `FIRESTORE_DATABASE`: Firestore database name
- `REQUIRE_AUTH`: Enable/disable authentication
- `API_KEYS`: Comma-separated valid API keys
- `RATE_LIMIT_REQUESTS`: Rate limit per hour
- `RATE_LIMIT_BURST`: Burst limit per minute

#### Worker Service
- `GOOGLE_CLOUD_PROJECT`: GCP project ID
- `PUBSUB_SUBSCRIPTION`: Pub/Sub subscription name
- `OPENWEBUI_BASE_URL`: OpenWebUI API endpoint
- `OPENWEBUI_API_KEY`: OpenWebUI API key
- `CUSTOM_SEARCH_API_KEY`: Google Custom Search API key (optional)
- `CUSTOM_SEARCH_ENGINE_ID`: Custom Search Engine ID (optional)
- `MAX_CONCURRENT_JOBS`: Maximum concurrent processing jobs

### Model Configuration

```python
TEXT_MODEL_CONFIG = {
    "pep-analysis": {
        "model": "politically-exposed-person-v2",
        "entity_types": ["person"],
        "fallback": "negative-news"
    },
    "negative-news": {
        "model": "negative-news",
        "entity_types": ["person"],
        "fallback": None
    },
    # ... other models
}
```

## 📊 Monitoring & Observability

### Health Checks
- API Service: `GET /health`
- Worker Service: `GET /health`

### Metrics
- API Service: `GET /metrics`
- Worker Service: `GET /metrics`

### Logging
```bash
# API Service logs
gcloud logs tail --follow --resource-type=cloud_run_revision \
    --filter='resource.labels.service_name="text-analysis-api-service"'

# Worker Service logs
gcloud logs tail --follow --resource-type=cloud_run_revision \
    --filter='resource.labels.service_name="text-analysis-worker-service"'
```

## 🔐 Security Features

### API Service Security
- API key authentication
- Rate limiting (per hour + burst)
- Input validation & sanitization
- Audit logging
- CORS protection
- SQL injection prevention

### Worker Service Security
- Internal-only access
- Secure API communication
- Input validation
- Error handling & retries
- Resource limits

## 🧪 Testing

### API Testing
```bash
# Health check
curl -X GET "https://text-analysis-api-service-asia-southeast2.run.app/health"

# Submit analysis (requires API key)
curl -X POST "https://text-analysis-api-service-asia-southeast2.run.app/api/analyze-text/pep-analysis" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "entity_type": "person",
    "additional_context": "CEO of tech company"
  }'
```

### Load Testing
```bash
# Install hey for load testing
go install github.com/rakyll/hey@latest

# Load test
hey -n 100 -c 10 -H "Authorization: Bearer your-api-key" \
  -m POST -T "application/json" \
  -d '{"name":"Test User","entity_type":"person"}' \
  https://text-analysis-api-service-asia-southeast2.run.app/api/analyze-text/pep-analysis
```

## 🔄 CI/CD Pipeline

Pipeline otomatis melalui Cloud Build:
1. **Git Push** → Trigger Cloud Build
2. **Build** → Build Docker images untuk both services
3. **Test** → Run security & validation tests
4. **Deploy** → Deploy ke Cloud Run
5. **Verify** → Health checks & smoke tests

## 📈 Scaling

### API Service Scaling
- **Auto-scaling**: 0-10 instances
- **Target**: CPU utilization < 70%
- **Memory**: 2Gi per instance
- **Concurrency**: 80 requests per instance

### Worker Service Scaling
- **Auto-scaling**: 1-10 instances
- **Target**: Message queue depth
- **Memory**: 4Gi per instance
- **Concurrency**: 5 jobs per instance

## 🛠️ Troubleshooting

### Common Issues

1. **API Service not receiving requests**
   - Check Cloud Run service status
   - Verify IAM permissions
   - Check rate limiting settings

2. **Worker Service not processing jobs**
   - Verify Pub/Sub subscription
   - Check OpenWebUI API connectivity
   - Review worker logs for errors

3. **High latency**
   - Check model response times
   - Review concurrent job limits
   - Monitor resource usage

### Debug Commands
```bash
# Service status
gcloud run services list --region=asia-southeast2

# Service logs
gcloud logs read --limit=50 --format="table(timestamp,severity,textPayload)" \
  --resource-type=cloud_run_revision \
  --filter='resource.labels.service_name="text-analysis-api-service"'

# Pub/Sub metrics
gcloud pubsub subscriptions describe character-screening-worker-sub

# Resource usage
gcloud run services describe text-analysis-api-service --region=asia-southeast2
```

## 🎯 Performance Optimization

### API Service
- Connection pooling untuk database
- Response caching untuk status checks
- Request queuing untuk high-volume periods

### Worker Service
- Batch processing untuk multiple requests
- Model response caching
- Intelligent retry dengan exponential backoff

## 📝 Development

### Local Development
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run API service locally
uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# Run worker service locally
python worker.py
```

### Testing Locally
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

---

## 📞 Support

Untuk pertanyaan dan support, silakan:
1. Check logs menggunakan commands di atas
2. Review metrics di Cloud Console
3. Verifikasi configuration
4. Contact development team dengan error details

## 🔄 Updates

Untuk update sistem:
1. Update code di repository
2. Push changes ke main branch
3. Cloud Build akan otomatis deploy changes
4. Verify deployment melalui health checks
