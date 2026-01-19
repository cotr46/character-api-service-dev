# 🔍 Character Screening API Service

A comprehensive API service for character screening including PEP (Politically Exposed Person) analysis, negative news detection, and law involvement checking for individuals and corporate entities.

## 🎯 Overview

This service provides robust character screening capabilities with enterprise-grade security, authentication, rate limiting, and comprehensive audit logging.

### Key Features

- **PEP Analysis** - Political exposure screening for individuals and corporations
- **Negative News Detection** - Media screening for reputational risks
- **Law Involvement Analysis** - Legal and regulatory violation screening
- **Comprehensive Screening** - Combined screening across all risk categories
- **Real-time Processing** - Async job processing with status tracking
- **Enterprise Security** - API key authentication, rate limiting, input sanitization
- **Comprehensive Metrics** - Performance, security, and business metrics
- **Audit Logging** - Full audit trail for compliance and monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHARACTER SCREENING API                     │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   FastAPI       │  │   Security      │  │   Metrics       ││
│  │   Endpoints     │  │   Layer         │  │   Collection    ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────┬─────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD PLATFORM                       │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   Firestore     │  │   Pub/Sub       │  │   Cloud Run     ││
│  │   (Job Store)   │  │   (Messaging)   │  │   (Compute)     ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────┬─────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                SCREENING MODEL ENDPOINTS                       │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   PEP Analysis  │  │  Negative News  │  │ Law Involvement ││
│  │   Models        │  │   Models        │  │   Models        ││
│  └─────────────────┘  └─────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud Project with Firestore and Pub/Sub enabled
- Access to character screening model endpoints
- API keys configured

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd character-screening-api

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export OPENWEBUI_BASE_URL="https://your-model-endpoint"
export OPENWEBUI_API_KEY="your-model-api-key"
export API_KEYS="your-api-key-1,your-api-key-2"

# Run the service
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Docker Deployment

```bash
# Build the image
docker build -t character-screening-api .

# Run the container
docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT="your-project" \
  -e OPENWEBUI_BASE_URL="https://your-endpoint" \
  -e OPENWEBUI_API_KEY="your-key" \
  character-screening-api
```

## 📡 API Endpoints

### Core Screening Endpoints

#### Submit Character Screening Request
```http
POST /api/screen
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "name": "John Smith",
  "entity_type": "person",
  "screening_types": ["pep-analysis", "negative-news", "law-involvement"],
  "additional_context": "CEO of Example Corp",
  "jurisdiction": "US"
}
```

#### Submit Comprehensive Screening
```http
POST /api/screen/comprehensive
Content-Type: application/json
Authorization: Bearer your-api-key

{
  "name": "Example Corporation",
  "entity_type": "corporate",
  "jurisdiction": "US"
}
```

#### Check Job Status
```http
GET /api/status/{job_id}
Authorization: Bearer your-api-key
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Metrics
```http
GET /api/metrics
```

#### Available Models
```http
GET /api/models
```

## 🔒 Security Features

### Authentication
- API key authentication via Bearer token
- Configurable API keys via environment variables
- Request authentication logging

### Rate Limiting
- Per-client rate limiting (500 requests/hour by default)
- Burst protection (50 requests/minute)
- Rate limit headers in responses

### Input Sanitization
- Comprehensive input validation
- XSS prevention
- SQL injection protection
- Unicode normalization
- HTML entity encoding

### Audit Logging
- All sensitive operations logged
- Request tracking with hashed names for privacy
- Security violation monitoring
- Failed authentication tracking

## 📊 Monitoring & Metrics

### Available Metrics
- Request volume and latency
- Success/failure rates by screening type
- Model availability and response times
- Risk level distribution
- Security metrics
- Rate limiting statistics

### Health Monitoring
- Service health checks
- Model endpoint availability
- Google Cloud service status
- Resource utilization tracking

## 🔧 Configuration

### Environment Variables

#### Required
```bash
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
OPENWEBUI_BASE_URL=https://your-model-endpoint
OPENWEBUI_API_KEY=your-model-api-key
```

#### Authentication & Security
```bash
REQUIRE_AUTH=true
API_KEYS=key1,key2,key3
AUDIT_SENSITIVE_OPERATIONS=true
RATE_LIMIT_REQUESTS=500
RATE_LIMIT_WINDOW=3600
RATE_LIMIT_BURST=50
```

#### Service Configuration
```bash
PORT=8080
FIRESTORE_DATABASE=character-screening-db
PUBSUB_TOPIC=character-screening-processing
TEXT_MODEL_TIMEOUT_SECONDS=45
TEXT_MODEL_RETRY_ATTEMPTS=3
```

## 🎭 Screening Types

### PEP Analysis
- Political exposure screening
- Government position detection
- Politically connected person identification
- Corporate political ties analysis

### Negative News Detection
- Media scandal screening
- Controversy detection
- Reputational risk assessment
- Financial misconduct identification

### Law Involvement Analysis
- Criminal charges screening
- Civil litigation detection
- Regulatory sanctions identification
- Compliance violation tracking

### Comprehensive Screening
- All screening types combined
- Risk level aggregation
- Overall risk assessment
- Complete risk profile generation

## 📈 Risk Levels

- **LOW** - No significant risks identified
- **MEDIUM** - Some risks present, further review recommended
- **HIGH** - Significant risks identified, detailed investigation required
- **CRITICAL** - Severe risks present, immediate attention needed

## 🚀 Deployment

### Cloud Build Deployment

1. Ensure Cloud Build API is enabled
2. Configure build triggers in Google Cloud Console
3. Push to main branch to trigger deployment

```bash
gcloud builds submit --config cloudbuild.yaml
```

### Cloud Run Deployment

```bash
gcloud run services replace character-screening-service.yaml \
  --region=asia-southeast2
```

### Manual Docker Deployment

```bash
# Build and push
docker build -t gcr.io/your-project/character-screening-api .
docker push gcr.io/your-project/character-screening-api

# Deploy to Cloud Run
gcloud run deploy character-screening-api \
  --image=gcr.io/your-project/character-screening-api \
  --region=asia-southeast2 \
  --platform=managed
```

## 🔍 Usage Examples

### Python Client Example

```python
import requests
import time

# Configuration
API_BASE_URL = "https://your-character-screening-api.run.app"
API_KEY = "your-api-key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Submit screening request
screening_data = {
    "name": "John Doe",
    "entity_type": "person",
    "screening_types": ["pep-analysis", "negative-news"],
    "jurisdiction": "US"
}

response = requests.post(
    f"{API_BASE_URL}/api/screen",
    headers=headers,
    json=screening_data
)

if response.status_code == 200:
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"Screening submitted: {job_id}")
    
    # Poll for results
    while True:
        status_response = requests.get(
            f"{API_BASE_URL}/api/status/{job_id}",
            headers=headers
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                print("Screening completed!")
                print(status_data["results"])
                break
            elif status_data["status"] == "failed":
                print("Screening failed!")
                print(status_data["error"])
                break
            else:
                print(f"Status: {status_data['status']}")
                time.sleep(5)
```

### cURL Examples

```bash
# Submit comprehensive screening
curl -X POST "https://your-api.run.app/api/screen/comprehensive" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Example Corp",
    "entity_type": "corporate",
    "jurisdiction": "US"
  }'

# Check status
curl -H "Authorization: Bearer your-api-key" \
  "https://your-api.run.app/api/status/job-id-here"

# Get metrics
curl "https://your-api.run.app/api/metrics"
```

## 🛠️ Development

### Local Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8080

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

### Testing

```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Load tests
python -m pytest tests/load/
```

## 📝 Changelog

### Version 2.0.0
- Complete refactor focused on character screening
- Removed document image processing capabilities
- Enhanced security and authentication
- Improved metrics and monitoring
- Added comprehensive screening support
- Updated API design and documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Check the API documentation at `/api/docs`
- Review health status at `/health`
- Check metrics at `/api/metrics`
- Contact: support@your-organization.com

## 🔮 Future Enhancements

- Machine learning risk scoring
- Real-time webhook notifications
- Advanced reporting and analytics
- Multi-language support
- Enhanced false positive detection
- Batch processing capabilities
- Custom screening rule configuration
