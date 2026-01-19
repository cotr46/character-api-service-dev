"""
Character Screening API Service
Focused API for character screening including PEP analysis, negative news, and law involvement checking
Enhanced with comprehensive security, authentication, and monitoring
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import os
import json
import time
import uuid
import asyncio
import aiohttp
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any

# Google Cloud imports
from google.cloud import storage, pubsub_v1, firestore
from google.api_core import exceptions as gcp_exceptions

# Security imports
from security import InputSanitizer, SecurityViolationType, security_metrics
from auth import rate_limiter, AuthConfig, AuditLogger, get_client_id, get_request_info, hash_sensitive_data
from character_screening_metrics import character_screening_metrics
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Initialize security
security = HTTPBearer(auto_error=False)

# Initialize FastAPI app
app = FastAPI(
    title="Character Screening API",
    description="Comprehensive character screening service for PEP analysis, negative news detection, and law involvement checking",
    version="2.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
PUBSUB_TOPIC = os.getenv("PUBSUB_TOPIC") 
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE")

# Text analysis model endpoints
OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY")
TEXT_MODEL_TIMEOUT_SECONDS = int(os.getenv("TEXT_MODEL_TIMEOUT_SECONDS", "30"))
TEXT_MODEL_RETRY_ATTEMPTS = int(os.getenv("TEXT_MODEL_RETRY_ATTEMPTS", "3"))

# Initialize Google Cloud clients
if PROJECT_ID:
    storage_client = storage.Client(project=PROJECT_ID)
    publisher = pubsub_v1.PublisherClient()
    firestore_client = firestore.Client(project=PROJECT_ID, database=FIRESTORE_DATABASE)
    pubsub_topic_path = publisher.topic_path(PROJECT_ID, PUBSUB_TOPIC)
else:
    storage_client = None
    publisher = None
    firestore_client = None
    pubsub_topic_path = None

print(f"🚀 Character Screening API initialized with project: {PROJECT_ID}")

async def authenticate_request(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Authenticate API request and apply rate limiting
    """
    request_info = get_request_info(request)
    client_id = get_client_id(request, credentials)
    
    # Check authentication if required
    if AuthConfig.REQUIRE_AUTH:
        if not credentials or not credentials.credentials:
            AuditLogger.log_authentication_event(
                "missing_credentials", None, False, request_info["ip"],
                {"reason": "No credentials provided"}
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Authentication required",
                    "message": "API key must be provided in Authorization header",
                    "format": "Bearer <api_key>"
                }
            )
        
        if credentials.credentials not in AuthConfig.VALID_API_KEYS:
            AuditLogger.log_authentication_event(
                "invalid_key", client_id, False, request_info["ip"],
                {"reason": "Invalid API key"}
            )
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid"
                }
            )
        
        AuditLogger.log_authentication_event(
            "success", client_id, True, request_info["ip"],
            {"method": "api_key"}
        )
    
    # Apply rate limiting
    allowed, rate_info = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        AuditLogger.log_rate_limit_violation(
            client_id, request_info["ip"], "api_request", rate_info
        )
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": rate_info.get("error", "Too many requests"),
                "retry_after": rate_info.get("retry_after", 3600),
                "limit": rate_info.get("limit"),
                "window": rate_info.get("window")
            },
            headers={"Retry-After": str(int(rate_info.get("retry_after", 3600)))}
        )
    
    return {
        "client_id": client_id,
        "authenticated": AuthConfig.REQUIRE_AUTH,
        "rate_limit": rate_info,
        "request_info": request_info
    }

def log_character_screening_audit(
    auth_info: Dict[str, Any],
    screening_type: str,
    entity_type: str,
    name: str,
    job_id: str
):
    """Log character screening request for audit purposes"""
    if AuthConfig.AUDIT_SENSITIVE_OPERATIONS:
        AuditLogger.log_text_analysis_request(
            client_id=auth_info["client_id"],
            analysis_type=screening_type,
            entity_type=entity_type,
            name_hash=hash_sensitive_data(name),
            job_id=job_id,
            request_ip=auth_info["request_info"]["ip"],
            user_agent=auth_info["request_info"]["user_agent"]
        )

# Character screening enums
class ScreeningType(str, Enum):
    PEP_ANALYSIS = "pep-analysis"
    NEGATIVE_NEWS = "negative-news"
    LAW_INVOLVEMENT = "law-involvement"
    COMPREHENSIVE = "comprehensive"  # Runs all applicable screenings

class EntityType(str, Enum):
    PERSON = "person"
    CORPORATE = "corporate"

class JobStatus(str, Enum):
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Character screening model configuration
SCREENING_MODEL_CONFIG = {
    "pep-analysis": {
        "person": {
            "model": "politically-exposed-person-v2",
            "description": "Political Exposure Person Analysis v2",
            "risk_categories": ["political_exposure", "government_position", "politically_connected"]
        },
        "corporate": {
            "model": "politically-exposed-person-corporate-v2",
            "description": "Corporate Political Exposure Analysis",
            "risk_categories": ["corporate_political_ties", "government_contracts", "regulatory_exposure"]
        }
    },
    "negative-news": {
        "person": {
            "model": "negative-news-person",
            "description": "Individual Negative News Analysis",
            "risk_categories": ["scandal", "controversy", "legal_issues", "reputation_damage"]
        },
        "corporate": {
            "model": "negative-news-corporate",
            "description": "Corporate Negative News Analysis", 
            "risk_categories": ["corporate_scandal", "regulatory_violations", "financial_misconduct", "environmental_issues"]
        }
    },
    "law-involvement": {
        "person": {
            "model": "law-involvement-person",
            "description": "Individual Legal Involvement Analysis",
            "risk_categories": ["criminal_charges", "civil_litigation", "regulatory_sanctions", "court_cases"]
        },
        "corporate": {
            "model": "law-involvement-corporate", 
            "description": "Corporate Legal Involvement Analysis",
            "risk_categories": ["corporate_litigation", "regulatory_enforcement", "compliance_violations", "legal_settlements"]
        }
    }
}

# Pydantic models
class CharacterScreeningRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Name to screen (person or corporate entity)")
    entity_type: EntityType = Field(..., description="Type of entity being screened")
    screening_types: List[ScreeningType] = Field(default=[ScreeningType.COMPREHENSIVE], description="Types of screening to perform")
    additional_context: Optional[str] = Field(None, max_length=1000, description="Optional additional context or aliases")
    jurisdiction: Optional[str] = Field(None, max_length=100, description="Specific jurisdiction or country of interest")
    
    @validator('name')
    def validate_name(cls, v):
        """Enhanced name validation with comprehensive security checks"""
        try:
            sanitized_name = InputSanitizer.sanitize_name(v, "name")
            security_metrics.record_validation(True)
            return sanitized_name
        except ValueError as e:
            security_metrics.record_validation(False, SecurityViolationType.INVALID_CHARACTERS)
            raise e
    
    @validator('additional_context')
    def validate_context(cls, v):
        """Enhanced context validation with security checks"""
        try:
            sanitized_context = InputSanitizer.sanitize_context(v)
            security_metrics.record_validation(True)
            return sanitized_context
        except ValueError as e:
            security_metrics.record_validation(False, SecurityViolationType.INVALID_CHARACTERS)
            raise e

class ScreeningResult(BaseModel):
    screening_type: str
    risk_level: RiskLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    findings: List[Dict[str, Any]]
    sources: List[str]
    last_updated: str

class CharacterScreeningResponse(BaseModel):
    job_id: str
    status: JobStatus
    name: str
    entity_type: str
    screening_results: List[ScreeningResult]
    overall_risk_level: RiskLevel
    completion_time: Optional[str]
    processing_duration: Optional[float]

async def check_model_availability(model_name: str) -> bool:
    """Check if a screening model is available"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            url = f"{OPENWEBUI_BASE_URL}/api/models"
            headers = {"Authorization": f"Bearer {OPENWEBUI_API_KEY}"}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [model.get("id", "") for model in models_data.get("data", [])]
                    return model_name in available_models
                return False
    except Exception as e:
        print(f"❌ Model availability check failed for {model_name}: {str(e)}")
        return False

async def validate_screening_request(screening_types: List[ScreeningType], entity_type: EntityType, name: str) -> Dict[str, Any]:
    """Validate screening request and check model availability"""
    
    validation_result = {
        "valid": True,
        "available_screenings": [],
        "unavailable_screenings": [],
        "fallback_recommendations": []
    }
    
    # Handle comprehensive screening
    if ScreeningType.COMPREHENSIVE in screening_types:
        screening_types = [ScreeningType.PEP_ANALYSIS, ScreeningType.NEGATIVE_NEWS, ScreeningType.LAW_INVOLVEMENT]
    
    for screening_type in screening_types:
        if screening_type == ScreeningType.COMPREHENSIVE:
            continue
            
        # Get model configuration
        model_config = SCREENING_MODEL_CONFIG.get(screening_type.value, {}).get(entity_type.value)
        
        if not model_config:
            validation_result["unavailable_screenings"].append({
                "screening_type": screening_type.value,
                "reason": f"No model configured for {screening_type.value} on {entity_type.value}"
            })
            continue
        
        model_name = model_config["model"]
        
        # Check model availability
        is_available = await check_model_availability(model_name)
        character_screening_metrics.record_model_availability_check(model_name, is_available)
        
        if is_available:
            validation_result["available_screenings"].append({
                "screening_type": screening_type.value,
                "model": model_name,
                "description": model_config["description"]
            })
        else:
            validation_result["unavailable_screenings"].append({
                "screening_type": screening_type.value,
                "model": model_name,
                "reason": "Model endpoint unavailable"
            })
    
    # Set validation status
    validation_result["valid"] = len(validation_result["available_screenings"]) > 0
    
    if not validation_result["valid"]:
        validation_result["fallback_recommendations"] = [
            "Try again later when models are available",
            "Contact support for alternative screening options",
            "Check if the entity type is correctly specified"
        ]
    
    return validation_result

def create_screening_job_record(job_id: str, screening_types: List[str], entity_type: str, name: str, 
                              additional_context: Optional[str], jurisdiction: Optional[str]) -> Dict[str, Any]:
    """Create Firestore job record for character screening"""
    
    job_data = {
        "job_id": job_id,
        "job_type": "character_screening",
        "status": JobStatus.SUBMITTED.value,
        "screening_types": screening_types,
        "entity_type": entity_type,
        "name": name,
        "additional_context": additional_context,
        "jurisdiction": jurisdiction,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "retry_count": 0,
        "version": "2.0.0"
    }
    
    try:
        if firestore_client:
            doc_ref = firestore_client.collection("character_screening_jobs").document(job_id)
            doc_ref.set(job_data)
            print(f"📝 Character screening job record created: {job_id}")
        return job_data
    except Exception as e:
        print(f"❌ Failed to create job record: {str(e)}")
        raise e

def publish_screening_message(job_id: str, screening_types: List[str], entity_type: str, name: str,
                            additional_context: Optional[str], jurisdiction: Optional[str]) -> str:
    """Publish message to Pub/Sub for worker processing"""
    
    message_data = {
        "job_id": job_id,
        "job_type": "character_screening",
        "screening_types": screening_types,
        "entity_type": entity_type,
        "name": name,
        "additional_context": additional_context,
        "jurisdiction": jurisdiction,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    }
    
    try:
        if publisher and pubsub_topic_path:
            message_json = json.dumps(message_data)
            future = publisher.publish(pubsub_topic_path, message_json.encode('utf-8'))
            message_id = future.result()
            print(f"📤 Published screening message: {message_id}")
            return message_id
        else:
            print("⚠️ Pub/Sub not configured - job submitted for local processing")
            return "local-processing"
    except Exception as e:
        print(f"❌ Failed to publish message: {str(e)}")
        raise e

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Character Screening API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "screen": "/api/screen",
            "comprehensive": "/api/screen/comprehensive",
            "status": "/api/status/{job_id}",
            "metrics": "/api/metrics",
            "docs": "/api/docs"
        },
        "supported_screenings": ["pep-analysis", "negative-news", "law-involvement", "comprehensive"],
        "supported_entities": ["person", "corporate"]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with model availability"""
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "character-screening-api",
        "version": "2.0.0"
    }
    
    # Check Google Cloud services
    gcp_services = {}
    
    # Check Firestore
    try:
        if firestore_client:
            # Quick test query
            test_collection = firestore_client.collection("health_check").limit(1)
            list(test_collection.stream())
            gcp_services["firestore"] = "healthy"
        else:
            gcp_services["firestore"] = "not_configured"
    except Exception as e:
        gcp_services["firestore"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Pub/Sub
    try:
        if publisher and pubsub_topic_path:
            # Check if topic exists
            publisher.get_topic(request={"topic": pubsub_topic_path})
            gcp_services["pubsub"] = "healthy"
        else:
            gcp_services["pubsub"] = "not_configured"
    except Exception as e:
        gcp_services["pubsub"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check screening models availability
    model_health = {}
    total_models = 0
    healthy_models = 0
    
    for screening_type, entity_configs in SCREENING_MODEL_CONFIG.items():
        for entity_type, config in entity_configs.items():
            model_name = config["model"]
            total_models += 1
            is_available = await check_model_availability(model_name)
            model_health[f"{screening_type}_{entity_type}"] = {
                "model": model_name,
                "status": "healthy" if is_available else "unavailable"
            }
            if is_available:
                healthy_models += 1
    
    # Overall model availability
    if healthy_models == 0:
        health_status["status"] = "unhealthy"
    elif healthy_models < total_models:
        health_status["status"] = "degraded"
    
    health_status.update({
        "gcp_services": gcp_services,
        "screening_models": {
            "healthy_models": healthy_models,
            "total_models": total_models,
            "availability_rate": round((healthy_models / max(total_models, 1)) * 100, 1),
            "models": model_health
        },
        "response_time_ms": round((time.time() - start_time) * 1000, 2)
    })
    
    # Return appropriate HTTP status
    if health_status["status"] == "unhealthy":
        return JSONResponse(content=health_status, status_code=503)
    elif health_status["status"] == "degraded":
        return JSONResponse(content=health_status, status_code=200)
    else:
        return JSONResponse(content=health_status, status_code=200)

@app.get("/api/status/{job_id}")
async def get_screening_status(job_id: str):
    """Get character screening job status"""
    try:
        if not firestore_client:
            raise HTTPException(status_code=503, detail="Firestore not configured")
        
        doc_ref = firestore_client.collection("character_screening_jobs").document(job_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        job_data = doc.to_dict()
        
        # Add computed fields
        submitted_at = datetime.fromisoformat(job_data["submitted_at"].replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        processing_duration = (current_time - submitted_at).total_seconds()
        
        response = {
            "job_id": job_id,
            "status": job_data["status"],
            "entity_type": job_data["entity_type"],
            "screening_types": job_data["screening_types"],
            "submitted_at": job_data["submitted_at"],
            "updated_at": job_data["updated_at"],
            "processing_duration_seconds": round(processing_duration, 2)
        }
        
        # Add results if completed
        if job_data["status"] == "completed" and "results" in job_data:
            response["results"] = job_data["results"]
        
        # Add error info if failed
        if job_data["status"] == "failed" and "error" in job_data:
            response["error"] = job_data["error"]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.post("/api/screen")
async def submit_character_screening(
    request: CharacterScreeningRequest,
    auth_info: Dict[str, Any] = Depends(authenticate_request)
):
    """
    Submit character screening request
    Supports PEP analysis, negative news detection, and law involvement checking
    Returns job_id for async status tracking
    """
    start_time = time.time()
    
    print(f"📤 Received character screening request for {request.entity_type}: '{request.name[:50]}...' from client {auth_info['client_id']}")
    
    try:
        # Validate screening request and check model availability  
        validation_result = await validate_screening_request(request.screening_types, request.entity_type, request.name)
        
        if not validation_result["valid"]:
            character_screening_metrics.record_request_failure(
                start_time, 
                "validation_failed",
                request.entity_type.value,
                "unknown",
                "model_unavailable"
            )
            
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Screening models unavailable",
                    "message": "The requested screening services are currently unavailable",
                    "unavailable_screenings": validation_result["unavailable_screenings"],
                    "recommendations": validation_result["fallback_recommendations"]
                }
            )
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Extract available screening types
        available_screening_types = [s["screening_type"] for s in validation_result["available_screenings"]]
        
        # Log audit trail for sensitive name processing
        log_character_screening_audit(
            auth_info=auth_info,
            screening_type=",".join(available_screening_types),
            entity_type=request.entity_type.value,
            name=request.name,
            job_id=job_id
        )
        
        # Create job record in Firestore
        job_data = create_screening_job_record(
            job_id=job_id,
            screening_types=available_screening_types,
            entity_type=request.entity_type.value,
            name=request.name,
            additional_context=request.additional_context,
            jurisdiction=request.jurisdiction
        )
        
        # Publish message to Pub/Sub for worker processing
        message_id = publish_screening_message(
            job_id=job_id,
            screening_types=available_screening_types,
            entity_type=request.entity_type.value,
            name=request.name,
            additional_context=request.additional_context,
            jurisdiction=request.jurisdiction
        )
        
        processing_time = time.time() - start_time
        
        # Record successful request in metrics
        character_screening_metrics.record_request_success(
            start_time,
            ",".join(available_screening_types),
            request.entity_type.value,
            "multiple_models"
        )
        
        print(f"✅ Character screening job submitted: {job_id} in {processing_time:.2f}s")
        
        return {
            "success": True,
            "job_id": job_id,
            "status": JobStatus.SUBMITTED.value,
            "screening_types": available_screening_types,
            "entity_type": request.entity_type.value,
            "name": request.name,
            "available_screenings": validation_result["available_screenings"],
            "message_id": message_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "processing_time": round(processing_time, 2),
            "message": f"Character screening job submitted successfully. Use GET /api/status/{job_id} to check progress.",
            "rate_limit": {
                "remaining": auth_info["rate_limit"]["remaining"],
                "burst_remaining": auth_info["rate_limit"]["burst_remaining"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Character screening job submission failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Record metrics failure
        character_screening_metrics.record_request_failure(
            start_time,
            "unknown",
            request.entity_type.value,
            "unknown", 
            "internal_error"
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_msg,
                "job_id": job_id if 'job_id' in locals() else None,
                "processing_time": round(processing_time, 2)
            }
        )

@app.post("/api/screen/comprehensive")
async def submit_comprehensive_screening(
    name: str,
    entity_type: EntityType,
    additional_context: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    auth_info: Dict[str, Any] = Depends(authenticate_request)
):
    """
    Submit comprehensive character screening (all screening types)
    Convenience endpoint for complete screening
    """
    request = CharacterScreeningRequest(
        name=name,
        entity_type=entity_type,
        screening_types=[ScreeningType.COMPREHENSIVE],
        additional_context=additional_context,
        jurisdiction=jurisdiction
    )
    
    return await submit_character_screening(request, auth_info)

@app.get("/api/metrics")
async def get_metrics():
    """Get comprehensive character screening metrics"""
    return {
        "character_screening_metrics": character_screening_metrics.get_comprehensive_metrics(),
        "security_metrics": security_metrics.get_metrics(),
        "rate_limiting_stats": rate_limiter.get_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/models")
async def get_available_models():
    """Get information about available screening models"""
    model_info = {}
    
    for screening_type, entity_configs in SCREENING_MODEL_CONFIG.items():
        model_info[screening_type] = {}
        
        for entity_type, config in entity_configs.items():
            model_name = config["model"]
            is_available = await check_model_availability(model_name)
            
            model_info[screening_type][entity_type] = {
                "model": model_name,
                "description": config["description"],
                "risk_categories": config["risk_categories"],
                "available": is_available,
                "endpoint": f"{OPENWEBUI_BASE_URL}/api/chat/completions" if is_available else None
            }
    
    return {
        "screening_models": model_info,
        "endpoint_base_url": OPENWEBUI_BASE_URL,
        "last_checked": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
