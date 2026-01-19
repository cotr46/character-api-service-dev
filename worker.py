"""
Character Screening Worker Service
Handles text analysis processing via external APIs and custom search tools
Processes jobs from Pub/Sub queue and updates results in Firestore
"""

import os
import json
import time
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# Google Cloud imports
from google.cloud import pubsub_v1, firestore, storage
from google.api_core import exceptions as gcp_exceptions

# Local imports
from text_analysis_metrics import text_analysis_metrics, AnalysisType, EntityType
from security import InputSanitizer, SecurityViolationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkerConfig:
    """Worker service configuration"""
    
    # Google Cloud Configuration
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
    SUBSCRIPTION_NAME = os.getenv("PUBSUB_SUBSCRIPTION", "character-screening-worker-sub")
    FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "document-processing-firestore")
    
    # OpenWebUI API Configuration
    OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "https://nexus-bnimove-369455734154.asia-southeast2.run.app")
    OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "sk-c2ebcb8d36aa4361a28560915d8ab6f2")
    
    # Model Configuration
    TEXT_MODEL_TIMEOUT = int(os.getenv("TEXT_MODEL_TIMEOUT_SECONDS", "30"))
    TEXT_MODEL_RETRY_ATTEMPTS = int(os.getenv("TEXT_MODEL_RETRY_ATTEMPTS", "3"))
    
    # Custom Search API Configuration (untuk tool tambahan)
    CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY", "")
    CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID", "")
    
    # Worker Configuration
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "5"))
    JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))  # 5 minutes
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))  # 1 minute
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY_BASE = int(os.getenv("RETRY_DELAY_BASE", "5"))  # exponential backoff base

# Text analysis model configuration
TEXT_MODEL_CONFIG = {
    "pep-analysis": {
        "model": "politically-exposed-person-v2",
        "entity_types": ["person"],
        "description": "Political Exposure Person Analysis v2",
        "fallback": "negative-news"
    },
    "negative-news": {
        "model": "negative-news", 
        "entity_types": ["person"],
        "description": "Individual Negative News Analysis",
        "fallback": None
    },
    "law-involvement": {
        "model": "law-involvement",
        "entity_types": ["person"], 
        "description": "Individual Law Involvement Analysis",
        "fallback": "negative-news"
    },
    "corporate-negative-news": {
        "model": "negative-news-corporate",
        "entity_types": ["corporate"],
        "description": "Corporate Negative News Analysis",
        "fallback": "corporate-law-involvement"
    },
    "corporate-law-involvement": {
        "model": "law-involvement-corporate",
        "entity_types": ["corporate"],
        "description": "Corporate Law Involvement Analysis",
        "fallback": None
    }
}

class CharacterScreeningWorker:
    """
    Main worker class for character screening processing
    """
    
    def __init__(self):
        self.project_id = WorkerConfig.PROJECT_ID
        self.subscription_path = f"projects/{self.project_id}/subscriptions/{WorkerConfig.SUBSCRIPTION_NAME}"
        
        # Initialize Google Cloud clients
        self.subscriber = pubsub_v1.SubscriberClient()
        self.firestore_client = firestore.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        
        # HTTP session for API calls
        self.http_session = None
        
        # Worker state
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=WorkerConfig.MAX_CONCURRENT_JOBS)
        
        # Health monitoring
        self.last_health_check = time.time()
        self.processed_jobs = 0
        self.failed_jobs = 0
        
        logger.info(f"🔧 Character Screening Worker initialized for project: {self.project_id}")
    
    async def start(self):
        """Start the worker service"""
        self.is_running = True
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=WorkerConfig.TEXT_MODEL_TIMEOUT + 10)
        self.http_session = aiohttp.ClientSession(timeout=timeout)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Character Screening Worker started")
        
        # Start health check task
        health_task = asyncio.create_task(self._health_check_loop())
        
        try:
            # Start processing messages
            await self._process_messages()
        finally:
            # Cleanup
            health_task.cancel()
            await self._shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📢 Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    async def _process_messages(self):
        """Process messages from Pub/Sub subscription"""
        def callback(message):
            """Pub/Sub message callback"""
            try:
                # Run async processing in thread pool
                future = self.executor.submit(
                    asyncio.run, 
                    self._process_message(message)
                )
                future.result(timeout=WorkerConfig.JOB_TIMEOUT)
            except Exception as e:
                logger.error(f"❌ Message processing failed: {str(e)}")
                message.nack()
        
        # Configure flow control
        flow_control = pubsub_v1.types.FlowControl(max_messages=WorkerConfig.MAX_CONCURRENT_JOBS)
        
        logger.info(f"👂 Listening for messages on {self.subscription_path}")
        
        # Start pulling messages
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path, 
            callback=callback,
            flow_control=flow_control
        )
        
        try:
            while self.is_running:
                try:
                    # Use a timeout to allow checking is_running periodically
                    streaming_pull_future.result(timeout=5)
                except Exception as timeout_ex:
                    # Continue if it's just a timeout
                    if "timeout" in str(timeout_ex).lower():
                        continue
                    raise
        finally:
            streaming_pull_future.cancel()
            logger.info("📬 Message subscription cancelled")
    
    async def _process_message(self, message):
        """Process individual Pub/Sub message"""
        start_time = time.time()
        job_id = None
        
        try:
            # Parse message data
            message_data = json.loads(message.data.decode('utf-8'))
            job_id = message_data.get('job_id')
            analysis_type = message_data.get('analysis_type')
            entity_type = message_data.get('entity_type')
            name = message_data.get('name')
            additional_context = message_data.get('additional_context')
            
            logger.info(f"🔄 Processing job {job_id}: {analysis_type} for {entity_type} - '{name[:50]}...'")
            
            # Validate message data
            if not all([job_id, analysis_type, entity_type, name]):
                raise ValueError("Missing required fields in message")
            
            # Get model configuration
            model_config = TEXT_MODEL_CONFIG.get(analysis_type)
            if not model_config:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Record metrics start
            metrics_start_time = text_analysis_metrics.record_request_start(
                analysis_type, entity_type, model_config["model"]
            )
            
            # Update job status to processing
            await self._update_job_status(
                job_id, "processing", 
                {"started_at": datetime.now(timezone.utc).isoformat()}
            )
            
            # Perform character screening
            result = await self._perform_character_screening(
                analysis_type, entity_type, name, additional_context, model_config
            )
            
            # Update job with results
            await self._update_job_status(
                job_id, "completed", 
                {
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                    "processing_time": time.time() - start_time
                }
            )
            
            # Record success metrics
            text_analysis_metrics.record_request_success(
                metrics_start_time, analysis_type, entity_type, 
                model_config["model"], result.get("model_response_time")
            )
            
            self.processed_jobs += 1
            logger.info(f"✅ Job {job_id} completed successfully in {time.time() - start_time:.2f}s")
            
            # Acknowledge message
            message.ack()
            
        except Exception as e:
            error_msg = f"Job processing failed: {str(e)}"
            logger.error(f"❌ Job {job_id} failed: {error_msg}")
            
            try:
                # Record failure metrics
                if job_id and analysis_type and entity_type:
                    model_name = TEXT_MODEL_CONFIG.get(analysis_type, {}).get("model", "unknown")
                    text_analysis_metrics.record_request_failure(
                        start_time, analysis_type, entity_type, model_name, str(e)
                    )
                
                # Update job status to failed
                if job_id:
                    await self._update_job_status(
                        job_id, "failed",
                        {
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "error": error_msg,
                            "processing_time": time.time() - start_time
                        }
                    )
                
                self.failed_jobs += 1
                
            except Exception as update_error:
                logger.error(f"❌ Failed to update job status: {str(update_error)}")
            
            # Nack message for retry
            message.nack()
    
    async def _perform_character_screening(
        self, 
        analysis_type: str, 
        entity_type: str, 
        name: str, 
        additional_context: Optional[str],
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform character screening using external APIs and custom search tools
        """
        model_start_time = time.time()
        
        try:
            # Primary analysis using OpenWebUI API
            primary_result = await self._call_openwebui_api(
                model_config["model"], name, entity_type, additional_context
            )
            
            # Enhanced screening using custom search API if available
            custom_search_result = None
            if WorkerConfig.CUSTOM_SEARCH_API_KEY:
                custom_search_result = await self._perform_custom_search(
                    name, entity_type, analysis_type
                )
            
            # Combine results
            combined_result = self._combine_screening_results(
                primary_result, custom_search_result, analysis_type
            )
            
            model_response_time = time.time() - model_start_time
            
            return {
                "analysis_type": analysis_type,
                "entity_type": entity_type,
                "name": name,
                "result": combined_result,
                "model_used": model_config["model"],
                "model_response_time": model_response_time,
                "custom_search_performed": custom_search_result is not None,
                "confidence_score": combined_result.get("confidence_score", 0.0),
                "risk_level": self._calculate_risk_level(combined_result),
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            # Try fallback model if available
            fallback_type = model_config.get("fallback")
            if fallback_type and fallback_type in TEXT_MODEL_CONFIG:
                logger.warning(f"⚠️ Primary model failed, trying fallback: {fallback_type}")
                
                text_analysis_metrics.record_fallback_usage(analysis_type, fallback_type)
                
                fallback_config = TEXT_MODEL_CONFIG[fallback_type]
                return await self._perform_character_screening(
                    fallback_type, entity_type, name, additional_context, fallback_config
                )
            else:
                raise e
    
    async def _call_openwebui_api(
        self, 
        model: str, 
        name: str, 
        entity_type: str, 
        additional_context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Call OpenWebUI API for text analysis
        """
        url = f"{WorkerConfig.OPENWEBUI_BASE_URL}/ollama/api/generate"
        
        # Prepare prompt based on analysis type
        prompt = self._build_analysis_prompt(model, name, entity_type, additional_context)
        
        headers = {
            "Authorization": f"Bearer {WorkerConfig.OPENWEBUI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        for attempt in range(WorkerConfig.TEXT_MODEL_RETRY_ATTEMPTS):
            try:
                async with self.http_session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_model_response(result.get("response", ""))
                    else:
                        error_text = await response.text()
                        raise aiohttp.ClientError(f"API error {response.status}: {error_text}")
                        
            except Exception as e:
                if attempt < WorkerConfig.TEXT_MODEL_RETRY_ATTEMPTS - 1:
                    wait_time = WorkerConfig.RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"⚠️ API call attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    
    def _build_analysis_prompt(
        self, 
        model: str, 
        name: str, 
        entity_type: str, 
        additional_context: Optional[str]
    ) -> str:
        """
        Build analysis prompt based on model and entity type
        """
        context_part = f" Additional context: {additional_context}" if additional_context else ""
        
        if "pep" in model.lower() or "political" in model.lower():
            return f"""Analyze if "{name}" is a Politically Exposed Person (PEP).
Entity type: {entity_type}
{context_part}

Provide analysis in JSON format with:
- is_pep: boolean
- confidence_score: float (0-1)
- risk_level: "low"|"medium"|"high"
- evidence: list of findings
- sources: list of information sources"""
        
        elif "negative" in model.lower():
            return f"""Analyze "{name}" for negative news and reputation issues.
Entity type: {entity_type}
{context_part}

Provide analysis in JSON format with:
- has_negative_news: boolean
- confidence_score: float (0-1)
- risk_level: "low"|"medium"|"high"
- findings: list of negative news items
- sources: list of information sources"""
        
        elif "law" in model.lower():
            return f"""Analyze "{name}" for legal involvement and law enforcement issues.
Entity type: {entity_type}
{context_part}

Provide analysis in JSON format with:
- has_legal_issues: boolean
- confidence_score: float (0-1)
- risk_level: "low"|"medium"|"high"
- legal_findings: list of legal issues
- sources: list of information sources"""
        
        else:
            return f"""Perform comprehensive character screening analysis for "{name}".
Entity type: {entity_type}
{context_part}

Provide analysis in JSON format with:
- has_issues: boolean
- confidence_score: float (0-1)
- risk_level: "low"|"medium"|"high"
- findings: list of issues found
- sources: list of information sources"""
    
    def _parse_model_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate model response
        """
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return {
                    "raw_response": response,
                    "confidence_score": 0.5,
                    "risk_level": "medium",
                    "findings": ["Analysis completed but format unclear"],
                    "sources": []
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse model response: {str(e)}")
            return {
                "raw_response": response,
                "parse_error": str(e),
                "confidence_score": 0.0,
                "risk_level": "unknown",
                "findings": [],
                "sources": []
            }
    
    async def _perform_custom_search(
        self, 
        name: str, 
        entity_type: str, 
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Perform custom search using Google Custom Search API or other search tools
        """
        if not all([WorkerConfig.CUSTOM_SEARCH_API_KEY, WorkerConfig.CUSTOM_SEARCH_ENGINE_ID]):
            logger.warning("⚠️ Custom search API not configured")
            return None
        
        try:
            # Build search query
            search_query = self._build_search_query(name, entity_type, analysis_type)
            
            # Call Google Custom Search API
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": WorkerConfig.CUSTOM_SEARCH_API_KEY,
                "cx": WorkerConfig.CUSTOM_SEARCH_ENGINE_ID,
                "q": search_query,
                "num": 10
            }
            
            async with self.http_session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._process_search_results(result, analysis_type)
                else:
                    logger.warning(f"⚠️ Custom search failed with status {response.status}")
                    return None
                    
        except Exception as e:
            logger.warning(f"⚠️ Custom search error: {str(e)}")
            return None
    
    def _build_search_query(self, name: str, entity_type: str, analysis_type: str) -> str:
        """
        Build search query based on analysis type
        """
        base_query = f'"{name}"'
        
        if analysis_type == "pep-analysis":
            return f'{base_query} "politically exposed person" OR "PEP" OR "government official"'
        elif "negative" in analysis_type:
            return f'{base_query} "scandal" OR "controversy" OR "negative news" OR "investigation"'
        elif "law" in analysis_type:
            return f'{base_query} "lawsuit" OR "legal case" OR "court" OR "prosecution"'
        else:
            return f'{base_query} "background check" OR "due diligence"'
    
    def _process_search_results(self, search_results: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Process and analyze custom search results
        """
        items = search_results.get("items", [])
        
        findings = []
        sources = []
        risk_indicators = 0
        
        for item in items[:5]:  # Analyze top 5 results
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            url = item.get("link", "")
            
            # Simple keyword-based risk assessment
            risk_keywords = [
                "scandal", "controversy", "lawsuit", "investigation", 
                "arrest", "fraud", "corruption", "sanctions"
            ]
            
            text = f"{title} {snippet}".lower()
            found_risks = [kw for kw in risk_keywords if kw in text]
            
            if found_risks:
                risk_indicators += len(found_risks)
                findings.append({
                    "title": title,
                    "snippet": snippet,
                    "risk_keywords": found_risks,
                    "relevance": "high" if len(found_risks) > 2 else "medium"
                })
            
            sources.append({
                "title": title,
                "url": url,
                "snippet": snippet[:200]
            })
        
        return {
            "search_performed": True,
            "total_results": len(items),
            "risk_indicators": risk_indicators,
            "findings": findings,
            "sources": sources,
            "confidence_score": min(0.8, risk_indicators * 0.1)
        }
    
    def _combine_screening_results(
        self, 
        primary_result: Dict[str, Any], 
        custom_search_result: Optional[Dict[str, Any]], 
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        Combine primary model results with custom search results
        """
        combined = primary_result.copy()
        
        if custom_search_result:
            # Enhance confidence score
            primary_confidence = primary_result.get("confidence_score", 0.0)
            search_confidence = custom_search_result.get("confidence_score", 0.0)
            
            # Weighted average (70% primary, 30% search)
            combined["confidence_score"] = (primary_confidence * 0.7) + (search_confidence * 0.3)
            
            # Add search findings
            combined["custom_search"] = custom_search_result
            
            # Update risk level based on combined evidence
            combined["risk_level"] = self._calculate_risk_level(combined)
        
        return combined
    
    def _calculate_risk_level(self, result: Dict[str, Any]) -> str:
        """
        Calculate overall risk level based on analysis results
        """
        confidence = result.get("confidence_score", 0.0)
        
        # Check for high-risk indicators
        high_risk_indicators = [
            result.get("is_pep", False),
            result.get("has_negative_news", False),
            result.get("has_legal_issues", False),
            result.get("has_issues", False)
        ]
        
        custom_search = result.get("custom_search", {})
        search_risk = custom_search.get("risk_indicators", 0)
        
        # Calculate risk score
        risk_score = 0
        if any(high_risk_indicators):
            risk_score += 0.5
        
        risk_score += confidence * 0.3
        risk_score += min(0.2, search_risk * 0.05)
        
        if risk_score >= 0.7:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    async def _update_job_status(self, job_id: str, status: str, additional_data: Dict[str, Any]):
        """
        Update job status in Firestore
        """
        try:
            doc_ref = self.firestore_client.collection("text_analysis_jobs").document(job_id)
            
            update_data = {
                "status": status,
                "updated_at": firestore.SERVER_TIMESTAMP,
                **additional_data
            }
            
            doc_ref.update(update_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to update job status for {job_id}: {str(e)}")
            raise
    
    async def _health_check_loop(self):
        """
        Periodic health check
        """
        while self.is_running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(WorkerConfig.HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"❌ Health check failed: {str(e)}")
    
    async def _perform_health_check(self):
        """
        Perform health check
        """
        current_time = time.time()
        uptime = current_time - self.last_health_check
        
        # Test Firestore connectivity
        try:
            self.firestore_client.collection("health_check").document("worker").set({
                "last_check": datetime.now(timezone.utc),
                "worker_id": f"worker-{os.getpid()}",
                "processed_jobs": self.processed_jobs,
                "failed_jobs": self.failed_jobs,
                "uptime": uptime
            })
        except Exception as e:
            logger.error(f"❌ Firestore health check failed: {str(e)}")
        
        # Test OpenWebUI API connectivity
        try:
            url = f"{WorkerConfig.OPENWEBUI_BASE_URL}/health"
            async with self.http_session.get(url, timeout=5) as response:
                if response.status != 200:
                    logger.warning(f"⚠️ OpenWebUI health check returned {response.status}")
        except Exception as e:
            logger.warning(f"⚠️ OpenWebUI health check failed: {str(e)}")
        
        self.last_health_check = current_time
        logger.debug(f"✅ Health check completed. Processed: {self.processed_jobs}, Failed: {self.failed_jobs}")
    
    async def _shutdown(self):
        """
        Graceful shutdown
        """
        logger.info("🛑 Shutting down worker service...")
        
        if self.http_session:
            await self.http_session.close()
        
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Worker service shutdown complete")

# Health check endpoint for Cloud Run
from fastapi import FastAPI

app = FastAPI(title="Character Screening Worker Service")

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "healthy",
        "service": "character-screening-worker",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get worker metrics"""
    return text_analysis_metrics.get_comprehensive_metrics()

# Main worker execution
async def main():
    """
    Main worker function
    """
    worker = CharacterScreeningWorker()
    await worker.start()

if __name__ == "__main__":
    # Check if running as FastAPI server (for health checks) or as worker
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        import uvicorn
        port = int(os.getenv("PORT", 8080))
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Run as worker
        asyncio.run(main())
