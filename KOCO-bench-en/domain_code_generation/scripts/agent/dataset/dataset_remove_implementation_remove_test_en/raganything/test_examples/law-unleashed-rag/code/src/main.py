"""
RAG-Anything Service - Main FastAPI application
"""

import os
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from src.services.rag_factory import RAGFactory
from src.services.rag_interface import RAGInterface
from src.services.auth_service import AuthService
from src.models.api_models import (
    ProcessDocumentRequest,
    ProcessFolderRequest,
    ProcessingJobResponse,
    ProcessingStatusResponse,
    HealthResponse,
    ErrorResponse,
    RAGApproachesResponse,
    RAGApproachInfo,
    QueryRequest,
    QueryResponse
)
from src.utils.firebase_utils import FirebaseManager
from src.utils.gcs_utils import GCSManager

# Configure standard Python logging for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Prometheus metrics - register only once
try:
    REQUEST_COUNT = Counter('rag_anything_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
    REQUEST_DURATION = Histogram('rag_anything_http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
    DOCUMENT_PROCESSING_DURATION = Histogram('rag_anything_document_processing_duration_seconds', 'Document processing duration')
except ValueError:
    # Metrics already registered (reload scenario)
    from prometheus_client import REGISTRY
    REQUEST_COUNT = REGISTRY._names_to_collectors['rag_anything_http_requests_total']
    REQUEST_DURATION = REGISTRY._names_to_collectors['rag_anything_http_request_duration_seconds']
    DOCUMENT_PROCESSING_DURATION = REGISTRY._names_to_collectors['rag_anything_document_processing_duration_seconds']

# Global service instances
firebase_manager: FirebaseManager | None = None
gcs_manager: GCSManager | None = None
auth_service: AuthService | None = None
rag_factory: RAGFactory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global firebase_manager, gcs_manager, auth_service, rag_factory
    
    logger.info("Starting RAG Test Bed service")
    
    try:
        # Initialize services
        firebase_manager = FirebaseManager()
        gcs_manager = GCSManager()
        auth_service = AuthService(firebase_manager)
        rag_factory = RAGFactory()
        
        # Test connections
        await firebase_manager.test_connection()
        await gcs_manager.test_connection()
        
        logger.info("RAG Test Bed service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start RAG Test Bed service: {e}")
        raise
    finally:
        logger.info("Shutting down RAG Test Bed service")


# Create FastAPI app
app = FastAPI(
    title="RAG Test Bed Service",
    description="Document processing service with multiple RAG approaches for law-unleashed",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get auth service
def get_auth_service() -> AuthService:
    if auth_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    return auth_service


# Dependency to get RAG service
def get_rag_service(approach: str = "raganything") -> RAGInterface:
    if rag_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG factory not available"
        )
    try:
        return rag_factory.create_rag_service(approach, firebase_manager, gcs_manager, auth_service)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    return HealthResponse(status="healthy", service="rag-test-bed")


@app.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check():
    """Detailed health check with service status"""
    health_status = {
        "status": "healthy",
        "service": "rag-test-bed",
        "services": {}
    }
    
    try:
        # Check Firebase connection
        await firebase_manager.test_connection()
        health_status["services"]["firebase"] = "healthy"
    except Exception as e:
        health_status["services"]["firebase"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    try:
        # Check GCS connection
        await gcs_manager.test_connection()
        health_status["services"]["gcs"] = "healthy"
    except Exception as e:
        health_status["services"]["gcs"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/rag-approaches", response_model=RAGApproachesResponse)
async def get_rag_approaches():
    """Get available RAG approaches and their capabilities"""
    if rag_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG factory not available"
        )
    
    approaches = {}
    for approach_name in rag_factory.get_available_approaches():
        try:
            approach_info = rag_factory.get_approach_info(approach_name)
            approaches[approach_name] = RAGApproachInfo(
                name=approach_info["name"],
                supported_parsers=approach_info["supported_parsers"],
                supported_models=approach_info["supported_models"],
                description=f"RAG approach using {approach_info['name']}"
            )
        except Exception as e:
            logger.warning(f"Could not get info for approach {approach_name}: {e}")
            approaches[approach_name] = RAGApproachInfo(
                name=approach_name,
                supported_parsers=[],
                supported_models=[],
                description=f"RAG approach: {approach_name}"
            )
    
    return RAGApproachesResponse(
        approaches=approaches,
        default_approach="raganything"
    )


@app.post("/process-document", response_model=ProcessingJobResponse)
async def process_document(
    request: ProcessDocumentRequest,
    background_tasks: BackgroundTasks,
    auth: AuthService = Depends(get_auth_service)
):
    """Process a single document with selected RAG approach"""
    logger.info(f"🎯 API endpoint /process-document called")
    logger.info(f"📋 Request data: user_id={request.user_id}, project_id={request.project_id}, workspace_id={request.workspace_id}")
    logger.info(f"📄 GCS path: {request.gcs_path}")
    logger.info(f"🔧 RAG approach: {request.rag_approach}")
    
    REQUEST_COUNT.labels(method="POST", endpoint="/process-document", status="started").inc()
    
    with REQUEST_DURATION.labels(method="POST", endpoint="/process-document").time():
        try:
            # Verify user access to workspace
            has_access = await auth.verify_workspace_access(
                request.user_id, 
                request.workspace_id
            )
            
            if not has_access:
                REQUEST_COUNT.labels(method="POST", endpoint="/process-document", status="403").inc()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not have access to this workspace"
                )
            
            # Get RAG service for the specified approach
            rag = get_rag_service(request.rag_approach)
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Create processing job in Firestore
            await firebase_manager.create_rag_processing_job(
                job_id=job_id,
                project_id=request.project_id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                job_type="document",
                gcs_path=request.gcs_path,
                config={
                    'rag_approach': request.rag_approach,
                    'parser': request.parser,
                    'parse_method': request.parse_method,
                    'model': request.model,
                    'config': request.config
                }
            )
            
            # Start background processing
            background_tasks.add_task(
                rag.process_document,
                job_id=job_id,
                user_id=request.user_id,
                project_id=request.project_id,
                workspace_id=request.workspace_id,
                gcs_path=request.gcs_path,
                parser=request.parser,
                parse_method=request.parse_method,
                model=request.model,
                config=request.config
            )
            
            REQUEST_COUNT.labels(method="POST", endpoint="/process-document", status="200").inc()
            return ProcessingJobResponse(
                job_id=job_id,
                status="pending",
                message=f"Document processing job created successfully with {request.rag_approach}",
                created_at=datetime.utcnow()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating document processing job: {e}")
            REQUEST_COUNT.labels(method="POST", endpoint="/process-document", status="500").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating document processing job: {str(e)}"
            )


@app.post("/process-folder", response_model=ProcessingJobResponse)
async def process_folder(
    request: ProcessFolderRequest,
    background_tasks: BackgroundTasks,
    auth: AuthService = Depends(get_auth_service)
):
    """Process all documents in a folder with selected RAG approach"""
    logger.info(f"🎯 API endpoint /process-folder called")
    logger.info(f"📋 Request data: user_id={request.user_id}, project_id={request.project_id}, workspace_id={request.workspace_id}")
    logger.info(f"📁 GCS folder path: {request.gcs_folder_path}")
    logger.info(f"🔧 RAG approach: {request.rag_approach}")
    
    REQUEST_COUNT.labels(method="POST", endpoint="/process-folder", status="started").inc()
    
    with REQUEST_DURATION.labels(method="POST", endpoint="/process-folder").time():
        try:
            # Verify user access to workspace
            has_access = await auth.verify_workspace_access(
                request.user_id, 
                request.workspace_id
            )
            
            if not has_access:
                REQUEST_COUNT.labels(method="POST", endpoint="/process-folder", status="403").inc()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not have access to this workspace"
                )
            
            # Get RAG service for the specified approach
            rag = get_rag_service(request.rag_approach)
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Create processing job in Firestore
            await firebase_manager.create_rag_processing_job(
                job_id=job_id,
                project_id=request.project_id,
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                job_type="folder",
                gcs_path=request.gcs_folder_path,
                config={
                    'rag_approach': request.rag_approach,
                    'parser': request.parser,
                    'parse_method': request.parse_method,
                    'model': request.model,
                    'config': request.config,
                    'file_extensions': request.file_extensions
                }
            )
            
            # Start background processing
            background_tasks.add_task(
                rag.process_folder,
                job_id=job_id,
                user_id=request.user_id,
                project_id=request.project_id,
                workspace_id=request.workspace_id,
                gcs_folder_path=request.gcs_folder_path,
                file_extensions=request.file_extensions,
                parser=request.parser,
                parse_method=request.parse_method,
                model=request.model,
                config=request.config
            )
            
            REQUEST_COUNT.labels(method="POST", endpoint="/process-folder", status="200").inc()
            return ProcessingJobResponse(
                job_id=job_id,
                status="pending",
                message=f"Folder processing job created successfully with {request.rag_approach}",
                created_at=datetime.utcnow()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating folder processing job: {e}")
            REQUEST_COUNT.labels(method="POST", endpoint="/process-folder", status="500").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating folder processing job: {str(e)}"
            )


@app.get("/processing-status/{job_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(
    job_id: str,
    user_id: str,
    auth: AuthService = Depends(get_auth_service)
):
    """Get processing status for a job"""
    REQUEST_COUNT.labels(method="GET", endpoint="/processing-status", status="started").inc()
    
    with REQUEST_DURATION.labels(method="GET", endpoint="/processing-status").time():
        try:
            # Get job information from Firebase
            job_info = await firebase_manager.get_rag_processing_job(job_id)
            
            if not job_info:
                REQUEST_COUNT.labels(method="GET", endpoint="/processing-status", status="404").inc()
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Processing job not found"
                )
            
            # Verify user access to the job
            if job_info.get('userId') != user_id:
                REQUEST_COUNT.labels(method="GET", endpoint="/processing-status", status="403").inc()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not have access to this job"
                )
            
            # Extract corpus information from result if available
            corpus_info = None
            result = job_info.get('result')
            if result and 'storage_info' in result:
                storage_info = result['storage_info']
                if 'corpus_name' in storage_info:
                    corpus_info = {
                        'corpus_name': storage_info['corpus_name'],
                        'gcs_bucket': storage_info.get('gcs_bucket'),
                        'gcs_folder_path': storage_info.get('gcs_folder_path')
                    }
            
            REQUEST_COUNT.labels(method="GET", endpoint="/processing-status", status="200").inc()
            return ProcessingStatusResponse(
                job_id=job_id,
                status=job_info.get('status', 'unknown'),
                message=job_info.get('graphProcessingMessage'),
                progress=job_info.get('progress', {}),
                created_at=job_info.get('createdAt'),
                updated_at=job_info.get('updatedAt'),
                error_message=job_info.get('errorMessage'),
                result=result,
                corpus_info=corpus_info
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting processing status for job {job_id}: {e}")
            REQUEST_COUNT.labels(method="GET", endpoint="/processing-status", status="500").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting processing status: {str(e)}"
            )


@app.get("/user-jobs/{user_id}")
async def get_user_jobs(
    user_id: str,
    limit: int = 50,
    auth: AuthService = Depends(get_auth_service)
):
    """Get processing jobs for a user"""
    REQUEST_COUNT.labels(method="GET", endpoint="/user-jobs", status="started").inc()
    
    with REQUEST_DURATION.labels(method="GET", endpoint="/user-jobs").time():
        try:
            # Verify user exists
            user_exists = await auth.verify_user_exists(user_id)
            if not user_exists:
                REQUEST_COUNT.labels(method="GET", endpoint="/user-jobs", status="404").inc()
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            # Get user jobs
            jobs = await firebase_manager.get_user_rag_processing_jobs(user_id, limit)
            
            REQUEST_COUNT.labels(method="GET", endpoint="/user-jobs", status="200").inc()
            return {
                "user_id": user_id,
                "jobs": jobs,
                "total": len(jobs)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting user jobs for user {user_id}: {e}")
            REQUEST_COUNT.labels(method="GET", endpoint="/user-jobs", status="500").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting user jobs: {str(e)}"
            )


@app.get("/project-jobs/{project_id}")
async def get_project_jobs(
    project_id: str,
    user_id: str,
    auth: AuthService = Depends(get_auth_service)
):
    """Get processing jobs for a project"""
    REQUEST_COUNT.labels(method="GET", endpoint="/project-jobs", status="started").inc()
    
    with REQUEST_DURATION.labels(method="GET", endpoint="/project-jobs").time():
        try:
            # Verify user access to project
            has_access = await auth.verify_project_access(user_id, project_id)
            if not has_access:
                REQUEST_COUNT.labels(method="GET", endpoint="/project-jobs", status="403").inc()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not have access to this project"
                )
            
            # Get project jobs
            jobs = await firebase_manager.get_project_rag_processing_jobs(project_id)
            
            REQUEST_COUNT.labels(method="GET", endpoint="/project-jobs", status="200").inc()
            return {
                "project_id": project_id,
                "jobs": jobs,
                "total": len(jobs)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting project jobs for project {project_id}: {e}")
            REQUEST_COUNT.labels(method="GET", endpoint="/project-jobs", status="500").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting project jobs: {str(e)}"
            )


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    auth: AuthService = Depends(get_auth_service)
):
    """Query processed documents using a specific RAG approach"""
    import time
    
    logger.info(f"🎯 API endpoint /query called")
    logger.info(f"📋 Request data: user_id={request.user_id}, project_id={request.project_id}")
    logger.info(f"🔧 RAG approach: {request.rag_approach}")
    logger.info(f"❓ Query: {request.query[:100]}...")
    
    REQUEST_COUNT.labels(method="POST", endpoint="/query", status="started").inc()
    start_time = time.time()
    
    with REQUEST_DURATION.labels(method="POST", endpoint="/query").time():
        try:
            # Verify user access to project
            has_access = await auth.verify_project_access(request.user_id, request.project_id)
            if not has_access:
                REQUEST_COUNT.labels(method="POST", endpoint="/query", status="403").inc()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User does not have access to this project"
                )
            
            # Get RAG service for the specified approach
            rag = get_rag_service(request.rag_approach)
            
            # Query the documents
            result = await rag.query_documents(
                user_id=request.user_id,
                project_id=request.project_id,
                query=request.query,
                model=request.model,
                config=request.config,
                corpus_info=request.corpus_info
            )
            
            processing_time = time.time() - start_time
            
            REQUEST_COUNT.labels(method="POST", endpoint="/query", status="200").inc()
            return QueryResponse(
                query=request.query,
                answer=result.get("answer", ""),
                rag_approach=request.rag_approach,
                model=request.model,
                sources=result.get("sources", []),
                metadata=result.get("metadata", {}),
                processing_time=processing_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            REQUEST_COUNT.labels(method="POST", endpoint="/query", status="500").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing query: {str(e)}"
            )


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("ENVIRONMENT") == "development"
    )
