"""
Firebase utilities for authentication and data access
"""

import logging
import os
from typing import List, Dict, Any, Optional
import firebase_admin
from firebase_admin import credentials, firestore, auth
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FirebaseManager:
    """Manager for Firebase operations"""
    
    def __init__(self):
        self.project_id = os.getenv("FIREBASE_PROJECT_ID")
        
        if not self.project_id:
            raise ValueError("FIREBASE_PROJECT_ID environment variable not set")
        
        self.app = None
        self.db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                self.app = firebase_admin.get_app()
            else:
                # Initialize Firebase Admin SDK using Application Default Credentials
                # This works in Google Cloud environments (Cloud Run, GKE, etc.)
                # and when GOOGLE_APPLICATION_CREDENTIALS is set to a service account file
                self.app = firebase_admin.initialize_app()
            
            # Initialize Firestore
            self.db = firestore.client()
            
            logger.info("Firebase Admin SDK initialized successfully using Application Default Credentials")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test Firebase connection"""
        try:
            # Test Firestore connection
            doc_ref = self.db.collection('_test').document('_test')
            doc_ref.set({'test': True})
            doc_ref.delete()
            
            logger.info("Firebase connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"Firebase connection test failed: {e}")
            return False
    
    async def get_user_document(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user document from Firestore
        
        Args:
            user_id: User ID
            
        Returns:
            User document data or None if not found
        """
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning(f"User document not found for user_id: {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"FirebaseManager: Error getting user document: {e}")
            return None
    
    async def get_project_document(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project document from Firestore
        
        Args:
            project_id: Project ID
            
        Returns:
            Project document data or None if not found
        """
        try:
            doc_ref = self.db.collection('projects').document(project_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning(f"Project document not found for project_id: {project_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting project document: {e}")
            return None
    
    async def get_workspace_document(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workspace document from Firestore
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            Workspace document data or None if not found
        """
        try:
            doc_ref = self.db.collection('workspaces').document(workspace_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning(f"Workspace document not found for workspace_id: {workspace_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting workspace document: {e}")
            return None
    
    async def update_project_processing_status(
        self,
        project_id: str,
        status: str,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update project processing status in Firestore
        
        Args:
            project_id: Project ID
            status: Processing status
            message: Status message
            metadata: Additional metadata
        """
        try:
            doc_ref = self.db.collection('rag_projects').document(project_id)
            
            update_data = {
                'graphProcessingStatus': status,
                'graphProcessingUpdatedAt': firestore.SERVER_TIMESTAMP
            }
            
            if message:
                update_data['graphProcessingMessage'] = message
            
            if metadata:
                update_data['graphProcessingMetadata'] = metadata
            
            doc_ref.set(update_data, merge=True)
            
            logger.info(f"Updated processing status for project {project_id}: {status}")
            
        except Exception as e:
            logger.error(f"Error updating processing status: {e}")
            raise
    
    async def create_rag_processing_job(
        self,
        job_id: str,
        project_id: str,
        user_id: str,
        workspace_id: str,
        job_type: str,
        gcs_path: str,
        config: Optional[Dict[str, Any]] = None,
        status: str = "pending"
    ) -> str:
        """
        Create a RAG processing job in Firestore
        
        Args:
            job_id: Unique job ID
            project_id: Project ID
            user_id: User ID
            workspace_id: Workspace ID
            job_type: Job type (document or folder)
            gcs_path: GCS path to process
            config: Processing configuration
            status: Initial job status
            
        Returns:
            Job ID
        """
        try:
            job_data = {
                'id': job_id,
                'projectId': project_id,
                'userId': user_id,
                'workspaceId': workspace_id,
                'jobType': job_type,
                'gcsPath': gcs_path,
                'status': status,
                'createdAt': firestore.SERVER_TIMESTAMP,
                'updatedAt': firestore.SERVER_TIMESTAMP,
                'progress': {
                    'totalDocuments': 0,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 0
                }
            }
            
            if config:
                job_data['config'] = config
            
            doc_ref = self.db.collection('rag_processing_jobs').document(job_id)
            doc_ref.set(job_data)
            
            logger.info(f"Created RAG processing job {job_id} for project {project_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating RAG processing job: {e}")
            raise
    
    async def update_rag_processing_job(
        self,
        job_id: str,
        updates: Dict[str, Any]
    ):
        """
        Update a RAG processing job in Firestore
        
        Args:
            job_id: Job ID
            updates: Dictionary of updates
        """
        try:
            doc_ref = self.db.collection('rag_processing_jobs').document(job_id)
            
            # Add timestamp to updates
            updates['updatedAt'] = firestore.SERVER_TIMESTAMP
            
            doc_ref.update(updates)
            
            logger.info(f"Updated RAG processing job {job_id}")
            
        except Exception as e:
            logger.error(f"Error updating RAG processing job: {e}")
            raise
    
    async def get_rag_processing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a RAG processing job from Firestore
        
        Args:
            job_id: Job ID
            
        Returns:
            Job document data or None if not found
        """
        try:
            doc_ref = self.db.collection('rag_processing_jobs').document(job_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                logger.warning(f"RAG processing job not found for job_id: {job_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting RAG processing job: {e}")
            return None
    
    async def get_user_rag_processing_jobs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get RAG processing jobs for a user
        
        Args:
            user_id: User ID
            limit: Maximum number of jobs to return
            
        Returns:
            List of job documents
        """
        try:
            jobs_ref = self.db.collection('rag_processing_jobs')
            query = jobs_ref.where('userId', '==', user_id).order_by('createdAt', direction=firestore.Query.DESCENDING).limit(limit)
            docs = query.stream()
            
            jobs = []
            for doc in docs:
                job_data = doc.to_dict()
                job_data['id'] = doc.id
                jobs.append(job_data)
            
            logger.info(f"Found {len(jobs)} RAG processing jobs for user {user_id}")
            return jobs
            
        except Exception as e:
            logger.error(f"Error getting user RAG processing jobs: {e}")
            return []
    
    async def get_project_rag_processing_jobs(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Get RAG processing jobs for a project
        
        Args:
            project_id: Project ID
            
        Returns:
            List of job documents
        """
        try:
            jobs_ref = self.db.collection('rag_processing_jobs')
            query = jobs_ref.where('projectId', '==', project_id).order_by('createdAt', direction=firestore.Query.DESCENDING)
            docs = query.stream()
            
            jobs = []
            for doc in docs:
                job_data = doc.to_dict()
                job_data['id'] = doc.id
                jobs.append(job_data)
            
            logger.info(f"Found {len(jobs)} RAG processing jobs for project {project_id}")
            return jobs
            
        except Exception as e:
            logger.error(f"Error getting project RAG processing jobs: {e}")
            return []
