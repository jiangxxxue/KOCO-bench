"""
RAG-Anything service for document and folder processing
"""

import logging
import os
import tempfile
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag import LightRAG

from src.utils.firebase_utils import FirebaseManager
from src.utils.gcs_utils import GCSManager
from src.services.auth_service import AuthService
from src.services.rag_interface import RAGInterface
from src.models.firebase_models import RAGProcessingMetadata

logger = logging.getLogger(__name__)


class RAGAnythingService(RAGInterface):
    """Service for processing documents with RAG-Anything"""
    
    def __init__(self, firebase_manager: FirebaseManager, gcs_manager: GCSManager, auth_service: AuthService):
        super().__init__(firebase_manager, gcs_manager, auth_service)
        
        # Storage configuration - use local directories for development
        self.storage_dir = os.getenv("RAG_STORAGE_DIR", "./storage")
        self.temp_dir = os.getenv("RAG_TEMP_DIR", "./temp")
        
        # Ensure directories exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize RAG-Anything with storage backends
        self._initialize_rag_storage()
    
    ##key
    def _initialize_rag_storage(self):
        """Initialize RAG-Anything following the official example"""
        try:
            # Log environment variables for debugging
            logger.info("Environment variables:")
            logger.info(f"  OPENAI_API_KEY: {'***' + os.getenv('OPENAI_API_KEY', 'NOT_SET')[-4:] if os.getenv('OPENAI_API_KEY') else 'NOT_SET'}")
            logger.info(f"  OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'NOT_SET')}")
            logger.info(f"  MODEL: {os.getenv('MODEL', 'NOT_SET')}")
            logger.info(f"  PARSER: {os.getenv('PARSER', 'NOT_SET')}")
            logger.info(f"  PARSE_METHOD: {os.getenv('PARSE_METHOD', 'NOT_SET')}")
            logger.info(f"  RAG_STORAGE_DIR: {os.getenv('RAG_STORAGE_DIR', 'NOT_SET')}")
            logger.info(f"  RAG_TEMP_DIR: {os.getenv('RAG_TEMP_DIR', 'NOT_SET')}")
            
            # Create RAGAnything configuration
            config = RAGAnythingConfig(
                working_dir=self.storage_dir,
                parser=os.getenv("PARSER", "mineru"),
                parse_method=os.getenv("PARSE_METHOD", "auto"),
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )
            
            # Define LLM model function
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    os.getenv("MODEL", "gpt-4o-mini"),
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    **kwargs,
                )
            
            # Define vision model function
            def vision_model_func(
                prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
            ):
                if messages:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=messages,
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL"),
                        **kwargs,
                    )
                elif image_data:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=[
                            {"role": "system", "content": system_prompt} if system_prompt else None,
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt},
                        ],
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL"),
                        **kwargs,
                    )
                else:
                    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
            
            # Define embedding function
            embedding_func = EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                ),
            )
            
            # Initialize RAGAnything with LightRAG integration
            self.rag_anything = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )
            
            # Store storage paths for result reporting
            self.kv_storage_path = os.path.join(self.storage_dir, "kv")
            self.vector_storage_path = os.path.join(self.storage_dir, "vector")
            self.graph_storage_path = os.path.join(self.storage_dir, "graph")
            self.doc_status_storage_path = os.path.join(self.storage_dir, "status")
            
            logger.info("RAG-Anything initialized with LightRAG")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG-Anything: {e}")
            raise
    
    ##key
    def _create_rag_instance(self):
        """Create a fresh RAGAnything instance for querying"""
        try:
            # Define LLM model function
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    os.getenv("MODEL", "gpt-4o-mini"),
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    **kwargs,
                )
            
            # Define vision model function
            def vision_model_func(
                prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
            ):
                if messages:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=messages,
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL"),
                        **kwargs,
                    )
                elif image_data:
                    return openai_complete_if_cache(
                        "gpt-4o",
                        "",
                        system_prompt=None,
                        history_messages=[],
                        messages=[
                            {"role": "system", "content": system_prompt} if system_prompt else None,
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt},
                        ],
                        api_key=os.getenv("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL"),
                        **kwargs,
                    )
                else:
                    return llm_model_func(prompt, system_prompt, history_messages, **kwargs)
            
            # Define embedding function
            embedding_func = EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                ),
            )
            
            # Try to manually initialize LightRAG with existing data
            try:
                logger.info("Attempting to initialize LightRAG with existing data...")
                lightrag = LightRAG(
                    working_dir=self.storage_dir,
                    llm_model_func=llm_model_func,
                    embedding_func=embedding_func,
                )
                logger.info("LightRAG initialized successfully with existing data")
            except Exception as e:
                logger.warning(f"Failed to initialize LightRAG with existing data: {e}")
                lightrag = None
            
            # Create RAGAnything configuration
            config = RAGAnythingConfig(
                working_dir=self.storage_dir,
                parser=os.getenv("PARSER", "mineru"),
                parse_method=os.getenv("PARSE_METHOD", "auto"),
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )
            
            # Initialize RAGAnything with LightRAG integration
            rag_instance = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )
            
            # If we successfully created a LightRAG instance, try to set it
            if lightrag is not None:
                try:
                    rag_instance.lightrag = lightrag
                    logger.info("Successfully set LightRAG instance on RAGAnything")
                except Exception as e:
                    logger.warning(f"Failed to set LightRAG instance: {e}")
            
            logger.info("Fresh RAGAnything instance created for querying")
            return rag_instance
            
        except Exception as e:
            logger.error(f"Failed to create RAGAnything instance: {e}")
            raise
    
    async def process_document(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_path: str,
        parser: str = "mineru",
        parse_method: str = "auto",
        model: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single document with RAG-Anything
        
        Args:
            job_id: Processing job ID
            user_id: User ID
            project_id: Project ID
            workspace_id: Workspace ID
            gcs_path: GCS path to the document
            parser: Parser to use (mineru or docling)
            parse_method: Parse method (auto, ocr, txt)
            model: LLM model to use
            config: Additional configuration
            
        Returns:
            Processing result
        """
        start_time = datetime.utcnow()
        local_file_path = None
        
        try:
            # Update job status to processing
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'processing',
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 0,
                    'currentDocument': gcs_path
                }
            })
            
            # Download file from GCS
            logger.info(f"Downloading document from GCS: {gcs_path}")
            local_file_path = await self.gcs_manager.download_file(gcs_path)
            
            # Process document with RAG-Anything directly
            logger.info(f"Processing document with RAG-Anything: {local_file_path}")
            
            # Process the document using the new API
            await self.rag_anything.process_document_complete(
                file_path=local_file_path,
                output_dir=self.temp_dir,
                parse_method=parse_method
            )
            
            result = {
                'success': True,
                'source': 'raganything_processed',
                'graph_created': True
            }
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update job status to completed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'completed',
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 1,
                    'failedDocuments': 0,
                    'overallProgress': 100,
                    'currentDocument': gcs_path
                },
                'result': {
                    'processing_time': processing_time,
                    'document_path': gcs_path,
                    'local_path': local_file_path,
                    'rag_result': result,
                    'storage_info': {
                        'kv_storage_path': self.kv_storage_path,
                        'vector_storage_path': self.vector_storage_path,
                        'graph_storage_path': self.graph_storage_path,
                        'doc_status_storage_path': self.doc_status_storage_path
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Document processed successfully: {Path(gcs_path).name}',
                {
                    'processing_time': processing_time,
                    'document_count': 1,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method
                }
            )
            
            logger.info(f"Successfully processed document: {gcs_path}")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'document_path': gcs_path,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error processing document {gcs_path}: {e}")
            
            # Update job status to failed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'failed',
                'errorMessage': str(e),
                'progress': {
                    'totalDocuments': 1,
                    'processedDocuments': 0,
                    'failedDocuments': 1,
                    'overallProgress': 0,
                    'currentDocument': gcs_path
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'failed',
                f'Document processing failed: {str(e)}'
            )
            
            raise
            
        finally:
            # Clean up temporary files
            if local_file_path and os.path.exists(local_file_path):
                try:
                    await self.gcs_manager.cleanup_temp_files([local_file_path])
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary file {local_file_path}: {e}")
    
    async def process_folder(
        self,
        job_id: str,
        user_id: str,
        project_id: str,
        workspace_id: str,
        gcs_folder_path: str,
        file_extensions: List[str] = None,
        parser: str = "mineru",
        parse_method: str = "auto",
        model: str = "gpt-4",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process all documents in a folder with RAG-Anything
        
        Args:
            job_id: Processing job ID
            user_id: User ID
            project_id: Project ID
            workspace_id: Workspace ID
            gcs_folder_path: GCS path to the folder
            file_extensions: List of file extensions to process
            parser: Parser to use (mineru or docling)
            parse_method: Parse method (auto, ocr, txt)
            model: LLM model to use
            config: Additional configuration
            
        Returns:
            Processing result
        """
        start_time = datetime.utcnow()
        downloaded_files = {}
        
        try:
            # Set default file extensions
            if file_extensions is None:
                file_extensions = [".pdf", ".doc", ".docx", ".txt", ".md", ".jpg", ".png", ".bmp", ".tiff"]
            
            # List files in folder
            logger.info(f"Listing files in folder: {gcs_folder_path}")
            file_paths = await self.gcs_manager.list_files_in_folder(gcs_folder_path, file_extensions)
            
            if not file_paths:
                raise ValueError(f"No files found in folder {gcs_folder_path} with extensions {file_extensions}")
            
            total_files = len(file_paths)
            logger.info(f"Found {total_files} files to process")
            
            # Update job status with total file count
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'processing',
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': 0,
                    'failedDocuments': 0,
                    'overallProgress': 0,
                    'currentDocument': None
                }
            })
            
            # Download all files
            logger.info("Downloading files from GCS")
            downloaded_files = await self.gcs_manager.download_files(file_paths)
            
            if not downloaded_files:
                raise ValueError("Failed to download any files from GCS")
            
            # Use the initialized RAG-Anything instance
            
            # Process each document
            processed_count = 0
            failed_count = 0
            processing_results = []
            
            for i, (gcs_path, local_path) in enumerate(downloaded_files.items()):
                try:
                    logger.info(f"Processing document {i+1}/{total_files}: {gcs_path}")
                    
                    # Update progress
                    progress = int((i / total_files) * 100)
                    await self.firebase_manager.update_rag_processing_job(job_id, {
                        'progress': {
                            'totalDocuments': total_files,
                            'processedDocuments': processed_count,
                            'failedDocuments': failed_count,
                            'overallProgress': progress,
                            'currentDocument': gcs_path
                        }
                    })
                    
                    # Process the document using the new API
                    result = await self.rag_anything.process_document_complete(
                        file_path=local_path,
                        output_dir=self.temp_dir,
                        parse_method=parse_method
                    )
                    
                    processed_count += 1
                    processing_results.append({
                        'gcs_path': gcs_path,
                        'local_path': local_path,
                        'success': True,
                        'result': result
                    })
                    
                    logger.info(f"Successfully processed: {gcs_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {gcs_path}: {e}")
                    failed_count += 1
                    processing_results.append({
                        'gcs_path': gcs_path,
                        'local_path': local_path,
                        'success': False,
                        'error': str(e)
                    })
                    continue
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update job status to completed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'completed',
                'progress': {
                    'totalDocuments': total_files,
                    'processedDocuments': processed_count,
                    'failedDocuments': failed_count,
                    'overallProgress': 100,
                    'currentDocument': None
                },
                'result': {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': processed_count,
                    'failed_files': failed_count,
                    'folder_path': gcs_folder_path,
                    'processing_results': processing_results,
                    'storage_info': {
                        'kv_storage_path': self.kv_storage_path,
                        'vector_storage_path': self.vector_storage_path,
                        'graph_storage_path': self.graph_storage_path,
                        'doc_status_storage_path': self.doc_status_storage_path
                    }
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'completed',
                f'Folder processed successfully: {processed_count}/{total_files} documents processed',
                {
                    'processing_time': processing_time,
                    'total_files': total_files,
                    'processed_files': processed_count,
                    'failed_files': failed_count,
                    'model': model,
                    'parser': parser,
                    'parse_method': parse_method
                }
            )
            
            logger.info(f"Successfully processed folder: {gcs_folder_path} ({processed_count}/{total_files} documents)")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'total_files': total_files,
                'processed_files': processed_count,
                'failed_files': failed_count,
                'folder_path': gcs_folder_path,
                'results': processing_results
            }
            
        except Exception as e:
            logger.error(f"Error processing folder {gcs_folder_path}: {e}")
            
            # Update job status to failed
            await self.firebase_manager.update_rag_processing_job(job_id, {
                'status': 'failed',
                'errorMessage': str(e),
                'progress': {
                    'totalDocuments': len(downloaded_files) if downloaded_files else 0,
                    'processedDocuments': 0,
                    'failedDocuments': len(downloaded_files) if downloaded_files else 0,
                    'overallProgress': 0,
                    'currentDocument': None
                }
            })
            
            # Update project processing status
            await self.firebase_manager.update_project_processing_status(
                project_id,
                'failed',
                f'Folder processing failed: {str(e)}'
            )
            
            raise
            
        finally:
            # Clean up temporary files
            if downloaded_files:
                try:
                    local_paths = list(downloaded_files.values())
                    await self.gcs_manager.cleanup_temp_files(local_paths)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary files: {e}")
    
    async def get_processing_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing status for a job
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status information or None if not found
        """
        try:
            return await self.firebase_manager.get_rag_processing_job(job_id)
        except Exception as e:
            logger.error(f"Error getting processing status for job {job_id}: {e}")
            return None
    
    def get_approach_name(self) -> str:
        """Get the name of this RAG approach"""
        return "RAGAnything"
    
    def get_supported_parsers(self) -> List[str]:
        """Get list of supported parsers for this approach"""
        return ["mineru", "docling"]
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for this approach"""
        return ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    async def query_documents(
        self,
        user_id: str,
        project_id: str,
        query: str,
        model: str = "gpt-4o-mini",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query processed documents
        
        Args:
            user_id: User ID
            project_id: Project ID
            query: Query to ask about the documents
            model: LLM model to use for answering
            config: Additional configuration
            
        Returns:
            Query result with answer, sources, and metadata
        """
        try:
            logger.info(f"Querying documents for project {project_id} with query: {query[:100]}...")
            
            # Create a fresh RAGAnything instance for querying to ensure it loads existing data
            logger.info("Creating fresh RAGAnything instance for querying...")
            rag_instance = self._create_rag_instance()
            
            # Query the RAG system
            result = await rag_instance.aquery(query, mode="hybrid")
            
            # Handle different response formats
            if isinstance(result, str):
                # If result is a string, treat it as the answer
                answer = result
                sources = []
            elif isinstance(result, dict):
                # If result is a dict, extract answer and sources
                answer = result.get("answer", "")
                sources = result.get("sources", [])
            else:
                # Fallback: convert to string
                answer = str(result)
                sources = []
            
            # Format sources for API response
            formatted_sources = []
            for source in sources:
                formatted_sources.append({
                    "content": source.get("content", ""),
                    "metadata": source.get("metadata", {}),
                    "score": source.get("score", 0.0)
                })
            
            return {
                "answer": answer,
                "sources": formatted_sources,
                "metadata": {
                    "project_id": project_id,
                    "model": model,
                    "query_length": len(query),
                    "sources_count": len(formatted_sources)
                }
            }
            
        except Exception as e:
            logger.error(f"Error querying documents for project {project_id}: {e}")
            raise