"""
Google Cloud Storage utilities for downloading files
"""

import logging
import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class GCSManager:
    """Manager for Google Cloud Storage operations"""
    
    def __init__(self):
        self.bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable not set")
        
        # Initialize GCS client
        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"GCS client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test GCS connection"""
        try:
            # Try to get bucket metadata
            self.bucket.reload()
            logger.info("GCS connection test successful")
            return True
        except Exception as e:
            logger.error(f"GCS connection test failed: {e}")
            return False
    
    def _parse_gcs_path(self, gcs_path: str) -> tuple[str, str]:
        """
        Parse GCS path to extract bucket and blob name
        
        Args:
            gcs_path: GCS path (gs://bucket/path or just path)
            
        Returns:
            Tuple of (bucket_name, blob_name)
        """
        if gcs_path.startswith('gs://'):
            # Remove gs:// prefix
            path_without_prefix = gcs_path[5:]
            # Split into bucket and blob name
            parts = path_without_prefix.split('/', 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            else:
                # Only bucket name provided
                return parts[0], ""
        else:
            # Assume it's just the blob name in the configured bucket
            return self.bucket_name, gcs_path
    
    async def list_files_in_folder(self, folder_path: str, file_extensions: List[str] = None) -> List[str]:
        """
        List files in a GCS folder
        
        Args:
            folder_path: GCS folder path
            file_extensions: List of file extensions to filter by
            
        Returns:
            List of file paths in the folder
        """
        try:
            bucket_name, blob_prefix = self._parse_gcs_path(folder_path)
            
            # Ensure folder path ends with /
            if blob_prefix and not blob_prefix.endswith('/'):
                blob_prefix += '/'
            
            # List blobs with the prefix
            blobs = self.client.list_blobs(bucket_name, prefix=blob_prefix)
            
            files = []
            for blob in blobs:
                # Skip directories (blobs ending with /)
                if blob.name.endswith('/'):
                    continue
                
                # Check file extension if specified
                if file_extensions:
                    file_path = Path(blob.name)
                    if file_path.suffix.lower() not in file_extensions:
                        continue
                
                # Return full GCS path
                files.append(f"gs://{bucket_name}/{blob.name}")
            
            logger.info(f"Found {len(files)} files in folder {folder_path}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in folder {folder_path}: {e}")
            return []
    
    async def download_file(self, gcs_path: str, local_path: Optional[str] = None) -> str:
        """
        Download a file from GCS to local storage
        
        Args:
            gcs_path: GCS path to the file
            local_path: Local path to save the file (optional)
            
        Returns:
            Local path where the file was saved
        """
        try:
            bucket_name, blob_name = self._parse_gcs_path(gcs_path)
            
            # Get blob
            blob = self.bucket.blob(blob_name)
            
            # Check if blob exists
            if not blob.exists():
                raise NotFound(f"File not found: {gcs_path}")
            
            # Generate local path if not provided
            if not local_path:
                # Create temp directory
                temp_dir = tempfile.mkdtemp(prefix="rag_processing_")
                filename = Path(blob_name).name
                local_path = os.path.join(temp_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            blob.download_to_filename(local_path)
            
            logger.info(f"Downloaded {gcs_path} to {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading file {gcs_path}: {e}")
            raise
    
    async def download_files(self, gcs_paths: List[str], temp_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Download multiple files from GCS
        
        Args:
            gcs_paths: List of GCS paths
            temp_dir: Temporary directory to save files (optional)
            
        Returns:
            Dictionary mapping GCS paths to local paths
        """
        try:
            if not temp_dir:
                temp_dir = tempfile.mkdtemp(prefix="rag_processing_")
            
            downloaded_files = {}
            
            for gcs_path in gcs_paths:
                try:
                    filename = Path(gcs_path).name
                    local_path = os.path.join(temp_dir, filename)
                    downloaded_path = await self.download_file(gcs_path, local_path)
                    downloaded_files[gcs_path] = downloaded_path
                except Exception as e:
                    logger.error(f"Failed to download {gcs_path}: {e}")
                    # Continue with other files
                    continue
            
            logger.info(f"Downloaded {len(downloaded_files)} files to {temp_dir}")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading files: {e}")
            raise
    
    async def get_file_info(self, gcs_path: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from GCS
        
        Args:
            gcs_path: GCS path to the file
            
        Returns:
            File information or None if not found
        """
        try:
            bucket_name, blob_name = self._parse_gcs_path(gcs_path)
            blob = self.bucket.blob(blob_name)
            
            if not blob.exists():
                return None
            
            blob.reload()
            
            return {
                'name': blob.name,
                'size': blob.size,
                'content_type': blob.content_type,
                'created': blob.time_created,
                'updated': blob.updated,
                'md5_hash': blob.md5_hash,
                'crc32c': blob.crc32c
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {gcs_path}: {e}")
            return None
    
    async def cleanup_temp_files(self, local_paths: List[str]):
        """
        Clean up temporary files
        
        Args:
            local_paths: List of local file paths to delete
        """
        try:
            for local_path in local_paths:
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                        logger.debug(f"Deleted temporary file: {local_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {local_path}: {e}")
            
            # Also try to remove parent directories if they're empty
            for local_path in local_paths:
                try:
                    parent_dir = os.path.dirname(local_path)
                    if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                        logger.debug(f"Deleted empty directory: {parent_dir}")
                except Exception as e:
                    logger.debug(f"Could not remove directory {parent_dir}: {e}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
