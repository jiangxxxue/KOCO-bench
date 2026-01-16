"""
Authentication service for user and workspace access control
"""

import logging
from typing import Optional, Dict, Any
from src.utils.firebase_utils import FirebaseManager

logger = logging.getLogger(__name__)


class AuthService:
    """Service for handling authentication and authorization"""
    
    def __init__(self, firebase_manager: FirebaseManager):
        self.firebase_manager = firebase_manager
    
    async def verify_user_exists(self, user_id: str) -> bool:
        """
        Verify that a user exists in the system
        
        Args:
            user_id: User ID to verify
            
        Returns:
            True if user exists, False otherwise
        """
        try:
            user_doc = await self.firebase_manager.get_user_document(user_id)
            return user_doc is not None
        except Exception as e:
            logger.error(f"Error verifying user existence: {e}")
            return False
    
    async def verify_workspace_access(self, user_id: str, workspace_id: str) -> bool:
        """
        Verify that a user has access to a workspace
        
        Args:
            user_id: User ID
            workspace_id: Workspace ID
            
        Returns:
            True if user has access, False otherwise
        """
        try:
            # Get user document
            user_doc = await self.firebase_manager.get_user_document(user_id)
            if not user_doc:
                logger.warning(f"User {user_id} not found")
                return False
            
            # Check if user is admin (admins have access to all workspaces)
            if user_doc.get('isAdmin', False):
                logger.info(f"Admin user {user_id} has access to workspace {workspace_id}")
                return True
            
            # Check workspace memberships
            memberships = user_doc.get('memberships', [])
            for membership in memberships:
                if membership.get('workspaceId') == workspace_id:
                    logger.info(f"User {user_id} has access to workspace {workspace_id}")
                    return True
            
            logger.warning(f"User {user_id} does not have access to workspace {workspace_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying workspace access: {e}")
            return False
    
    async def verify_project_access(self, user_id: str, project_id: str) -> bool:
        """
        Verify that a user has access to a project
        
        Args:
            user_id: User ID
            project_id: Project ID (may be a storage key like "NbbabGQy3gkCJIDzkSoE_vertex_rag_1")
            
        Returns:
            True if user has access, False otherwise
        """
        try:
            # Extract actual project ID from storage key if needed
            # Storage keys are in format: "actual_project_id_approach_iteration"
            actual_project_id = project_id
            if '_' in project_id:
                # Extract the first part before the first underscore as the actual project ID
                actual_project_id = project_id.split('_')[0]
                logger.info(f"Extracted project ID {actual_project_id} from storage key {project_id}")
            
            # Get project document using the actual project ID
            project_doc = await self.firebase_manager.get_project_document(actual_project_id)
            if not project_doc:
                logger.warning(f"Project {actual_project_id} not found")
                return False
            
            # Check if user is the project owner
            if project_doc.get('userId') == user_id:
                logger.info(f"User {user_id} is owner of project {project_id}")
                return True
            
            # Check workspace access
            workspace_id = project_doc.get('workspaceId')
            if workspace_id:
                has_workspace_access = await self.verify_workspace_access(user_id, workspace_id)
                if has_workspace_access:
                    logger.info(f"User {user_id} has workspace access to project {project_id}")
                    return True
            
            logger.warning(f"User {user_id} does not have access to project {project_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error verifying project access: {e}")
            return False
    
    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user information
        
        Args:
            user_id: User ID
            
        Returns:
            User information or None if not found
        """
        try:
            return await self.firebase_manager.get_user_document(user_id)
        except Exception as e:
            logger.error(f"Error getting user info: {e}")
            return None
    
    async def get_project_info(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project information
        
        Args:
            project_id: Project ID
            
        Returns:
            Project information or None if not found
        """
        try:
            return await self.firebase_manager.get_project_document(project_id)
        except Exception as e:
            logger.error(f"Error getting project info: {e}")
            return None
