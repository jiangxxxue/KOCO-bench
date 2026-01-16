"""
Document data model
"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ProcessedDocument:
    """Data class for processed documents"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_path: str = ""
    processed_path: str = ""
    text_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_type: str = ""
    file_size: int = 0
    processed_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
