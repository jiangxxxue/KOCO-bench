"""
Mindmap result data model
"""
from datetime import datetime
from typing import Dict
from dataclasses import dataclass, field


@dataclass 
class MindmapResult:
    """Data class for mindmap generation results"""
    document_id: str
    mermaid_syntax: str = ""
    html_content: str = ""
    markdown_outline: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    provider: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)
