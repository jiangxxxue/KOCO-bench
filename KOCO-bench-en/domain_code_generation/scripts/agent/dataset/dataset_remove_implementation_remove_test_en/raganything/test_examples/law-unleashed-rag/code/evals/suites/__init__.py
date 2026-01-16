"""
Test suite definitions for RAG evaluation
"""

from .sample_test_suites import get_all_sample_evaluation_suites
from .real_world_test_suites import get_real_world_evaluation_suites

__all__ = [
    'get_all_sample_evaluation_suites',
    'get_real_world_evaluation_suites'
]

