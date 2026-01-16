"""
Evaluation framework for RAG approach comparison
"""

from .suites.sample_test_suites import get_all_sample_evaluation_suites
from .suites.real_world_test_suites import get_real_world_evaluation_suites
from .suites.unstructured_test_suites import get_unstructured_evaluation_suites
from .evaluation_service import EvaluationService
from .eval_models import (
    EvaluationCase, EvaluationSuite, EvaluationRun, EvaluationCaseResult,
    EvaluationRequest, EvaluationResponse, EvaluationStatusResponse,
    EvaluationResultsResponse, EvaluationStatus, MetricType, EvaluationType
)

def get_all_evaluation_suites():
    """Get all available evaluation suites"""
    sample_suites = get_all_sample_evaluation_suites()
    real_world_suites = get_real_world_evaluation_suites()
    unstructured_suites = get_unstructured_evaluation_suites()
    return {**sample_suites, **real_world_suites, **unstructured_suites}

__all__ = [
    'get_all_evaluation_suites',
    'get_all_sample_evaluation_suites', 
    'get_real_world_evaluation_suites',
    'get_unstructured_evaluation_suites',
    'EvaluationService',
    'EvaluationCase', 'EvaluationSuite', 'EvaluationRun', 'EvaluationCaseResult',
    'EvaluationRequest', 'EvaluationResponse', 'EvaluationStatusResponse',
    'EvaluationResultsResponse', 'EvaluationStatus', 'MetricType', 'EvaluationType'
]