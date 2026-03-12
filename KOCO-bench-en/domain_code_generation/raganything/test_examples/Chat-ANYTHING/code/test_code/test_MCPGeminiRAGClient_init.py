import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from lightrag.utils import EmbeddingFunc

# ====== Key: Correctly calculate project path (resolves your provided path structure) ======
current_dir = os.path.dirname(os.path.abspath(__file__))
# Project root directory = parent directory of testcode (code)
project_root = os.path.dirname(current_dir)  # code directory

# Add rag-web-ui/backend to Python path
backend_dir = os.path.join(project_root, 'rag-web-ui', 'backend')
sys.path.append(backend_dir)

# Now can import correctly (from backend/app/services)
from app.services.my_rag import MCPGeminiRAGClient


@pytest.mark.asyncio
async def test_rag_initialization():
    """Test RAG initialization configuration (shield all external dependencies)"""
    mock_api_key = "test_api_key"

    # 1. Mock Google GenAI Client (completely shield real calls)
    with patch('google.genai.Client', return_value=MagicMock()):
        # 2. Create client instance (without triggering any API calls)
        client = MCPGeminiRAGClient(mock_api_key)

        # 3. Verify RAG configuration (exactly as defined in code)
        rag_config = client.rag.config

        # Core configuration verification
        assert rag_config.working_dir == "./rag_storage"
        assert rag_config.enable_table_processing is True
        assert rag_config.enable_equation_processing is True
        assert rag_config.context_window == 5
        assert rag_config.context_mode == "chunk"
        assert rag_config.include_headers is True
        assert rag_config.include_captions is True
        assert rag_config.context_filter_content_types == ["text"]

        # Additional configuration verification
        assert rag_config.max_context_tokens == 4000
        assert rag_config.parser == "mineru"
        assert rag_config.parse_method == "auto"

        # Verify dependency functions are passed in
        assert client.rag.embedding_func is not None
        assert client.rag.llm_model_func is not None

    print("✅ RAG initialization test passed! All configuration verifications passed")

@pytest.mark.asyncio
async def test_rag_integration_init():
    """Integration test: Verify __init__ correctly assembles RAGAnything and its dependency functions (without calling external services)"""
    mock_api_key = "test_api_key"

    with patch('google.genai.Client') as mock_genai_client:
        mock_genai_instance = MagicMock()
        mock_genai_client.return_value = mock_genai_instance

        client = MCPGeminiRAGClient(mock_api_key)

        # Verify Gemini client was initialized correctly
        mock_genai_client.assert_called_once_with(api_key=mock_api_key)

        # Verify RAG instance exists
        assert client.rag is not None
        assert client.rag.embedding_func is not None

        # Check that func call returns a coroutine, not that func itself is async
        emb_func = client.rag.embedding_func
        assert isinstance(emb_func, EmbeddingFunc)
        assert emb_func.embedding_dim == 3072
        assert emb_func.max_token_size == 8192
        assert callable(emb_func.func)

        # Call func, check if return value is a coroutine (even if not actually executed)
        dummy_texts = ["hello"]
        result = emb_func.func(dummy_texts)
        import asyncio
        assert asyncio.iscoroutine(result), f"Expected coroutine, got {type(result)}"

        # Similarly check if llm_model_func can be awaited (it itself is async def)
        llm_func = client.rag.llm_model_func
        import inspect
        assert inspect.iscoroutinefunction(llm_func), "llm_model_func should be async def"

        # Other simple verifications
        assert client.history == []
        assert client.session is None

    print("✅ RAG integration initialization test passed!")