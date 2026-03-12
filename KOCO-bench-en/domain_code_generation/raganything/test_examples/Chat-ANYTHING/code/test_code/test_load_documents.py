import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# ====== Path setup (consistent with your structure) ======
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # code/
backend_dir = os.path.join(project_root, 'rag-web-ui', 'backend')
sys.path.append(backend_dir)

from app.services.my_rag import MCPGeminiRAGClient


@pytest.fixture
def mock_gemini_client():
    """Provide an initialized MCPGeminiRAGClient (mock Gemini)"""
    with patch('google.genai.Client', return_value=MagicMock()):
        client = MCPGeminiRAGClient(gemini_api_key="test_key")
        return client


# ======================
# Unit test: Only verify call behavior
# ======================
@pytest.mark.asyncio
async def test_load_documents_unit(mock_gemini_client):
    """Unit test: Verify load_documents correctly calls RAG method and sets flag"""
    client = mock_gemini_client

    # Mock RAG's process_document_complete as async mock
    with patch.object(client.rag, 'process_document_complete', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = None

        # Call the function under test
        await client.load_documents("dummy.pdf")

        # Verify RAG method was called correctly
        mock_process.assert_awaited_once_with(
            file_path="dummy.pdf",
            output_dir="./output",
            parse_method="auto"
        )

        # Verify status flag was set
        assert client.document_loaded is True

    print("✅ load_documents unit test passed: calls and state are correct")

