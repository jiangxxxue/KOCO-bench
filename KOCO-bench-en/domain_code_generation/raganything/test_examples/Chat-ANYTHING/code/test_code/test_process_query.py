import pytest
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# ====== Path setup ======
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
backend_dir = os.path.join(project_root, 'rag-web-ui', 'backend')
sys.path.append(backend_dir)

from app.services.my_rag import MCPGeminiRAGClient


@pytest.fixture
def mock_client():
    """Create a fully mocked MCPGeminiRAGClient"""
    with patch('google.genai.Client') as mock_genai:
        mock_gemini_instance = MagicMock()
        mock_genai.return_value = mock_gemini_instance

        client = MCPGeminiRAGClient(gemini_api_key="test_key")

        # Mock RAG methods
        client.rag.aquery = AsyncMock(return_value="RAG answer about AI")
        client.rag.chat = AsyncMock(return_value="Final answer from tools")

        # Mock get_mcp_tools
        client.get_mcp_tools = AsyncMock(return_value=["tool1", "tool2"])

        # Reset history to ensure test independence
        client.history = []

        return client


def _create_gemini_mock_response(text: str):
    """Helper function: Create a reliable Gemini mock response"""
    part = MagicMock()
    part.text = text

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


# ==============================
# Branch 1: Gemini judges as YES
# ==============================
@pytest.mark.asyncio
async def test_process_query_gemini_approves_rag(mock_client):
    # Use helper function to create mock response
    mock_judge_response = _create_gemini_mock_response("YES")
    mock_client.gemini.models.generate_content.return_value = mock_judge_response

    query = "What is RAG?"
    result = await mock_client.process_query(query)

    # Verify behavior
    mock_client.rag.aquery.assert_awaited_once_with(query, mode="hybrid")
    mock_client.gemini.models.generate_content.assert_called_once()
    mock_client.rag.chat.assert_not_called()  # chat not called

    # Verify history
    assert len(mock_client.history) == 2
    assert mock_client.history[0] == {"role": "user", "content": query}
    assert mock_client.history[1] == {"role": "model", "content": "RAG answer about AI"}

    # Verify return value
    assert result == "RAG answer about AI"


# ==============================
# Branch 2: Gemini judges as NO
# ==============================
@pytest.mark.asyncio
async def test_process_query_gemini_rejects_rag(mock_client):
    # Use helper function to create mock response
    mock_judge_response = _create_gemini_mock_response("NO")
    mock_client.gemini.models.generate_content.return_value = mock_judge_response

    query = "What is the weather in Tokyo?"
    result = await mock_client.process_query(query)

    expected_history_actual_value = [
        {'role': 'user', 'content': 'What is the weather in Tokyo?'},
        {'role': 'model', 'content': 'Final answer from tools'}
    ]

    mock_client.rag.aquery.assert_awaited_once_with(query, mode="hybrid")
    mock_client.gemini.models.generate_content.assert_called_once()

    mock_client.rag.chat.assert_awaited_once_with(
        query,
        tools=["tool1", "tool2"],
        multimodal=True,
        history_messages=expected_history_actual_value
    )

    assert mock_client.history == expected_history_actual_value
    assert result == "Final answer from tools"


@pytest.mark.asyncio
async def test_process_query_gemini_judge_case_insensitive(mock_client):
    # Return lowercase "yes"
    mock_judge_response = _create_gemini_mock_response("yes")
    mock_client.gemini.models.generate_content.return_value = mock_judge_response

    result = await mock_client.process_query("Test query")
    assert result == "RAG answer about AI"
    mock_client.rag.chat.assert_not_called()