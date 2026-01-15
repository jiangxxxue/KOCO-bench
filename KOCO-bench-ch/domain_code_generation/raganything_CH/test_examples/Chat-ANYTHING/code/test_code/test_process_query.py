import pytest
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# ====== 路径设置 ======
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
backend_dir = os.path.join(project_root, 'rag-web-ui', 'backend')
sys.path.append(backend_dir)

from app.services.my_rag import MCPGeminiRAGClient


@pytest.fixture
def mock_client():
    """创建一个完全 mock 的 MCPGeminiRAGClient"""
    with patch('google.genai.Client') as mock_genai:
        mock_gemini_instance = MagicMock()
        mock_genai.return_value = mock_gemini_instance

        client = MCPGeminiRAGClient(gemini_api_key="test_key")

        # Mock RAG 方法
        client.rag.aquery = AsyncMock(return_value="RAG answer about AI")
        client.rag.chat = AsyncMock(return_value="Final answer from tools")

        # Mock get_mcp_tools
        client.get_mcp_tools = AsyncMock(return_value=["tool1", "tool2"])

        # 重置 history 确保每次测试独立
        client.history = []

        return client


def _create_gemini_mock_response(text: str):
    """辅助函数：创建可靠的 Gemini mock 响应"""
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
# 分支 1: Gemini 判断为 YES
# ==============================
@pytest.mark.asyncio
async def test_process_query_gemini_approves_rag(mock_client):
    # 使用辅助函数创建 mock 响应
    mock_judge_response = _create_gemini_mock_response("YES")
    mock_client.gemini.models.generate_content.return_value = mock_judge_response

    query = "What is RAG?"
    result = await mock_client.process_query(query)

    # 验证行为
    mock_client.rag.aquery.assert_awaited_once_with(query, mode="hybrid")
    mock_client.gemini.models.generate_content.assert_called_once()
    mock_client.rag.chat.assert_not_called()  # 未调用 chat

    # 验证 history
    assert len(mock_client.history) == 2
    assert mock_client.history[0] == {"role": "user", "content": query}
    assert mock_client.history[1] == {"role": "model", "content": "RAG answer about AI"}

    # 验证返回值
    assert result == "RAG answer about AI"


# ==============================
# 分支 2: Gemini 判断为 NO
# ==============================
@pytest.mark.asyncio
async def test_process_query_gemini_rejects_rag(mock_client):
    # 使用辅助函数创建 mock 响应
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
    # 返回小写 "yes"
    mock_judge_response = _create_gemini_mock_response("yes")
    mock_client.gemini.models.generate_content.return_value = mock_judge_response

    result = await mock_client.process_query("Test query")
    assert result == "RAG answer about AI"
    mock_client.rag.chat.assert_not_called()