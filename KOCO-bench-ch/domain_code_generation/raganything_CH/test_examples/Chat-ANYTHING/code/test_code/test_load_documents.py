import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

# ====== 路径设置（与你的结构一致）======
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # code/
backend_dir = os.path.join(project_root, 'rag-web-ui', 'backend')
sys.path.append(backend_dir)

from app.services.my_rag import MCPGeminiRAGClient


@pytest.fixture
def mock_gemini_client():
    """提供一个已初始化的 MCPGeminiRAGClient（mock Gemini）"""
    with patch('google.genai.Client', return_value=MagicMock()):
        client = MCPGeminiRAGClient(gemini_api_key="test_key")
        return client


# ======================
# 单元测试：只验证调用行为
# ======================
@pytest.mark.asyncio
async def test_load_documents_unit(mock_gemini_client):
    """单元测试：验证 load_documents 正确调用 RAG 方法并设置标志"""
    client = mock_gemini_client

    # Mock RAG 的 process_document_complete 为异步 mock
    with patch.object(client.rag, 'process_document_complete', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = None

        # 调用被测函数
        await client.load_documents("dummy.pdf")

        # 验证 RAG 方法被正确调用
        mock_process.assert_awaited_once_with(
            file_path="dummy.pdf",
            output_dir="./output",
            parse_method="auto"
        )

        # 验证状态标志被设置
        assert client.document_loaded is True

    print("✅ load_documents 单元测试通过：调用和状态正确")

