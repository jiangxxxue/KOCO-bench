import pytest
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from lightrag.utils import EmbeddingFunc

# ====== 关键：正确计算项目路径（解决您提供的路径结构） ======
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录 = testcode 的父目录 (code)
project_root = os.path.dirname(current_dir)  # code目录

# 添加 rag-web-ui/backend 到 Python 路径
backend_dir = os.path.join(project_root, 'rag-web-ui', 'backend')
sys.path.append(backend_dir)

# 现在可以正确导入（从 backend/app/services 导入）
from app.services.my_rag import MCPGeminiRAGClient


@pytest.mark.asyncio
async def test_rag_initialization():
    """测试 RAG 初始化配置（屏蔽所有外部依赖）"""
    mock_api_key = "test_api_key"

    # 1. 模拟 Google GenAI Client（完全屏蔽真实调用）
    with patch('google.genai.Client', return_value=MagicMock()):
        # 2. 创建客户端实例（不触发任何 API 调用）
        client = MCPGeminiRAGClient(mock_api_key)

        # 3. 验证 RAG 配置（与代码中定义的完全一致）
        rag_config = client.rag.config

        # 核心配置验证
        assert rag_config.working_dir == "./rag_storage"
        assert rag_config.enable_table_processing is True
        assert rag_config.enable_equation_processing is True
        assert rag_config.context_window == 5
        assert rag_config.context_mode == "chunk"
        assert rag_config.include_headers is True
        assert rag_config.include_captions is True
        assert rag_config.context_filter_content_types == ["text"]

        # 补充配置验证
        assert rag_config.max_context_tokens == 4000
        assert rag_config.parser == "mineru"
        assert rag_config.parse_method == "auto"

        # 验证依赖函数传入
        assert client.rag.embedding_func is not None
        assert client.rag.llm_model_func is not None

    print("✅ RAG 初始化测试通过！所有配置验证通过")

@pytest.mark.asyncio
async def test_rag_integration_init():
    """集成测试：验证 __init__ 能正确组装 RAGAnything 及其依赖函数（不调用外部服务）"""
    mock_api_key = "test_api_key"

    with patch('google.genai.Client') as mock_genai_client:
        mock_genai_instance = MagicMock()
        mock_genai_client.return_value = mock_genai_instance

        client = MCPGeminiRAGClient(mock_api_key)

        # 验证 Gemini client 被正确初始化
        mock_genai_client.assert_called_once_with(api_key=mock_api_key)

        # 验证 RAG 实例存在
        assert client.rag is not None
        assert client.rag.embedding_func is not None

        # 检查 func 调用后返回的是 coroutine，而非 func 本身是 async
        emb_func = client.rag.embedding_func
        assert isinstance(emb_func, EmbeddingFunc)
        assert emb_func.embedding_dim == 3072
        assert emb_func.max_token_size == 8192
        assert callable(emb_func.func)

        # 调用 func，检查返回值是否为 coroutine（即使没真执行）
        dummy_texts = ["hello"]
        result = emb_func.func(dummy_texts)
        import asyncio
        assert asyncio.iscoroutine(result), f"Expected coroutine, got {type(result)}"

        # 同样检查 llm_model_func 是否可 await（它本身就是 async def）
        llm_func = client.rag.llm_model_func
        import inspect
        assert inspect.iscoroutinefunction(llm_func), "llm_model_func should be async def"

        # 其他简单验证
        assert client.history == []
        assert client.session is None

    print("✅ RAG 集成初始化测试通过！")