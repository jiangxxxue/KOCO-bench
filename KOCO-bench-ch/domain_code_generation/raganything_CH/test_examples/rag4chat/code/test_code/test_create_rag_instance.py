import sys
import os

# ==============================
# 🔑 关键：在导入任何东西前，先 mock sentence_transformers
# ==============================
from unittest.mock import MagicMock

# 创建一个 fake CrossEncoder 类，实例化时不干任何事
class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        # 返回与输入数量一致的假分数（例如全为 0.5）
        return [0.5] * len(pairs)

# 替换 sys.modules 中的 sentence_transformers.CrossEncoder
# 这样当 build_database.py 执行 from sentence_transformers import CrossEncoder 时，拿到的是 fake 的
import types
fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.CrossEncoder = FakeCrossEncoder
sys.modules["sentence_transformers"] = fake_sentence_transformers

# ==============================
# 现在可以安全地设置路径并导入其他模块
# ==============================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 设置 Hugging Face 离线（双重保险）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ==============================
# 正常导入 pytest 和其他依赖
# ==============================
import pytest
from raganything import RAGAnythingConfig, RAGAnything
from lightrag.utils import EmbeddingFunc
from unittest.mock import patch, AsyncMock, MagicMock

# ⚠️ 注意：现在才导入 RAGDatabaseManager！此时 sentence_transformers 已被 mock
from services.build_database import RAGDatabaseManager


@pytest.mark.asyncio
async def test_create_rag_instance_unit():
    """Unit test for _create_rag_instance: mocks all external dependencies."""

    manager = RAGDatabaseManager(
        working_dir="./test_work",
        output_dir="./test_out",
        llm_model="mock-llm",
        embed_model="mock-embed",
        vision_model="mock-vision",
        parser="mineru"
    )

    # Mock 所有外部依赖（注意 patch 路径是 services.build_database）
    with patch("services.build_database.ollama_complete_async", new_callable=AsyncMock) as mock_llm, \
         patch("services.build_database.ollama_vision_complete_async", new_callable=AsyncMock) as mock_vision, \
         patch("services.build_database.rerank_model_func", None), \
         patch("services.build_database.RAGAnything") as mock_rag_class, \
         patch("services.build_database.AsyncEmbeddingWrapper") as mock_emb_wrapper:

        # 配置 mocks
        mock_llm.return_value = "mocked llm response"
        mock_vision.return_value = "mocked vision response"
        mock_rag_instance = MagicMock()
        mock_rag_class.return_value = mock_rag_instance

        mock_emb_instance = MagicMock()
        mock_emb_instance.embed = AsyncMock(return_value=[[0.1] * 1024])
        mock_emb_wrapper.return_value = mock_emb_instance

        # Act
        rag_instance = await manager._create_rag_instance()

        # Assert
        assert mock_rag_class.called
        call_kwargs = mock_rag_class.call_args.kwargs
        config = call_kwargs["config"]
        assert isinstance(config, RAGAnythingConfig)
        assert config.working_dir == "./test_work"
        assert config.parser == "mineru"
        assert config.enable_image_processing is True

        assert hasattr(manager, "llm_model_func")
        assert hasattr(manager, "vision_model_func")
        assert hasattr(manager, "embedding_func")
        assert isinstance(manager.embedding_func, EmbeddingFunc)
        assert manager.embedding_func.embedding_dim == 1024
        assert manager.embedding_func.max_token_size == 512

        assert "rerank_model_func" not in call_kwargs
        assert rag_instance == mock_rag_instance

        print("✅ _create_rag_instance unit test passed!")