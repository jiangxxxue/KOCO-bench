import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# ====== Mock firebase_admin ======
sys.modules["firebase_admin"] = MagicMock()
sys.modules["firebase_admin.credentials"] = MagicMock()
sys.modules["firebase_admin.firestore"] = MagicMock()
sys.modules["firebase_admin.storage"] = MagicMock()

# ====== Add project root to path ======
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# ====== Import after path setup ======
from src.services.rag_anything_service import RAGAnythingService, EmbeddingFunc


@pytest.fixture
def mock_managers():
    return {
        "firebase_manager": MagicMock(),
        "gcs_manager": MagicMock(),
        "auth_service": MagicMock(),
    }


@pytest.mark.asyncio
async def test_create_rag_instance_unit(mock_managers):
    with patch("src.services.rag_anything_service.LightRAG") as mock_lightrag_class, \
         patch("src.services.rag_anything_service.RAGAnything") as mock_rag_class, \
         patch("src.services.rag_anything_service.openai_complete_if_cache") as mock_llm, \
         patch("src.services.rag_anything_service.openai_embed") as mock_emb:

        mock_llm.return_value = "mocked response"
        mock_emb.return_value = [[0.1] * 3072]
        mock_rag_instance = MagicMock()
        mock_rag_class.return_value = mock_rag_instance

        # 禁用自动初始化，避免干扰
        with patch.object(RAGAnythingService, '_initialize_rag_storage'):
            service = RAGAnythingService(**mock_managers)

        rag_instance = service._create_rag_instance()

        # 现在 RAGAnything 应该只被调用一次
        mock_rag_class.assert_called_once()
        call_kwargs = mock_rag_class.call_args.kwargs
        config = call_kwargs["config"]
        assert config.parser == "mineru"
        assert config.working_dir == service.storage_dir
        assert config.enable_image_processing is True
        assert config.enable_table_processing is True
        assert config.enable_equation_processing is True

        emb_func = call_kwargs["embedding_func"]
        assert emb_func.embedding_dim == 3072

        assert rag_instance is not None


@pytest.mark.asyncio
async def test_create_rag_instance_integration(mock_managers, monkeypatch, tmp_path):
    """集成测试：验证 _create_rag_instance 创建功能可用的 RAG 实例"""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()  # 确保存在

    monkeypatch.setenv("RAG_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    async def fake_llm(*args, **kwargs):
        return "ok"

    async def fake_embed(texts, **kwargs):
        return [[0.1] * 3072 for _ in texts]

    with patch("src.services.rag_anything_service.openai_complete_if_cache", side_effect=fake_llm), \
         patch("src.services.rag_anything_service.openai_embed", side_effect=fake_embed):

        service = RAGAnythingService(**mock_managers)
        rag_instance = service._create_rag_instance()

        assert rag_instance is not None

        # 验证 embedding 可执行
        result = await rag_instance.embedding_func.func(["hello"])
        assert len(result[0]) == 3072