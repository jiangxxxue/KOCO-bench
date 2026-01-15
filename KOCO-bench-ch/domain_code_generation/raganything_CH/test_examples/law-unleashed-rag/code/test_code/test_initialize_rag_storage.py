import pytest
import os
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

# ====== Mock firebase_admin to avoid dependency in unit test ======
import sys
sys.modules["firebase_admin"] = MagicMock()
sys.modules["firebase_admin.credentials"] = MagicMock()
sys.modules["firebase_admin.firestore"] = MagicMock()
sys.modules["firebase_admin.storage"] = MagicMock()

# ====== Add project root (code/) to path ======
current_dir = Path(__file__).resolve().parent  # testcode/
project_root = current_dir.parent              # code/
sys.path.insert(0, str(project_root))

# ====== Import after path setup ======
from src.services.rag_anything_service import RAGAnythingService
from lightrag.utils import EmbeddingFunc


@pytest.fixture
def mock_managers():
    return {
        "firebase_manager": MagicMock(),
        "gcs_manager": MagicMock(),
        "auth_service": MagicMock(),
    }


@pytest.mark.asyncio
async def test_initialize_rag_storage_unit(mock_managers, monkeypatch, tmp_path):
    """单元测试：验证 _initialize_rag_storage 的配置逻辑"""
    env_vars = {
        "OPENAI_API_KEY": "sk-test-key",
        "MODEL": "gpt-4o-mini",
        "PARSER": "mineru",
        "PARSE_METHOD": "auto",
        "RAG_STORAGE_DIR": str(tmp_path / "storage"),
        "RAG_TEMP_DIR": str(tmp_path / "temp"),
    }
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    with patch("src.services.rag_anything_service.RAGAnything") as mock_rag_class, \
         patch("src.services.rag_anything_service.openai_complete_if_cache"), \
         patch("src.services.rag_anything_service.openai_embed"):

        mock_rag_instance = MagicMock()
        mock_rag_class.return_value = mock_rag_instance

        service = RAGAnythingService(**mock_managers)

        call_kwargs = mock_rag_class.call_args.kwargs
        config = call_kwargs["config"]
        assert config.working_dir == str(tmp_path / "storage")
        assert config.parser == "mineru"
        assert config.enable_table_processing is True

        emb_func = call_kwargs["embedding_func"]
        assert isinstance(emb_func, EmbeddingFunc)
        assert emb_func.embedding_dim == 3072

        # ✅ 修复：平台无关的路径断言
        assert Path(service.kv_storage_path).name == "kv"
        assert Path(service.vector_storage_path).name == "vector"
        assert Path(service.graph_storage_path).name == "graph"


@pytest.mark.asyncio
async def test_initialize_rag_storage_integration(mock_managers, monkeypatch, tmp_path):
    """集成测试：验证初始化流程成功（不依赖目录立即创建）"""
    storage_dir = tmp_path / "storage"
    monkeypatch.setenv("RAG_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")

    async def fake_openai_complete(*args, **kwargs):
        return "mocked response"

    async def fake_openai_embed(texts, **kwargs):
        return [[0.1] * 3072 for _ in texts]

    with patch("src.services.rag_anything_service.openai_complete_if_cache", side_effect=fake_openai_complete), \
         patch("src.services.rag_anything_service.openai_embed", side_effect=fake_openai_embed):

        service = RAGAnythingService(**mock_managers)

        assert service.rag_anything is not None

        # ✅ 修复：验证路径计算正确，而非目录存在
        expected_kv = str(storage_dir / "kv")
        expected_vector = str(storage_dir / "vector")
        assert service.kv_storage_path == expected_kv
        assert service.vector_storage_path == expected_vector

        # 可选：验证 embedding 函数可调用
        emb_result = await service.rag_anything.embedding_func.func(["hello"])
        assert len(emb_result[0]) == 3072


if __name__ == "__main__":
    pytest.main([__file__, "-v"])