import sys
import os

# ==============================
# 🔑 Key: Mock sentence_transformers before importing anything
# ==============================
from unittest.mock import MagicMock

# Create a fake CrossEncoder class that does nothing when instantiated
class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        # Return fake scores consistent with input count (e.g., all 0.5)
        return [0.5] * len(pairs)

# Replace sentence_transformers.CrossEncoder in sys.modules
# So when build_database.py executes "from sentence_transformers import CrossEncoder", it gets the fake one
import types
fake_sentence_transformers = types.ModuleType("sentence_transformers")
fake_sentence_transformers.CrossEncoder = FakeCrossEncoder
sys.modules["sentence_transformers"] = fake_sentence_transformers

# ==============================
# Now it's safe to set up paths and import other modules
# ==============================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Set Hugging Face offline (double insurance)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ==============================
# Normal import of pytest and other dependencies
# ==============================
import pytest
from raganything import RAGAnythingConfig, RAGAnything
from lightrag.utils import EmbeddingFunc
from unittest.mock import patch, AsyncMock, MagicMock

# ⚠️ Note: Only now import RAGDatabaseManager! sentence_transformers is already mocked
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

    # Mock all external dependencies (note patch path is services.build_database)
    with patch("services.build_database.ollama_complete_async", new_callable=AsyncMock) as mock_llm, \
         patch("services.build_database.ollama_vision_complete_async", new_callable=AsyncMock) as mock_vision, \
         patch("services.build_database.rerank_model_func", None), \
         patch("services.build_database.RAGAnything") as mock_rag_class, \
         patch("services.build_database.AsyncEmbeddingWrapper") as mock_emb_wrapper:

        # Configure mocks
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
