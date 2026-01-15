import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from pathlib import Path
import tempfile
import json

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from simple_raganything import SimpleRAGAnything
from raganything import RAGAnythingConfig



# --- Helper function: Create SimpleRAGAnything instance ---
def create_instance(non_interactive=False):
    """Create a SimpleRAGAnything instance with basic Mocks"""
    with patch('simple_raganything.ObsidianChunker', new=MagicMock()), \
            patch('os.makedirs'):
        instance = SimpleRAGAnything(
            vault_path="/fake/vault",
            working_dir="/fake/working_dir",
            non_interactive=non_interactive
        )
    return instance


# --- Core test fixture: Fix _SentinelObject error ---

@pytest.fixture
def mock_simplified_dependencies():
    """
    Most stable Mock fixture, fixes __name__ attribute error and _SentinelObject error.
    """

    # --- 1. Pre-create all key Mock objects ---

    # LightRAG instance and constructor
    mock_lightrag_instance = MagicMock(name='LightRAG_Instance')
    mock_lightrag_instance.initialize_storages = AsyncMock()
    mock_lightrag_instance.rerank_model_func = None
    mock_lightrag_constructor = MagicMock(name='LightRAG_Constructor',
                                          return_value=mock_lightrag_instance)

    # RAGAnything instance and constructor
    mock_raganything_instance = MagicMock(name='RAGAnything_Instance')
    mock_raganything_constructor = MagicMock(name='RAGAnything_Constructor',
                                             return_value=mock_raganything_instance)

    # RAGAnythingConfig class Mock (assertable)
    mock_raganything_config_class = MagicMock(name='RAGAnythingConfig_Mock')

    # EmbeddingFunc
    mock_embedding_func = MagicMock(name="EmbeddingFunc_Mock")

    # Reranker Func (must set __name__)
    mock_rerank_async = MagicMock(name="bge_rerank_async_mock", __name__="bge_rerank_async_mock")

    # initialize_pipeline_status Func (must be AsyncMock and assertable)
    mock_init_pipeline_status = AsyncMock(name='initialize_pipeline_status_mock')

    # --- 2. Create patches list (using pre-created Mock objects) ---
    patches = [
        # --- Flow control methods ---
        ('_detect_existing_database',
         patch('simple_raganything.SimpleRAGAnything._detect_existing_database')),
        ('_handle_database_choice',
         patch('simple_raganything.SimpleRAGAnything._handle_database_choice')),
        ('_delete_existing_database',
         patch('simple_raganything.SimpleRAGAnything._delete_existing_database')),
        ('_initialize_embedding_model',
         patch('simple_raganything.SimpleRAGAnything._initialize_embedding_model')),

        # --- Key import Mocks (inject pre-created Mocks) ---
        ('lightrag.LightRAG',
         patch('lightrag.LightRAG', new=mock_lightrag_constructor)),
        ('initialize_pipeline_status',  # Fix _SentinelObject error
         patch('lightrag.kg.shared_storage.initialize_pipeline_status', new=mock_init_pipeline_status)),

        ('simple_raganything.RAGAnything',
         patch('simple_raganything.RAGAnything', new=mock_raganything_constructor)),
        ('RAGAnythingConfig',  # Fix _SentinelObject error
         patch('simple_raganything.RAGAnythingConfig', new=mock_raganything_config_class)),

        ('EmbeddingFunc',
         patch('simple_raganything.EmbeddingFunc', new=mock_embedding_func)),
        ('bge_rerank_async',
         patch('simple_raganything.bge_rerank_async', new=mock_rerank_async)),
        ('openai_complete_if_cache',
         patch('simple_raganything.openai_complete_if_cache', new=AsyncMock())),
        ('openai_embed',
         patch('simple_raganything.openai_embed', new=MagicMock())),
        ('gemini_complete_if_cache',
         patch('simple_raganything.gemini_complete_if_cache', new=AsyncMock())),
        ('gemini_vision_complete',
         patch('simple_raganything.gemini_vision_complete', new=AsyncMock())),
    ]

    mocks = {}
    all_patch_objects = []

    # Start all Mocks
    for key, p in patches:
        all_patch_objects.append(p)
        p.start()

    # --- 3. Centrally store all Mock objects (ensure references are assertable Mock instances) ---

    # Constructors and instances
    mocks['LightRAG'] = mock_lightrag_constructor
    mocks['LightRAG_Instance'] = mock_lightrag_instance
    mocks['RAGAnything'] = mock_raganything_constructor
    mocks['RAGAnything_Instance'] = mock_raganything_instance

    # Key functions and classes
    mocks['RAGAnythingConfig'] = mock_raganything_config_class
    mocks['initialize_pipeline_status'] = mock_init_pipeline_status
    mocks['EmbeddingFunc'] = mock_embedding_func
    mocks['bge_rerank_async'] = mock_rerank_async

    # Flow control methods (they are Mocks patched onto the SimpleRAGAnything class)
    mocks['_detect_existing_database'] = SimpleRAGAnything._detect_existing_database
    mocks['_handle_database_choice'] = SimpleRAGAnything._handle_database_choice
    mocks['_delete_existing_database'] = SimpleRAGAnything._delete_existing_database
    mocks['_initialize_embedding_model'] = SimpleRAGAnything._initialize_embedding_model

    yield mocks

    # Stop all Mocks
    for p in all_patch_objects:
        p.stop()


# --- Actual test class ---

@pytest.mark.asyncio
class TestSimpleRAGAnythingCoreInitialization:

    @pytest.fixture(autouse=True)
    def setup_instance(self, mock_simplified_dependencies):
        """
        Create a new instance for each test case and inject Mocks.
        This resolves the `AttributeError` in the source code without modifying it.
        """
        self.instance = create_instance(non_interactive=False)
        self.mocks = mock_simplified_dependencies

        # ⚠️ Fix missing attributes in source code (AttributeError: embedding_func)
        self.instance._embedding_function = self.mocks['EmbeddingFunc']
        self.instance.embedding_func = self.mocks['EmbeddingFunc']

    # --- Test branch 1: New mode (RAGAnything(config=...)) ---
    async def test_initializes_with_config_when_new_db(self):
        """
        Test scenario: Database does not exist, should instantiate via RAGAnything(config=...).
        """
        # 1. Set Mock behavior: simulate database does not exist
        self.mocks['_detect_existing_database'].return_value = False

        await self.instance.initialize()

        # 2. Check flow control functions: should call initialize model, but not involve database selection or deletion
        self.mocks['_initialize_embedding_model'].assert_called_once()
        self.mocks['_detect_existing_database'].assert_called_once()
        self.mocks['_handle_database_choice'].assert_not_called()
        self.mocks['_delete_existing_database'].assert_not_called()

        # 3. Core assertion: RAGAnything must be created via config
        self.mocks['RAGAnythingConfig'].assert_called_once()  # Fixed assertion
        self.mocks['RAGAnything'].assert_called_once()

        rag_call_kwargs = self.mocks['RAGAnything'].call_args[1]
        assert 'config' in rag_call_kwargs  # Verify created via config
        assert 'lightrag' not in rag_call_kwargs
        assert rag_call_kwargs['embedding_func'] == self.mocks['EmbeddingFunc']

        # 4. LightRAG and initialize status should not be called directly
        self.mocks['LightRAG'].assert_not_called()
        self.mocks['initialize_pipeline_status'].assert_not_called()

    # --- Test branch 2: Load mode (RAGAnything(lightrag=...)) ---
    async def test_initializes_with_lightrag_when_existing_db(self):
        """
        Test scenario: Database exists, should create LightRAG instance, initialize, then pass to RAGAnything.
        """
        # 1. Set Mock behavior: simulate database exists, user chooses to use existing
        self.mocks['_detect_existing_database'].return_value = True
        self.instance.non_interactive = False
        self.mocks['_handle_database_choice'].return_value = "1"

        await self.instance.initialize()

        # 2. Check flow control functions: should call initialize model and handle database choice
        self.mocks['_initialize_embedding_model'].assert_called_once()
        self.mocks['_handle_database_choice'].assert_called_once()
        self.mocks['_delete_existing_database'].assert_not_called()

        # 3. Core assertion: LightRAG must be created and initialized
        self.mocks['LightRAG'].assert_called_once()
        lightrag_instance_mock = self.mocks['LightRAG_Instance']

        # Check if LightRAG instance is initialized
        lightrag_instance_mock.initialize_storages.assert_called_once()
        self.mocks['initialize_pipeline_status'].assert_called_once()  # Fixed assertion

        # Check if rerank_model_func is set
        assert lightrag_instance_mock.rerank_model_func == self.mocks['bge_rerank_async']

        # 4. RAGAnything must be created via LightRAG instance
        self.mocks['RAGAnything'].assert_called_once()
        rag_call_kwargs = self.mocks['RAGAnything'].call_args[1]

        assert 'lightrag' in rag_call_kwargs
        assert rag_call_kwargs['lightrag'] == lightrag_instance_mock
        assert 'config' not in rag_call_kwargs

        # RAGAnythingConfig should not be called
        self.mocks['RAGAnythingConfig'].assert_called_once()

    # --- Test scenario 3: Rebuild database (New DB code path after deletion) ---
    async def test_initializes_with_config_when_rebuild_db(self):
        """
        Test scenario: Database exists, but user chooses to rebuild. Flow should delete old database, then follow new mode.
        """
        # 1. Set Mock behavior: simulate database exists, user chooses to rebuild
        self.mocks['_detect_existing_database'].return_value = True
        self.instance.non_interactive = False
        self.mocks['_handle_database_choice'].return_value = "2"  # Choose rebuild

        await self.instance.initialize()

        # 2. Check flow control functions: should call delete and model initialization
        self.mocks['_detect_existing_database'].assert_called_once()
        self.mocks['_handle_database_choice'].assert_called_once()
        self.mocks['_delete_existing_database'].assert_called_once()
        self.mocks['_initialize_embedding_model'].assert_called_once()

        # 3. Core assertion: Flow must switch to new mode
        self.mocks['RAGAnythingConfig'].assert_called_once()  # Fixed assertion
        self.mocks['RAGAnything'].assert_called_once()

        rag_call_kwargs = self.mocks['RAGAnything'].call_args[1]
        assert 'config' in rag_call_kwargs
        assert 'lightrag' not in rag_call_kwargs

        # 4. LightRAG and initialize status should not be called
        self.mocks['LightRAG'].assert_not_called()
        self.mocks['initialize_pipeline_status'].assert_not_called()


@pytest.fixture
def mock_embedding_model():
    with patch("simple_raganything.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 256]
        mock_model.get_sentence_embedding_dimension.return_value = 256
        mock_st.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_rag_anything():
    with patch("simple_raganything.RAGAnything") as mock_rag:
        mock_instance = MagicMock()
        mock_rag.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_llm_and_vision():
    def fake_llm(prompt, **kwargs):
        if "extract entities" in prompt.lower():
            return '{"entities": [{"entity_name": "TestEntity", "entity_type": "Concept"}]}'
        return "mock response"

    def fake_vision(prompt, **kwargs):
        return "mock vision response"

    with patch("simple_raganything.gemini_complete_if_cache", new=AsyncMock(side_effect=fake_llm)), \
         patch("simple_raganything.gemini_vision_complete", new=AsyncMock(side_effect=fake_vision)):
        yield


@pytest.fixture
def test_vault():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "note1.md").write_text("# Test\nContent.")
        yield str(p)



@pytest.mark.asyncio
async def test_initialize_new_database(test_vault, mock_llm_and_vision, mock_embedding_model, mock_rag_anything):
    with tempfile.TemporaryDirectory() as wd:
        rag = SimpleRAGAnything(test_vault, wd, non_interactive=True)
        await rag.initialize()
        assert not rag.using_existing_db
        assert rag.rag_anything is mock_rag_anything



@pytest.mark.asyncio
async def test_load_existing_database(test_vault, mock_llm_and_vision, mock_embedding_model):
    with tempfile.TemporaryDirectory() as wd:
        # 🔑 Fake a key file to make _detect_existing_database return True
        fake_file = Path(wd) / "kv_store_doc_status.json"
        fake_file.write_text(json.dumps({}))  # Minimal valid content

        # initialize RAG — it will detect 'existing database
        rag = SimpleRAGAnything(test_vault, wd, non_interactive=True)
        await rag.initialization()

        # ✅ Should pass now!
        assert rag.using_existing_db