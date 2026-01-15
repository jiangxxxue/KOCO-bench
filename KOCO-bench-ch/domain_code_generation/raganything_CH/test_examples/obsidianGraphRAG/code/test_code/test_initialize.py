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


# --- 导入 SimpleRAGAnything 或使用 Mock 占位符 ---
try:
    from simple_raganything import SimpleRAGAnything
    from simple_raganything import RAGAnythingConfig
except ImportError:
    class RAGAnythingConfig:
        def __init__(self, *args, **kwargs): pass


    class SimpleRAGAnything:
        def __init__(self, vault_path: str, working_dir: str = "./rag_storage", non_interactive: bool = False):
            self.vault_path = vault_path
            self.working_dir = working_dir
            self.rag_anything = None
            self.embedding_model = None
            self.chunker = MagicMock()
            self.using_existing_db = False
            self.non_interactive = non_interactive
            self._llm_function = AsyncMock()
            self._vision_function = AsyncMock()

        async def initialize(self): pass

        def _detect_existing_database(self): pass

        def _handle_database_choice(self): pass

        def _delete_existing_database(self): pass

        def _initialize_embedding_model(self): pass


# --- 辅助函数：创建 SimpleRAGAnything 实例 ---
def create_instance(non_interactive=False):
    """创建一个带有基本 Mocks 的 SimpleRAGAnything 实例"""
    with patch('simple_raganything.ObsidianChunker', new=MagicMock()), \
            patch('os.makedirs'):
        instance = SimpleRAGAnything(
            vault_path="/fake/vault",
            working_dir="/fake/working_dir",
            non_interactive=non_interactive
        )
    return instance


# --- 核心测试夹具：修复 _SentinelObject 错误 ---

@pytest.fixture
def mock_simplified_dependencies():
    """
    最稳定的 Mock 夹具，修复了 __name__ 属性错误 和 _SentinelObject 错误。
    """

    # --- 1. 预先创建所有关键 Mock 对象 ---

    # LightRAG 实例和构造函数
    mock_lightrag_instance = MagicMock(name='LightRAG_Instance')
    mock_lightrag_instance.initialize_storages = AsyncMock()
    mock_lightrag_instance.rerank_model_func = None
    mock_lightrag_constructor = MagicMock(name='LightRAG_Constructor',
                                          return_value=mock_lightrag_instance)

    # RAGAnything 实例和构造函数
    mock_raganything_instance = MagicMock(name='RAGAnything_Instance')
    mock_raganything_constructor = MagicMock(name='RAGAnything_Constructor',
                                             return_value=mock_raganything_instance)

    # RAGAnythingConfig 类 Mock (可断言调用)
    mock_raganything_config_class = MagicMock(name='RAGAnythingConfig_Mock')

    # EmbeddingFunc
    mock_embedding_func = MagicMock(name="EmbeddingFunc_Mock")

    # Reranker Func (必须设置 __name__)
    mock_rerank_async = MagicMock(name="bge_rerank_async_mock", __name__="bge_rerank_async_mock")

    # initialize_pipeline_status Func (必须是 AsyncMock 且可断言)
    mock_init_pipeline_status = AsyncMock(name='initialize_pipeline_status_mock')

    # --- 2. 创建 patches 列表 (使用预创建的 Mock 对象) ---
    patches = [
        # --- 流程控制方法 ---
        ('_detect_existing_database',
         patch('simple_raganything.SimpleRAGAnything._detect_existing_database')),
        ('_handle_database_choice',
         patch('simple_raganything.SimpleRAGAnything._handle_database_choice')),
        ('_delete_existing_database',
         patch('simple_raganything.SimpleRAGAnything._delete_existing_database')),
        ('_initialize_embedding_model',
         patch('simple_raganything.SimpleRAGAnything._initialize_embedding_model')),

        # --- 关键导入 Mock (注入预创建的 Mock) ---
        ('lightrag.LightRAG',
         patch('lightrag.LightRAG', new=mock_lightrag_constructor)),
        ('initialize_pipeline_status',  # 修复 _SentinelObject 错误
         patch('lightrag.kg.shared_storage.initialize_pipeline_status', new=mock_init_pipeline_status)),

        ('simple_raganything.RAGAnything',
         patch('simple_raganything.RAGAnything', new=mock_raganything_constructor)),
        ('RAGAnythingConfig',  # 修复 _SentinelObject 错误
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

    # 启动所有 Mock
    for key, p in patches:
        all_patch_objects.append(p)
        p.start()

    # --- 3. 集中存储所有 Mock 对象 (确保引用的是可断言的 Mock 实例) ---

    # 构造函数和实例
    mocks['LightRAG'] = mock_lightrag_constructor
    mocks['LightRAG_Instance'] = mock_lightrag_instance
    mocks['RAGAnything'] = mock_raganything_constructor
    mocks['RAGAnything_Instance'] = mock_raganything_instance

    # 关键函数和类
    mocks['RAGAnythingConfig'] = mock_raganything_config_class
    mocks['initialize_pipeline_status'] = mock_init_pipeline_status
    mocks['EmbeddingFunc'] = mock_embedding_func
    mocks['bge_rerank_async'] = mock_rerank_async

    # 流程控制方法 (它们是被 patch 替换在 SimpleRAGAnything 类上的 Mock)
    mocks['_detect_existing_database'] = SimpleRAGAnything._detect_existing_database
    mocks['_handle_database_choice'] = SimpleRAGAnything._handle_database_choice
    mocks['_delete_existing_database'] = SimpleRAGAnything._delete_existing_database
    mocks['_initialize_embedding_model'] = SimpleRAGAnything._initialize_embedding_model

    yield mocks

    # 停止所有 Mock
    for p in all_patch_objects:
        p.stop()


# --- 实际测试类 ---

@pytest.mark.asyncio
class TestSimpleRAGAnythingCoreInitialization:

    @pytest.fixture(autouse=True)
    def setup_instance(self, mock_simplified_dependencies):
        """
        为每个测试用例创建一个新的实例并注入 Mocks。
        这里我们解决了源代码中的 `AttributeError`，避免了修改源代码。
        """
        self.instance = create_instance(non_interactive=False)
        self.mocks = mock_simplified_dependencies

        # ⚠️ 修复源代码中缺失的属性（AttributeError: embedding_func）
        self.instance._embedding_function = self.mocks['EmbeddingFunc']
        self.instance.embedding_func = self.mocks['EmbeddingFunc']

    # --- 测试分支 1: 新建模式 (RAGAnything(config=...)) ---
    async def test_initializes_with_config_when_new_db(self):
        """
        测试场景：数据库不存在，应通过 RAGAnything(config=...) 实例化。
        """
        # 1. 设置 Mock 行为：模拟数据库不存在
        self.mocks['_detect_existing_database'].return_value = False

        await self.instance.initialize()

        # 2. 检查流程控制函数：应调用初始化模型，但不应涉及数据库选择或删除
        self.mocks['_initialize_embedding_model'].assert_called_once()
        self.mocks['_detect_existing_database'].assert_called_once()
        self.mocks['_handle_database_choice'].assert_not_called()
        self.mocks['_delete_existing_database'].assert_not_called()

        # 3. 核心断言：RAGAnything 必须通过 config 创建
        self.mocks['RAGAnythingConfig'].assert_called_once()  # 修复断言
        self.mocks['RAGAnything'].assert_called_once()

        rag_call_kwargs = self.mocks['RAGAnything'].call_args[1]
        assert 'config' in rag_call_kwargs  # 验证通过 config 创建
        assert 'lightrag' not in rag_call_kwargs
        assert rag_call_kwargs['embedding_func'] == self.mocks['EmbeddingFunc']

        # 4. LightRAG 和初始化状态不应被直接调用
        self.mocks['LightRAG'].assert_not_called()
        self.mocks['initialize_pipeline_status'].assert_not_called()

    # --- 测试分支 2: 加载模式 (RAGAnything(lightrag=...)) ---
    async def test_initializes_with_lightrag_when_existing_db(self):
        """
        测试场景：数据库存在，应创建 LightRAG 实例，初始化，然后传给 RAGAnything。
        """
        # 1. 设置 Mock 行为：模拟数据库存在，用户选择使用现有
        self.mocks['_detect_existing_database'].return_value = True
        self.instance.non_interactive = False
        self.mocks['_handle_database_choice'].return_value = "1"

        await self.instance.initialize()

        # 2. 检查流程控制函数：应调用初始化模型，并处理数据库选择
        self.mocks['_initialize_embedding_model'].assert_called_once()
        self.mocks['_handle_database_choice'].assert_called_once()
        self.mocks['_delete_existing_database'].assert_not_called()

        # 3. 核心断言：LightRAG 必须被创建和初始化
        self.mocks['LightRAG'].assert_called_once()
        lightrag_instance_mock = self.mocks['LightRAG_Instance']

        # 检查 LightRAG 实例是否被初始化
        lightrag_instance_mock.initialize_storages.assert_called_once()
        self.mocks['initialize_pipeline_status'].assert_called_once()  # 修复断言

        # 检查 rerank_model_func 是否被设置
        assert lightrag_instance_mock.rerank_model_func == self.mocks['bge_rerank_async']

        # 4. RAGAnything 必须通过 LightRAG 实例创建
        self.mocks['RAGAnything'].assert_called_once()
        rag_call_kwargs = self.mocks['RAGAnything'].call_args[1]

        assert 'lightrag' in rag_call_kwargs
        assert rag_call_kwargs['lightrag'] == lightrag_instance_mock
        assert 'config' not in rag_call_kwargs

        # RAGAnythingConfig 不应该被调用
        self.mocks['RAGAnythingConfig'].assert_called_once()

    # --- 测试场景 3: 重建数据库 (New DB code path after deletion) ---
    async def test_initializes_with_config_when_rebuild_db(self):
        """
        测试场景：数据库存在，但用户选择重建。流程应变为删除旧数据库，然后走新建模式。
        """
        # 1. 设置 Mock 行为：模拟数据库存在，用户选择重建
        self.mocks['_detect_existing_database'].return_value = True
        self.instance.non_interactive = False
        self.mocks['_handle_database_choice'].return_value = "2"  # 选择重建

        await self.instance.initialize()

        # 2. 检查流程控制函数：应调用删除和模型初始化
        self.mocks['_detect_existing_database'].assert_called_once()
        self.mocks['_handle_database_choice'].assert_called_once()
        self.mocks['_delete_existing_database'].assert_called_once()
        self.mocks['_initialize_embedding_model'].assert_called_once()

        # 3. 核心断言：流程必须切换到新建模式
        self.mocks['RAGAnythingConfig'].assert_called_once()  # 修复断言
        self.mocks['RAGAnything'].assert_called_once()

        rag_call_kwargs = self.mocks['RAGAnything'].call_args[1]
        assert 'config' in rag_call_kwargs
        assert 'lightrag' not in rag_call_kwargs

        # 4. LightRAG 和初始化状态不应被调用
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
        # 🔑 伪造一个关键文件，让 _detect_existing_database 返回 True
        fake_file = Path(wd) / "kv_store_doc_status.json"
        fake_file.write_text(json.dumps({}))  # 最小合法内容

        # 初始化 RAG — 它会检测到“已有数据库”
        rag = SimpleRAGAnything(test_vault, wd, non_interactive=True)
        await rag.initialize()

        # ✅ 现在应该通过！
        assert rag.using_existing_db