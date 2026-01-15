"""
配置文件，包含所有常量和配置项
"""

# 1. 文件路径
VECTOR_STORE_PATH = "faiss_index"
HISTORY_FILE = "chat_history.json"

# 2. 模型配置
DEFAULT_MODEL = "qwen3:8b"
AVAILABLE_MODELS = ["qwen2.5:7b", "deepseek-r1:7b", "qwen3:8b", "llava:latest"]

EMBEDDING_MODEL = "text-embedding-v4"
AVAILABLE_EMBEDDING_MODELS = ["text-embedding-v4", "bge-m3:latest"]
EMBEDDING_BASE_URL = "http://localhost:11434"


# 3. RAG配置
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 30
MAX_RETRIEVED_DOCS = 3


# 4. 高德地图API配置
AMAP_API_KEY = "48257ed7b33d55e349260a9837436968" 

# 4. 数据库配置
DB_PATH = r"D:\adavance\bigmodel\2.原创案例：Agentic RAG智能问答系统Agent\chinook.db"

# 5. LangChain配置
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

# 6. 对话历史配置
MAX_HISTORY_TURNS = 5 