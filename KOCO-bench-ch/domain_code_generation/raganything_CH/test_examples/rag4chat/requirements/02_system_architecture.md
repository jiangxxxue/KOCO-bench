# rag4chat: 系统架构与模块设计

## 1. rag4chat 模块文件结构

### 1.1 文件组织

项目源代码主要位于 `code/` 目录下，并围绕 `services`, `utils`, `models` 等核心目录进行功能划分。

```
code/
├── app.py                 # Streamlit Web UI 应用主入口
├── app_api.py             # Flask 后端 API 服务主入口
├── services/              # 核心业务服务
│   ├── build_database.py  # 知识库构建与管理的核心引擎
│   ├── vector_store.py    # 向量存储服务的上层封装
│   ├── get_top_from_rag.py# 从向量数据库中执行检索的底层实现
│   └── weather_tools.py   # 天气查询工具的实现
├── utils/                 # 通用工具模块
│   ├── document_processor.py # 文档解析与切分工具
│   ├── chat_history.py    # 会话历史管理器
│   └── ui_components.py   # Streamlit UI 组件封装
├── models/                # 智能体与模型定义
│   └── agent.py           # 定义与LLM交互的RAGAgent
└── config/                # 配置文件
    └── settings.py        # 存放模型名称、API Key等全局配置
```

### 1.2 模块职责划分

- **app.py / app_api.py**: 系统的两个主要入口。`app.py` 负责提供一个完整的、可交互的Streamlit图形界面；`app_api.py` 则提供了一个无头（Headless）的Flask API服务，便于第三方系统集成。
- **services/build_database.py**: 定义了 `RAGDatabaseManager`，是系统的数据处理核心。它调用 `RAGAnything` 库来完成文档的解析、向量化和索引构建。
- **services/vector_store.py**: 定义了 `VectorStoreService`，作为应用层与数据底层之间的“中间人”。它为上层应用提供了简洁的接口（如 `search_documents`），并将具体的数据库操作委托给 `RAGDatabaseManager` 和 `get_top_from_rag`。
- **models/agent.py**: 定义了 `RAGAgent`，是与大语言模型（LLM）沟通的“代理”。它负责根据检索到的上下文，构建最终的提示词（Prompt），并从LLM获取答案。
- **utils/document_processor.py**: 定义了 `DocumentProcessor`，负责在知识库构建的初期，对上传的各类文档（PDF, DOCX等）进行文本提取和切块。

## 2. rag4chat 模块设计

### 2.1 应用层 (app.py & app_api.py)

- **核心功能**: 作为系统的两个门户，分别满足UI交互和API集成的需求。
- **`app.py` (Streamlit)**: 编排所有后端服务，管理UI状态（如“思考中”），渲染聊天界面，并通过 `VectorStoreService` 和 `RAGAgent` 完成一次完整的RAG流程。
- **`app_api.py` (Flask)**: 遵循OpenAI的API格式，提供了 `/v1/chat/completions` 接口。它接收JSON格式的请求，调用核心逻辑，并能以流式（Streaming）或非流式的方式返回结果。

### 2.2 知识库构建引擎 (services/build_database.py)

- **核心类**: `RAGDatabaseManager`
- **核心功能**: 封装了 `RAGAnything` 库的复杂性，是构建知识库的执行引擎。
- **主要方法**: 
    - `_create_rag_instance`: 负责配置和实例化 `RAGAnything` 对象，包括设置LLM（Ollama）、嵌入模型（阿里云百炼）和视觉模型（Llava）的调用函数。
    - `add_document`: 接收单个文件路径，并调用 `RAGAnything` 的 `process_document_complete` 方法来完成该文件的完整处理和索引流程。

### 2.3 检索服务 (services/vector_store.py)

- **核心类**: `VectorStoreService`
- **核心功能**: 为上层应用提供一个简洁、稳定的数据检索接口，将底层的具体实现（无论是Faiss, Chroma还是其他）隔离开来。
- **主要方法**: 
    - `create_vector_store`: 调用 `RAGDatabaseManager` 来从一批文档中构建或重建整个向量数据库。
    - `search_documents`: 接收用户查询，调用 `get_top_from_rag.py` 中的 `query_and_find_topk` 函数，在 `vdb_chunks.json` 文件中执行向量相似度搜索，并返回最相关的文档片段。

### 2.4 LLM交互代理 (models/agent.py)

- **核心类**: `RAGAgent`
- **核心功能**: 作为与LLM沟通的最后一道关卡，负责构建提示词和解析LLM的响应。
- **实现细节**: 
    - 它使用了 `agno` 库来创建一个Agent，并为其配置了Ollama驱动的 `qwen3:8b` 模型。
    - 它定义了 `query_weather` 等工具，展示了系统的函数调用（Function Calling）能力。
    - 其 `run` 方法是核心，它会根据有无 `context`（从检索服务获取的上下文）来动态构建不同的提示词模板，确保LLM在回答时能充分利用检索到的信息。

## 3. 数据协议与模块交互

### 3.1 API 数据协议

Flask API (`app_api.py`) 接收的请求体遵循了类似OpenAI的结构，核心字段包括：

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "isRAG": true,       // 是否启用RAG
  "similarity": 0.5,   // 相似度阈值
  "stream": false      // 是否流式返回
}
```

### 3.2 知识库构建流程

1.  用户在 `app.py` 的UI界面上传一个或多个文档。
2.  `UIComponents` 将文件传递给 `DocumentProcessor` 进行初步处理。
3.  `VectorStoreService` 的 `create_vector_store` 方法被调用。
4.  `VectorStoreService` 委托 `RAGDatabaseManager` 的 `add_documents` 方法。
5.  `RAGDatabaseManager` 为每个文档调用 `RAGAnything` 的 `process_document_complete`，后者完成**解析、切块、调用嵌入模型、生成向量、写入本地JSON文件**的全过程。

### 3.3 RAG查询流程

1.  用户在UI或通过API提交一个问题。
2.  `app.py` 或 `app_api.py` 接收问题，并调用 `VectorStoreService.search_documents`。
3.  `VectorStoreService` 调用 `get_top_from_rag.query_and_find_topk`，后者读取 `vdb_chunks.json`，计算并返回最相关的文档块（chunks）。
4.  `app.py` 将这些文档块格式化为 `context` 字符串，并连同原始问题一起传递给 `RAGAgent.run` 方法。
5.  `RAGAgent` 构建最终的提示词，并从Ollama获取答案。
6.  答案被返回给前端并显示。

### 3.4 模块交互图

```mermaid
graph TD
    subgraph Entrypoints
        A[UI - app.py] --> C{VectorStoreService};
        B[API - app_api.py] --> G{agent_response};
    end

    subgraph Knowledge Base Pipeline
        C -- create_vector_store --> D[RAGDatabaseManager];
        D -- uses --> E[RAGAnything Library];
        E -- uses --> F[Embedding Model];
    end

    subgraph Query Pipeline
        G -- calls --> C;
        C -- search_documents --> H[query_and_find_topk];
        H -- reads --> I[vdb_chunks.json];
        C -- returns context --> G;
        G -- runs --> J[RAGAgent];
        J -- calls --> K[LLM (Ollama)];
        K -- returns answer --> G;
        G -- returns to --> B;
    end
```
