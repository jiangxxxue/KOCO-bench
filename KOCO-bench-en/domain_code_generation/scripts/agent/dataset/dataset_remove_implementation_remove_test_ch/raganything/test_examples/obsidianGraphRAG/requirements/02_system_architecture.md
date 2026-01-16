# ObsidianGraphRAG 系统架构与模块设计

## 1. ObsidianGraphRAG 模块文件结构

### 1.1 文件组织

项目采用清晰的功能分区结构，核心逻辑、UI 和执行脚本分离，便于维护和扩展。

```
obsidianGraphRAG/
├── src/                   # 核心系统逻辑
│   ├── simple_raganything.py    # RAG主编排类，负责初始化、处理和查询
│   ├── obsidian_chunker.py      # Obsidian 专用笔记切分器
│   ├── vault_monitor.py         # 笔记库变化监视与增量同步逻辑
│   ├── gemini_llm.py            # 对接 Google Gemini API 的封装
│   └── bge_reranker.py          # 对接 BGE Reranker 模型的封装
├── ui/                    # Web 用户界面
│   ├── app.py                   # FastAPI 后端，提供WebSocket聊天和API
│   └── static/                  # HTML/CSS/JS 前端静态文件
├── run_obsidian_raganything.py  # 首次构建知识库的执行脚本
├── run_incremental_sync.py      # 执行增量同步的脚本
├── run_ui.py                  # 启动 Web UI 的脚本
└── requirements.txt           # Python 依赖
```

### 1.2 模块职责划分

- **simple_raganything.py**: 系统的“大脑”，负责初始化和编排所有核心组件。它管理着数据库的生命周期（新建或加载），调用切分器处理文档，并执行查询逻辑。
- **obsidian_chunker.py**: 专为 Obsidian 优化的“预处理器”。它不仅将笔记切分成适合模型处理的小块，更重要的是，它会在每个文本块中注入丰富的元数据（如双向链接、标签、文件关系），最大限度地保留了知识的上下文。
- **vault_monitor.py**: “增量更新”功能的核心。它通过比对文件的哈希值来高效地检测出笔记库中的新增、修改和删除，并调用 RAG 引擎执行相应的更新，避免了每次都全量重建索引。
- **gemini_llm.py / bge_reranker.py**: 模型的“连接器”。它们将外部的 Gemini API 和本地的 BGE Reranker 模型封装成符合 `LightRAG` 框架要求的标准接口，实现了模型的即插即用。
- **ui/app.py**: 用户界面的“服务端”。它通过 FastAPI 构建，一方面在应用启动时加载 RAG 系统，另一方面通过 WebSocket 为前端提供实时的流式聊天服务，并通过 REST API 提供系统状态查询和手动同步等管理功能。

## 2. ObsidianGraphRAG 模块设计

### 2.1 RAG 编排器 (SimpleRAGAnything)

- **核心类**: `SimpleRAGAnything`
- **核心功能**: 
    1.  **初始化 (`initialize`)**: 检查是否存在旧的知识库，并根据用户的选择（或在Web模式下自动）决定是加载旧库还是创建新库。它负责加载嵌入模型、配置RAG引擎和重排器。
    2.  **全量处理 (`process_vault`)**: 调用 `ObsidianChunker` 处理整个笔记库，然后将切分好的、富含元数据的文本块批量送入 `RAGAnything` 引擎进行索引。
    3.  **查询 (`query`)**: 接收问题，调用 `RAGAnything` 的查询接口，并确保重排器被激活，最后返回LLM生成的答案。

### 2.2 Obsidian 切分器 (ObsidianChunker)

- **核心类**: `ObsidianChunker`
- **核心功能**: 将 `.md` 文件切分成约2000个token的文本块。其先进之处在于，它并非简单地切割文本，而是通过 `_enhance_chunk_with_metadata` 方法，在每个文本块的头部添加了结构化的元数据，如：
    - **文件信息**: 笔记名称、路径、创建/修改日期。
    - **知识连接**: 显式地列出该文本块中包含的所有 `[[Wikilinks]]` 和 `#Tags`。
    - **文档结构**: 标注该文本块是文件中的第几块、总共有几块，以及它的上一个和下一个文本块的ID。

### 2.3 增量同步模块 (VaultMonitor)

- **核心类**: `VaultMonitor`, `IncrementalVaultUpdater`
- **核心功能**: 
    - `VaultMonitor`: 负责“检测”。它维护一个 `vault_tracking.json` 文件，记录着每个已处理文件的哈希值。每次扫描时，它会与当前文件系统的状态进行比对，从而快速找出“新、改、删”三种变化。
    - `IncrementalVaultUpdater`: 负责“执行”。它接收 `VaultMonitor` 的检测结果，并对 `RAGAnything` 数据库执行具体的操作：为新文件调用 `insert_content_list`，为修改过的文件先调用 `adelete_by_doc_id` 再 `insert_content_list`，为已删除的文件调用 `adelete_by_doc_id`。

### 2.4 Web UI 后端 (ui/app.py)

- **核心框架**: FastAPI
- **核心功能**: 
    - **启动加载**: 在服务器启动时，创建一个全局的 `SimpleRAGAnything` 实例并完成初始化，确保系统随时可用。
    - **实时聊天**: 通过 `/ws/chat` WebSocket 端点，接收用户问题，调用RAG系统的 `query` 方法，并将LLM返回的答案以流式（逐词）的方式实时推送给前端，提供了优秀的交互体验。
    - **系统管理**: 通过 `/api/sync` 等REST API端点，允许用户从前端界面手动触发增量同步，或查询当前系统的配置和状态。

## 3. 数据协议与模块交互

### 3.1 核心数据结构：增强型文本块 (Enhanced Chunk)

系统中最核心的数据单元是由 `ObsidianChunker` 生成的“增强型文本块”。它是一个包含了丰富元数据和原始文本的字典结构，确保了即使文本被切分，其上下文和知识关联也能被完整保留并送入RAG系统。

### 3.2 核心流程

- **索引流程**: `run_obsidian_raganything.py` -> `SimpleRAGAnything.process_vault` -> `ObsidianChunker` -> `RAGAnything.insert_content_list` -> 本地存储。
- **查询流程**: `UI` -> `WebSocket` -> `SimpleRAGAnything.query` -> `RAGAnything.aquery` -> 检索 -> `BGE Reranker` -> `Gemini LLM` -> `UI`。
- **同步流程**: `UI Sync Button` -> `/api/sync` -> `VaultMonitor.scan_vault` -> `IncrementalVaultUpdater.sync_vault` -> RAG数据库更新。

### 3.3 数据流

- **输入数据**: 用户本地文件系统中的 `.md` 文件。
- **中间数据**: 由 `ObsidianChunker` 生成的、富含元数据的文本块（在内存中处理）。
- **持久化数据**: 所有知识都存储在 `rag_storage/` 目录下，主要包括：
    - `LightRAG` 的数据库文件（如图数据库GraphML，向量数据库JSON等）。
    - `vault_tracking.json`：用于增量同步的文件状态跟踪文件。
- **输出数据**: 通过WebSocket传输的流式文本，最终在前端UI上聚合成完整的答案。

### 3.4 模块交互图

```mermaid
graph TD
    subgraph User Interaction
        A[User] <--> B[Web UI (static/index.html)]
    end

    subgraph Backend
        B <-->|WebSocket| C[FastAPI App (ui/app.py)]
        C -- query --> D[SimpleRAGAnything]
        C -- sync --> E[IncrementalVaultUpdater]
        E -- scans with --> F[VaultMonitor]
        D -- uses --> G[RAGAnything Library]
        G -- gets context from --> H[Knowledge DB (rag_storage/)]
        G -- reranks with --> I[BGE Reranker]
        G -- generates with --> J[Gemini LLM]
    end

    subgraph Initial Build
        K[run_obsidian_raganything.py] --> D
        D -- chunks with --> L[ObsidianChunker]
        D -- builds --> H
    end
```
