# RAG 测试平台系统架构与模块设计

## 1. RAG 平台模块文件结构

### 1.1 文件组织

系统的核心逻辑位于 `src/` 目录下，采用了清晰的服务化和模块化结构。

```
src/
├── main.py              # FastAPI 应用主入口和API路由
├── models/              # Pydantic 数据模型 (API请求和响应结构)
├── services/            # 核心业务逻辑服务
│   ├── rag_interface.py # RAG 服务的抽象基类接口
│   ├── rag_factory.py   # RAG 服务工厂，用于创建具体的RAG服务实例
│   ├── rag_anything_service.py  # RAGAnything 方案的具体实现
│   ├── rag_vertex_service.py    # Vertex AI RAG 方案的具体实现
│   ├── evidence_sweep_service.py # EvidenceSweep 方案的具体实现
│   └── auth_service.py  # 用户认证与工作空间授权服务
└── utils/               # 工具类
    ├── firebase_utils.py  # Firebase/Firestore 数据库交互
    └── gcs_utils.py       # Google Cloud Storage 文件操作
```

### 1.2 模块职责划分

- **main.py**: 应用的门户。负责定义所有API端点，处理HTTP请求，通过依赖注入调用相应服务，并管理后台任务。
- **services/rag_interface.py**: 定义了所有RAG实现都必须遵守的“合同”（接口），确保了不同RAG方案可以被统一调用。
- **services/rag_factory.py**: 实现了工厂设计模式。根据API请求中指定的 `rag_approach` 参数，动态地创建并返回一个具体的RAG服务实例。
- **services/*_service.py**: 每种RAG方案的具体实现。它们封装了各自所需的数据处理、索引构建和查询逻辑。
- **services/auth_service.py**: 负责用户身份验证和权限检查，确保用户只能访问其有权访问的工作空间和项目。
- **utils/**: 提供了与外部服务（Firebase 和 GCS）交互的底层封装，使核心业务逻辑更清晰。

## 2. RAG 平台模块设计

### 2.1 FastAPI 应用 (main.py)

- **核心功能**: 作为系统的总控制器，负责API路由、依赖注入、后台任务管理和监控指标的暴露。
- **主要接口**: `/process-document`, `/process-folder`, `/query`, `/processing-status`。
- **实现细节**: 
    - 使用 `lifespan` 上下文管理器在应用启动时初始化所有单例服务（如 `RAGFactory`, `FirebaseManager`）。
    - 通过 FastAPI 的 `Depends` 系统，在处理请求时按需注入 `RAGFactory` 和 `AuthService`。
    - 对于耗时的文档处理任务，利用 `BackgroundTasks` 实现异步执行，立即向客户端返回一个任务ID，避免了长轮询。

### 2.2 RAG 服务工厂 (rag_factory.py)

- **核心类**: `RAGFactory`
- **核心功能**: 根据传入的 `approach` 字符串（如 `"raganything"`），动态地创建并返回一个具体的RAG服务实例。
- **设计模式**: 工厂模式 (Factory Pattern)。
- **实现细节**: 工厂内部维护一个 `_implementations` 字典，该字典在初始化时动态扫描并注册所有可用的RAG服务。这种设计极大地提高了系统的可扩展性，当需要支持一种新的RAG方法时，只需添加一个新的服务实现文件，而无需修改任何现有核心代码。

### 2.3 RAG 服务接口 (rag_interface.py)

- **核心类**: `RAGInterface` (抽象基类)
- **核心功能**: 定义了所有RAG实现都必须遵守的统一方法，如 `process_document`, `process_folder`, `query_documents`。
- **设计模式**: 策略模式 (Strategy Pattern) 的核心，将算法的实现与使用分离开。
- **目的**: 确保 `main.py` 中的API路由逻辑可以统一地处理任何一种RAG方案，而无需关心其内部具体实现。

### 2.4 具体RAG服务 (rag_anything_service.py, etc.)

- **核心类**: `RAGAnythingService`, `RAGVertexService`, `EvidenceSweepService`。
- **核心功能**: 封装特定RAG技术栈的完整逻辑，包括初始化、文档处理和查询。
- **实现细节**: 每个服务类都继承自 `RAGInterface` 并实现了其所有抽象方法。例如：
    - `RAGAnythingService` 负责调用 `RAG-Anything` 库，在本地文件系统上创建和管理索引。
    - `RAGVertexService` 负责调用 Google Cloud 的 API，在云端创建和管理 RAG Corpus。
    - `EvidenceSweepService` 则执行一个自定义的多步骤LLM调用链（页面筛选 -> 证据提取 -> 综合分析）。

## 3. 数据协议与模块交互

### 3.1 API 数据协议 (models/api_models.py)

系统使用 Pydantic 模型来定义清晰、强类型的API数据结构。例如，处理单个文档的请求体由 `ProcessDocumentRequest` 模型定义：

```python
class ProcessDocumentRequest(BaseModel):
    user_id: str
    project_id: str
    workspace_id: str
    gcs_path: str
    rag_approach: str
    parser: Optional[str] = "mineru"
    model: Optional[str] = "gpt-4o-mini"
    config: Optional[Dict[str, Any]] = None
```

### 3.2 异步处理流程

1.  客户端向 `/process-document` 发送一个POST请求。
2.  `main.py` 接收请求，并立即调用 `AuthService` 验证用户权限。
3.  验证通过后，`FirebaseManager` 在数据库中创建一个状态为 `pending` 的任务记录。
4.  `main.py` 将任务ID立即返回给客户端（HTTP 200 OK）。
5.  一个 `BackgroundTask` 在后台被触发，它通过 `RAGFactory` 获取正确的RAG服务实例，并调用其 `process_document` 方法。
6.  该RAG服务在执行过程中，会持续通过 `FirebaseManager` 更新数据库中的任务进度和最终结果。
7.  客户端可以使用任务ID，通过轮询 `/processing-status/{job_id}` 端点来获取实时进度。

### 3.3 数据流

- **输入数据**: Google Cloud Storage (GCS) 上的文件路径（字符串）。
- **中间数据**: 服务从GCS下载文件到本地临时目录进行处理。
- **处理结果**: 每种RAG方案将其索引和元数据存储在各自的后端（`RAGAnything` 使用本地文件系统，`Vertex AI` 使用Google Cloud服务）。
- **状态数据**: 所有任务的元数据、状态和进度都以JSON对象的形式存储在 Firestore 中。
- **输出数据**: API以JSON格式返回任务ID、状态或查询结果。

### 3.4 模块交互图

```mermaid
graph TD
    A[User via API] --> B[FastAPI App (main.py)];
    B -- validates with --> C[AuthService];
    B -- creates job & returns job_id --> D[FirebaseManager];
    subgraph Background Task
        B -- triggers --> E{RAG Factory};
        E -- creates --> F[RAG Service (e.g., RAGAnythingService)];
        F -- downloads from --> G[GCS Manager];
        F -- processes & indexes --> H[RAG Backend Storage];
        F -- updates status --> D;
    end
    A -- polls status --> B;
    B -- gets status --> D;
```
