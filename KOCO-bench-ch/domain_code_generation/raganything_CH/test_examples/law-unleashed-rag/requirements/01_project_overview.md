# law-unleashed-rag: RAG 测试平台服务

## 1. 系统概述

`law-unleashed-rag` 是一个基于 FastAPI 构建的、用于测试和比较多种检索增强生成（RAG）方案的测试平台服务。该系统提供了一个统一的API接口，允许用户使用不同的RAG实现（如 `RAGAnything`, `Vertex AI RAG`, `EvidenceSweep`）来处理存储在Google Cloud Storage (GCS) 上的文档，并通过 Firestore 对处理过程进行实时追踪。

该平台的核心价值在于其内置的综合评估框架，能够对不同的RAG方法在准确性、性能和成本等多个维度上进行客观比较，从而帮助开发者为特定场景（如法律、医疗文档分析）选择最优的技术栈。

## 2. 工作流

该系统的工作流是围绕API调用和后台异步处理来设计的。用户通过API提交一个文档处理任务，系统在后台完成处理，并允许用户随时查询任务状态。

```
Algorithm: RAG Test Bed Service Pipeline
Input: GCS document path, User ID, chosen RAG approach, parser, etc.
Output: A job ID for tracking, and processed knowledge in the backend.

// 1. User initiates a processing job via API
1: POST /process-document with {gcs_path, rag_approach, parser, ...}
2:
3: // 2. FastAPI Backend handles the request
4: Authenticate user and workspace
5: Create a job ID and record the job in Firestore with "pending" status
6: Return job_id to the user immediately
7.
8: // 3. Background task starts processing
9: Download the document from GCS
10: Select the RAG implementation using the RAG Factory (e.g., RAGAnything, Vertex AI)
11: Use the selected RAG approach to process the document (parse, chunk, embed, index)
12: Update the job status and progress in Firestore throughout the process
13:
14: // 4. User checks status
15: GET /processing-status/{job_id} to monitor progress
```

Pipeline中的关键组件说明：
- **FastAPI Endpoints**: 提供异步任务创建（`/process-document`, `/process-folder`）和状态查询（`/processing-status`）的接口。
- **RAG Factory**: 一个核心组件，根据用户的请求参数（`rag_approach`）动态选择并初始化相应的RAG实现（如 `RAGAnythingService`, `VertexAIRAGService`）。
- **GCS Manager**: 负责与 Google Cloud Storage 交互，安全地下载源文件。
- **Firestore Service**: 负责在 Firestore 数据库中创建、更新和查询处理任务的状态和进度。
- **RAG Implementations**: 各种RAG方案的具体实现，每个方案都封装了自己的文档处理、索引和存储逻辑。

## 3. 应用场景

### RAG 方法比较与评估
- **输入**: 一组标准化的测试文档（如法律合同、医疗记录）和一系列预设的评估指标。
- **输出**: 对不同RAG方法（`RAGAnything`, `Vertex AI RAG`, `EvidenceSweep`）在准确性、性能、成本等方面的详细比较报告。
- **目的**: 为特定业务场景选择最合适的RAG技术栈，提供数据驱动的决策支持。

### 可扩展的异步文档处理服务
- **输入**: 存储在GCS上的海量文档。
- **输出**: 一个可通过API调用的、结构化的知识库，以及用于监控处理进度的任务ID。
- **目的**: 提供一个企业级的、可监控的、异步的文档处理流水线，解耦了任务提交和实际处理过程，适用于大规模文档的知识化场景。
