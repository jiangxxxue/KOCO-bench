# RAG-Anything 服务核心函数描述

本文档详细描述了 `RAGAnythingService` 中的两个核心初始化函数。这两个函数分别负责在服务启动时创建用于“写入”（文档处理）的持久化RAG实例，以及在查询时创建用于“读取”的临时RAG实例，以确保数据的一致性和实时性。

---

# FUNCTION: _initialize_rag_storage

## 功能概述
此函数在 `RAGAnythingService` 服务首次初始化时被调用，其核心职责是创建一个长期存在的、用于处理和索引文档的 `RAGAnything` 共享实例 (`self.rag_anything`)。

## 函数签名

def _initialize_rag_storage(self) -> None

## 输入参数
- `self`: `RAGAnythingService` 的实例对象。

(该函数不接收直接参数，而是通过 `os.getenv` 从环境变量中读取所有配置)

## 详细描述
从环境变量读取配置，创建RAG引擎配置并定义LLM、视觉和嵌入模型函数，初始化RAGAnything实例用于文档处理和索引。

## 输出
- **无直接返回值**。

## 函数实现
code/src/services/rag_anything_service.py:line 46-154`

## 测试代码
code/testcode/test_initialize_rag_storage.py

---

# FUNCTION: _create_rag_instance

## 功能概述
此函数用于在每次执行查询 (`query_documents`) 时，创建一个全新的、临时的 `RAGAnything` 实例。其核心目的是确保查询引擎能够加载并使用磁盘上最新状态的知识库，避免因服务长期运行而导致的数据不一致问题。

## 函数签名

def _create_rag_instance(self) -> RAGAnything

## 输入参数
- `self`: `RAGAnythingService` 的实例对象。

## 详细描述
重新定义模型函数，从存储目录加载已有的知识库数据到LightRAG实例，创建新的RAGAnything实例并注入数据，返回包含最新知识库状态的实例。

## 输出
- **返回值**: `rag_instance` - 一个全新的、已准备好执行查询的 `RAGAnything` 对象实例。

## 函数实现
code/src/services/rag_anything_service.py:line 157-271`

## 测试代码
code/testcode/test_create_rag_instance.py
