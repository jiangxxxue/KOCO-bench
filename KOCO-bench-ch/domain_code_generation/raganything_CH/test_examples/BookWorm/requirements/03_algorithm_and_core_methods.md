# RAGAnything 核心函数描述

# FUNCTION: process_with_rag

## 函数签名

async def process_with_rag(file_path: str, output_dir: str, api_key: str, base_url: str = None, working_dir: str = None, parser: str = None) -> str

## 功能概述
该函数是 `RAGAnything` 库的一个端到端示例，完整地展示了处理单个文档并进行多种模式查询的全过程。它首先配置并初始化 `RAGAnything` 引擎，然后处理指定文档以构建知识库，最后执行一系列预设的文本和多模态查询，以验证和展示系统的核心能力。

## 输入参数
- `file_path: str`: 需要处理的文档的路径。
- `output_dir: str`: RAG处理结果的输出目录。
- `api_key: str`: 用于访问大语言模型（LLM）服务的API密钥。
- `base_url: str`: LLM API的基础URL，默认为 `http://brainmachine:11434`。
- `working_dir: str`: RAG持久化存储（如向量索引、缓存等）的工作目录，默认为 `./rag_storage`。
- `parser: str`: 指定文档解析器，默认为 `minerus`。

## 详细描述

配置工作目录和文档解析方式，设置好文本、图像和嵌入模型，初始化RAG系统后处理文档，最后用纯文本、表格和公式三种方式验证文档理解能力。


## 输出
该函数没有显式的返回值 (`return` None)。

## 函数实现
code/raganything_example.py:line 90-245

## 测试代码
code/test_code/test_process_with_rag.py