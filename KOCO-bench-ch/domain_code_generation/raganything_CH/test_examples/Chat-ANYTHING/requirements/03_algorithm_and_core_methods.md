# RAGAnything 核心函数描述


# FUNCTION: MCPGeminiRAGClient.__init__

## 功能概述

创建一个集成了本地文档知识库、Gemini人工智能模型和外部工具调用能力的统一智能客户端。建立一个完整的智能问答系统，既能从本地文档中检索知识，又能调用外部工具获取实时信息。

## 函数签名

def __init__(self, gemini_api_key: str) -> RAGAnything

## 输入参数

- `gemini_api_key: str`: 用于访问Gemini API服务的密钥

## 详细描述

该函数初始化会话管理，连接 Gemini 模型，配置文档处理与文本向量化能力，设置语言理解模型，初始化知识检索引擎以构建多层次智能服务系统。

## 输出

- **返回值**: `rag` - 已配置完成的知识检索与问答引擎实例（RAGAnything类型）

## 函数实现

code/rag-web-ui/backend/app/services/my_rag.py:line 15-152

## 测试代码
code/test_code/test_MCPGeminiRAGClient_init.py

---

# FUNCTION: load_documents

## 功能概述

将新文档集成到知识库系统中，使其内容可被检索和查询。将原始文档转换为结构化的知识表示，并建立相应的检索索引。

## 函数签名

async def load_documents(self, file_path: str) -> None

## 输入参数

- `file_path: str`: 需要加载的文档文件路径

## 详细描述

该函数对指定文档进行结构化解析处理（含文本、图像等元素），标记加载状态并反馈处理结果，实现文档智能化入库。

## 输出

该函数没有显式的返回值

## 函数实现

code/rag-web-ui/backend/app/services/my_rag.py:line 155-163

## 测试代码
code/test_code/test_load_documents.py

---

# FUNCTION: process_query

## 功能概述

该函数是智能问答系统的核心处理引擎，需要能智能地判断何时使用本地知识库、何时需要获取外部实时信息。通过先检索本地文档再评估是否需要外部支持的策略，确保为用户提供最准确和最新的答案。

## 函数签名

async def process_query(self, query: str) -> str

## 输入参数

- `query: str`: 用户的查询问题

## 详细描述

该函数先明确当前可用的外部工具与服务，实时维护对话上下文以衔接多轮交互，通过混合检索策略从本地知识库中精准查找相关信息， 智能评估检索结果的充分性与完整性，若本地知识不足以解答或需实时数据则调用外部工具补充，最终整合信息生成答案并同步更新对话历史，为后续交互提供持续支持。


## 输出

- **返回值**: `str` - 针对用户问题的智能回答，可能基于本地知识库、外部实时信息或两者的结合

## 函数实现

code/rag-web-ui/backend/app/services/my_rag.py:line 217-248

## 测试代码
code/test_code/test_process_query.py
