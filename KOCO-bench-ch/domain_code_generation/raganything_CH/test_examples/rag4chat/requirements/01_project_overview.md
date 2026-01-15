# rag4chat: 本地的智能客服构建

## 1. 系统概述

`rag4chat` 是一个功能全面的、可本地部署的智能客服聊天应用。该项目旨在提供一个灵活的框架，用户既可以通过其内置的、基于 **Streamlit** 的Web图形界面直接与AI助手交互，也可以通过其提供的、基于 **Flask** 的API服务，将智能问答能力集成到现有的业务系统中。

系统的核心是其强大的检索增强生成（RAG）能力。用户可以通过界面或代码，将产品手册、FAQ、知识库文档（支持md, docx, txt, pdf等多种格式）方便地构建成一个向量知识库。在回答问题时，系统能够首先从知识库中检索最相关的信息，然后将这些信息作为上下文，交由大语言模型（LLM）生成精准、可靠的答案。此外，系统还具备一定的意图识别（如转人工、天气查询）和多模态对话能力。

## 2. 工作流

`rag4chat` 的核心工作流围绕着“知识库构建”和“RAG问答”两个环节。以下以其 Streamlit 应用为例，展示其主要处理流程。

```
Algorithm: rag4chat RAG Pipeline (Streamlit App)
Input: User query, Uploaded documents (optional)
Output: Answer from the chatbot

// Part 1: Knowledge Base Construction (via UI)
1: User uploads documents (PDF, DOCX, MD, etc.) through the web interface.
2: DocumentProcessor parses the files.
3: VectorStoreService creates embeddings and indexes the documents.
4: The knowledge base is now ready.

// Part 2: Chat Interaction (RAG Enabled)
5: procedure HandleUserQuery(query_text)
6:    // Step 1: Retrieve
7:    // 从向量数据库中搜索与问题最相关的文档片段
8:    relevant_docs = VectorStoreService.search_documents(query_text)
9:
10:   // Step 2: Augment
11:   // 将检索到的文档片段整合成提供给LLM的上下文
12:   context = VectorStoreService.get_context(relevant_docs)
13:
14:   // Step 3: Generate
15:   // 调用RAG代理，将问题和上下文一起发送给LLM
16:   agent = RAGAgent(selected_model)
17:   answer = agent.run(prompt=query_text, context=context)
18:
19:   // Step 4: Display
20:   Display answer in the chat UI.
21:   Add query and answer to chat history.
22: end procedure
```

Pipeline中的关键组件说明：
- **app.py**: 基于 Streamlit 的主应用，负责渲染前端UI、管理会话状态和编排整个问答流程。
- **app_api.py**: 基于 Flask 的后端服务，提供 `/v1/chat/completions` 等API接口，使RAG功能可以被其他程序调用。
- **DocumentProcessor**: 负责处理用户上传的文档，包括文件读取和文本解析。
- **VectorStoreService**: 核心的向量存储服务。它管理着知识库的构建（将文档向量化并索引）和检索（根据问题进行相似度搜索）。
- **RAGAgent**: 与大语言模型（LLM）直接交互的代理。它接收用户问题和从向量数据库中检索到的上下文，并生成最终的回答。
- **ChatHistoryManager**: 会话历史管理器，用于支持多轮对话。

## 3. 应用场景

### 企业智能客服
- **输入**: 企业的内部文档，如产品手册、操作指南、规章制度、FAQ列表等。
- **输出**: 一个能够7x24小时在线，为客户或内部员工准确解答各类问题的智能客服机器人。
- **目的**: 自动化处理常见问题，释放人工客服的压力，提供标准化、高质量的信息服务。

### 个人知识库助手
- **输入**: 任何个人积累的电子文档，如学习笔记、研究资料、会议纪要、电子书等。
- **输出**: 一个完全私有化、充分理解用户个人知识体系的智能助手。
- **目的**: 帮助用户在自己积累的知识海洋中快速检索信息、总结观点、激发灵感，让知识“活”起来。
