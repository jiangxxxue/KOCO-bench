# BookWorm: 基于RAG的图书问答系统

## 1. 系统概述

BookWorm是一个基于检索增强生成（RAG）技术的交互式图书问答系统。它利用Streamlit构建用户界面，并以RAG-Anything库作为核心后端，实现了对书籍内容的深度解析与智能问答。

当用户选择一本书并提出问题时，系统首先会利用RAG-Anything引擎对书籍文件进行解析、分块和向量化，并建立索引。然后，系统会根据用户的问题，从索引中检索最相关的内容片段，并将其与问题一并交给大型语言模型（LLM），最终生成一个精准、忠于原文的答案。

## 2. 工作流程

```
Algorithm: BookWorm RAG Pipeline
Input: User-selected book, User query
Output: Answer to the query

1: Initialize Streamlit UI
2: book_path ← get_selected_book_from_user()
3: query ← get_query_from_user()
4: 
5: // Get or Create Assistant for the book
6: assistant ← get_or_create_assistant(book_path)
7: 
8: // Query the assistant
9: response ← assistant.chat(query)
10: 
11: // Display response
12: display_response_in_ui(response)
13: return response

Function get_or_create_assistant(book_path):
    // Check if an assistant for this book already exists
    if assistant_exists(book_path) then
        return load_assistant(book_path)
    else
        // Create a new assistant
        rag_anything ← initialize_RAGAnything()
        // Parse the book file and insert its content into the vector store
        rag_anything.parse_file_and_insert(book_path)
        save_assistant(rag_anything, book_path)
        return rag_anything
    end if
```

Pipeline中的关键函数说明：
- `initialize_RAGAnything()`: 初始化RAG-Anything引擎，用于处理文档解析和检索。
- `parse_file_and_insert(book_path)`: 解析指定的书籍文件，将其内容处理后存入向量数据库。
- `get_or_create_assistant(book_path)`: 获取或创建一个与特定书籍绑定的问答“助手”。
- `assistant.chat(query)`: 接收用户问题，执行RAG流程（检索+生成），并返回答案。
- `Streamlit UI`: 提供用户交互界面，包括书籍选择、问题输入和答案展示。


## 3. 应用场景

### 交互式图书内容问答
- **输入**: 用户指定的书籍文件（如PDF, TXT等）以及针对该书内容的自然语言问题。
- **输出**: 基于书籍内容的自然语言回答。
- **验证方式**: 用户可以通过比对答案和书中的原文来验证回答的准确性。
- **数据来源**: 用户上传的本地书籍文件。
