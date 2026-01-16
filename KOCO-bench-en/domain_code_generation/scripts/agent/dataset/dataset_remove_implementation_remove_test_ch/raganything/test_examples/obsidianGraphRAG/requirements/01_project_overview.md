# ObsidianGraphRAG: 与你的笔记对话

## 1. 系统概述

ObsidianGraphRAG 是一个旨在将个人 Obsidian 笔记库（Vault）转化为一个可查询、可对话的本地知识库的系统。它致力于解决个人知识管理中的核心痛点：笔记内容繁多，导致信息遗忘和思想连接断裂。通过本系统，用户可以像与一位博学的私人助理交谈一样，与自己所有的笔记进行互动，发掘被遗忘的知识和隐藏在笔记间的深层联系。

系统的核心设计哲学是“本地优先”和“数据私有”。所有的知识解析、向量嵌入和相关性重排（Reranking）过程均在用户本地设备上运行，仅在最后生成答案的阶段调用外部的 Gemini API（免费套餐），从而在实现强大功能的同时，最大限度地保障了用户数据的隐私和安全，且几乎无运行成本。

## 2. 工作流

系统的核心工作流分为两个主要阶段：**知识库构建（首次构建与增量同步）**和**查询应答**。

```
Algorithm: ObsidianGraphRAG Pipeline
Input: Obsidian Vault Path, User Query
Output: Answer based on user's notes

// Part 1: Indexing (Initial Build or Incremental Sync)
1: procedure RunIndexing(vault_path, is_incremental)
2:    if is_incremental then
3:        // 增量同步：仅获取被修改或新增的笔记
4:        notes_to_process = vault_monitor.get_changed_notes(vault_path)
5:    else
6:        // 首次构建：获取全部笔记
7:        notes_to_process = vault_monitor.get_all_notes(vault_path)
8:    end if
9:
10:   for each note in notes_to_process do
11:       // 使用为Obsidian优化的切分器处理笔记
12:       chunks = obsidian_chunker.chunk(note)
13:       // 添加文本块并生成向量嵌入
14:       rag_system.add_chunks(chunks)
15:       // 提取实体和关系，更新知识图谱
16:       rag_system.extract_and_add_graph_elements(chunks)
17:   end for
18:   // 更新文件状态，以便下次增量同步
19:   vault_monitor.update_sync_status(processed_notes)
20: end procedure
21:
// Part 2: Querying
22: procedure HandleQuery(query_text)
23:   // 步骤1: 混合检索
24:   retrieved_chunks = rag_system.vector_search(query_text)
25:   retrieved_graph_context = rag_system.graph_search(query_text)
26:
27:   // 步骤2: 结果重排
28:   reranked_chunks = bge_reranker.rerank(query_text, retrieved_chunks)
29:
30:   // 步骤3: 上下文构建与答案生成
31:   context = format_prompt(reranked_chunks, retrieved_graph_context)
32:   answer = gemini_llm.generate(query_text, context)
33:
34:   return answer
35: end procedure
```

Pipeline中的关键组件说明：
- **run_obsidian_raganything.py**: 用于首次执行的脚本，它会完整地处理整个Obsidian笔记库，构建初始的知识库。
- **run_incremental_sync.py**: 日常使用的增量同步脚本。它通过比较文件哈希，只处理新增或被修改过的笔记，极大地提高了更新效率。
- **run_ui.py**: 启动一个基于 FastAPI 和 WebSocket 的本地Web服务器，提供类似ChatGPT的交互界面。
- **obsidian_chunker.py**: 一个为Obsidian笔记定制的文本切分器，能够理解和保留笔记间的wikilinks，从而在切分时保留上下文的关联性。
- **bge_reranker.py**: BGE重排模型，在检索后对结果进行二次排序，将最相关的内容排在前面，能有效提升答案的准确率（15-30%）。
- **gemini_llm.py**: 封装了与 Google Gemini API 的交互逻辑，负责将最终的上下文和问题发送给LLM，并以流式方式返回生成的答案。

## 3. 应用场景

### 个人知识管理与回顾
- **输入**: 一个包含长期积累的、大量笔记的Obsidian库，以及一个关于笔记内容的自然语言问题（例如：“我之前对于‘决策模型’都记录了哪些想法？”）。
- **输出**: 一个基于用户自己笔记的、综合性的回答，可能会揭示不同时间、不同主题笔记之间的隐藏联系。
- **目的**: 帮助用户对抗遗忘，高效地回顾、整合和再利用自己记录过的知识，真正实现知识的复利。

### 创意与研究辅助
- **输入**: 一个关于研究主题或创意方向的模糊问题（例如：“我的笔记里，有哪些关于‘心理学’和‘市场营销’的交叉内容？”）。
- **输出**: 系统从知识库中找出所有相关的概念、笔记和连接，提供一个结构化的概览或总结，作为创新的起点。
- **目的**: 辅助研究和创作过程，通过深度挖掘现有笔记的内在潜力来激发新想法，而不是从零开始。
