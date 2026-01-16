import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from lightrag  import LightRAG
from lightrag.utils import EmbeddingFunc
# 假设 lightrag 的检索模块结构，具体导入路径可能需要根据你的 lightrag 版本调整
# 常见的可能是 lightrag.operate.retrieval 或类似
# 如果下面的导入失败，请查看你的 lightrag 安装目录下的源码结构
try:
    from lightrag.operate import retrieval
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    print("Warning: Could not import lightrag.operate.retrieval. Retrieval details might not be available.")
import os
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "bge-m3:latest"
VLM_MODEL = "llava:latest"

# =======================
# 工具函数：调用 Ollama API（异步）
# ... (保持不变) ...
# =======================

async def ollama_complete_async(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    image_data: Optional[str] = None,
    **kwargs
) -> str:
    """
    异步调用 Ollama，支持文本 + 图像输入
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # 构造 content：支持文本 + 图像
    content = [{"type": "text", "text": prompt}]
    if image_data:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })

    messages.append({"role": "user", "content": content})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:11434/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Ollama error {resp.status}: {text}")
            result = await resp.json()
            return result["choices"][0]["message"]["content"]


async def ollama_embed_async(texts: List[str], model: str = "bge-m3:latest") -> List[List[float]]:
    """
    异步获取嵌入向量
    """
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            if not text.strip():
                embeddings.append([0.0] * 1024)
                continue
            payload = {"model": model, "input": text}
            try:
                async with session.post("http://localhost:11434/api/embeddings", json=payload) as resp:
                    if resp.status != 200:
                        text_resp = await resp.text()
                        logger.warning(f"Embedding error for text '{text[:50]}...': {text_resp}")
                        embeddings.append([0.0] * 1024)
                    else:
                        result = await resp.json()
                        embedding = result["embedding"]
                        # 补齐或截断到 1024 维度
                        if len(embedding) > 1024:
                            embedding = embedding[:1024]
                        elif len(embedding) < 1024:
                            embedding += [0.0] * (1024 - len(embedding))
                        embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Embedding exception for text '{text[:50]}...': {e}")
                embeddings.append([0.0] * 1024)
    return embeddings


# =======================
# 包装函数（在异步环境中直接调用）
# =======================

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await ollama_complete_async(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

# async def embedding_func(texts):
#     return await ollama_embed_async(texts, EMBED_MODEL)
from build_database import RAGDatabaseManager

async def embedding_func(texts):
    db_manager = RAGDatabaseManager(embed_model="text-embedding-v4")
    await db_manager._create_rag_instance()
    embed_func = db_manager.embedding_func.func  # 提取真正的异步函数

    embeddings = await embed_func(texts)
    return embeddings

async def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
    return await ollama_complete_async(
        model=VLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        image_data=image_data,
        **kwargs
    )

# =======================
# LightRAG 配置
# =======================

lightrag_working_dir = "./rag_storage"

# 创建或加载 LightRAG 实例
lightrag_instance = LightRAG(
    working_dir=lightrag_working_dir,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=512,
        func=embedding_func
    )
)

# =======================
# 示例查询并尝试输出命中的知识库条目
# =======================

async def main():
    print("✅ 正在加载已存在的 LightRAG 实例...")
    await lightrag_instance.initialize_storages()

    query_text = "语义图像如何构建"

    # 1. 尝试通过底层检索模块获取 chunks
    print(f"\n🔍 正在检索与 '{query_text}' 相关的知识库条目...")
    
    if RETRIEVAL_AVAILABLE and hasattr(lightrag_instance, 'chunk_vdb') and hasattr(lightrag_instance, 'embedding_func'):
        try:
            # 使用 lightrag 内部的 retrieval 模块进行检索
            # 这通常需要 query_text, top_k, chunk_db, embed_func 等参数
            # 注意：这个调用的具体签名依赖于 lightrag 的版本
            # 假设有一个 text_retrieve 函数
            if hasattr(retrieval, 'text_retrieve'):
                 # 假设 text_retrieve 的签名类似于:
                 # async def text_retrieve(query, EmbeddingFunc, chunk_db, top_k: int = 10)
                 retrieved_chunks = await retrieval.text_retrieve(
                     query=query_text,
                     embedding_func=lightrag_instance.embedding_func, # 注意：这里可能需要直接传入 func
                     chunk_db=lightrag_instance.chunk_vdb,
                     top_k=5
                 )
                 print(f"\n📄 通过 retrieval 模块检索到的 Top {len(retrieved_chunks)} 知识库条目:")
                 if retrieved_chunks:
                     # retrieved_chunks 的格式可能是一个包含 id 和/或 content 的列表或字典
                     # 需要根据实际返回格式调整
                     for i, chunk_info in enumerate(retrieved_chunks):
                         print(f"\n--- 条目 {i+1} ---")
                         # 尝试打印可用信息
                         if isinstance(chunk_info, dict):
                             for key, value in chunk_info.items():
                                 if key != 'content' or len(str(value)) < 200: # 避免打印过长内容
                                     print(f"{key}: {value}")
                                 else:
                                     print(f"{key}: {str(value)[:200]}...")
                         else:
                             print(f"Chunk Info: {chunk_info}")
                 else:
                     print("未检索到任何相关条目。")
            else:
                print("lightrag.operate.retrieval 模块中未找到 text_retrieve 函数。")
        except Exception as e:
            print(f"通过 retrieval 模块检索时发生错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("无法使用底层 retrieval 模块进行检索 (模块未导入或 LightRAG 实例缺少必要属性)。")

    # 2. 如果上述方法失败，尝试直接访问 chunk_vdb 的 aquery 方法 (如果存在)
    if hasattr(lightrag_instance, 'chunk_vdb'):
        try:
            query_embedding = await embedding_func([query_text])
            # 尝试直接查询 chunk_vdb
            # 假设 aquery 方法接受 embedding 向量和 top_k 参数
            # 注意：API 可能是 query, aquery, search 等，参数也可能不同
            if hasattr(lightrag_instance.chunk_vdb, 'query') or hasattr(lightrag_instance.chunk_vdb, 'aquery') or hasattr(lightrag_instance.chunk_vdb, 'search'):
                # 尝试最常见的 aquery
                vdb_query_func = getattr(lightrag_instance.chunk_vdb, 'aquery', getattr(lightrag_instance.chunk_vdb, 'query', getattr(lightrag_instance.chunk_vdb, 'search', None)))
                if vdb_query_func:
                    # 注意：这里的参数需要根据 nano-vectordb 的实际 API 调整
                    # 常见的可能是 aquery(query_vector, top_k)
                    top_k = 5
                    # query_embedding[0] 是第一个（也是唯一一个）查询文本的 embedding 向量
                    searched_result = await vdb_query_func(query_embedding[0], top_k=top_k)
                    
                    print(f"\n📄 通过直接查询 chunk_vdb 检索到的 Top {top_k} 知识库条目:")
                    # nano-vectordb 的 aquery 通常返回一个字典，包含 'distances', 'ids', 'metadatas' (如果存储了), 'documents' (如果存储了原始文本)
                    # 我们主要关心 'metadatas' 或 'documents'
                    if isinstance(searched_result, dict):
                        # 假设 'metadatas' 包含了我们存储的 chunk 信息 (content, file_path 等)
                        metadatas = searched_result.get('metadatas', [])
                        documents = searched_result.get('documents', []) # 如果存储了原始文档
                        ids = searched_result.get('ids', [])
                        
                        for i, (chunk_id, metadata) in enumerate(zip(ids, metadatas)):
                             print(f"\n--- 条目 {i+1} ---")
                             print(f"ID: {chunk_id}")
                             if isinstance(metadata, dict):
                                 # 打印元数据中的关键信息
                                 content = metadata.get('content', 'N/A')
                                 file_path = metadata.get('file_path', 'N/A')
                                 full_doc_id = metadata.get('full_doc_id', 'N/A')
                                 print(f"内容预览: {content[:200]}..." if content and len(content) > 200 else f"内容: {content}")
                                 print(f"来源文件: {file_path}")
                                 print(f"文档 ID: {full_doc_id}")
                             else:
                                 print(f"Metadata: {metadata}")
                             
                             # 如果 documents 列表也有对应内容，也可以打印
                             if i < len(documents):
                                 doc_content = documents[i]
                                 if doc_content and doc_content != metadata.get('content'): # 避免重复打印
                                     print(f"文档内容 (来自 documents): {doc_content[:100]}...")
                    else:
                        print(f"检索结果格式未知: {type(searched_result)}")
                else:
                    print("chunk_vdb 没有找到 query/aquery/search 方法。")
            else:
                print("chunk_vdb 没有找到 query/aquery/search 方法。")
                
        except Exception as e:
             print(f"直接查询 chunk_vdb 时发生错误: {e}")
             import traceback
             traceback.print_exc()
    else:
         print("LightRAG 实例没有 chunk_vdb 属性。")


    # 3. 执行查询并生成答案 (可选)
    print(f"\n🤖 正在生成针对 '{query_text}' 的回答...")
    try:
        result = await lightrag_instance.aquery(query_text)
        print(f"\n💬 查询结果: {result}")
    except Exception as e:
        print(f"查询生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())