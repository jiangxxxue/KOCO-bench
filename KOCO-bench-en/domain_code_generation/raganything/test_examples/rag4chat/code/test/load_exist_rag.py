import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything
import os
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# 工具函数：调用 Ollama API（异步）
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

async def embedding_func(texts):
    return await ollama_embed_async(texts, EMBED_MODEL)

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

LLM_MODEL = "qwen3:8b"
EMBED_MODEL = "bge-m3:latest"
VLM_MODEL = "llava:latest"

lightrag_working_dir = "./rag_storage"

# 创建或加载 LightRAG 实例
lightrag_instance = LightRAG(
    working_dir=lightrag_working_dir,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=5122,
        func=embedding_func
    )
)

# =======================
# 初始化 RAGAnything
# =======================

rag = RAGAnything(
    lightrag=lightrag_instance,
    vision_model_func=vision_model_func
)


# =======================
# 示例查询
# =======================

async def main():
    print("✅ 正在加载已存在的 LightRAG 实例...")
    await lightrag_instance.initialize_storages()

    # 查询示例
    result = await rag.aquery("这个知识库中处理了哪些数据？", mode="hybrid")
    print("🔍 查询结果:", result)

    # 可选：添加新文档
    # await rag.process_document_complete(file_path="path/to/new/multimodal_document.pdf", output_dir="./output")


if __name__ == "__main__":
    asyncio.run(main())