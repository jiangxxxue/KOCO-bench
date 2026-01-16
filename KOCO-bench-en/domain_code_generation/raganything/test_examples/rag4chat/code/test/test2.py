import asyncio
import aiohttp
import base64
from typing import List, Dict, Any, Optional
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import os
import shutil
import logging
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 如果要使用 rerank，需要安装 sentence-transformers
try:
    from sentence_transformers import CrossEncoder
    RERANK_AVAILABLE = True
except ImportError:
    RERANK_AVAILABLE = False


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

    # 构建 payload
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    # 过滤 kwargs，只保留可 JSON 序列化的基础类型
    safe_kwargs = {
        k: v for k, v in kwargs.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }
    payload.update(safe_kwargs)

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:11434/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Ollama error {resp.status}: {text}")
            result = await resp.json()
            return result["choices"][0]["message"]["content"]


async def ollama_embed_async(texts: List[str], model: str = "bge-m3:latest") -> List[List[float]]:
    """
    异步获取嵌入向量，确保返回正确的维度
    """
    embeddings = []
    async with aiohttp.ClientSession() as session:
        for text in texts:
            if not text.strip():  # 处理空文本
                # 返回零向量
                embeddings.append([0.0] * 1024)
                continue
                
            payload = {"model": model, "input": text}
            try:
                async with session.post("http://localhost:11434/api/embeddings", json=payload) as resp:
                    if resp.status != 200:
                        text_resp = await resp.text()
                        logger.warning(f"Embedding error for text '{text[:50]}...': {text_resp}")
                        # 返回零向量作为后备
                        embeddings.append([0.0] * 1024)
                    else:
                        result = await resp.json()
                        embedding = result["embedding"]
                        # 确保维度正确
                        if len(embedding) != 1024:
                            logger.warning(f"Embedding dimension mismatch: expected 1024, got {len(embedding)}")
                            # 调整维度
                            if len(embedding) > 1024:
                                embedding = embedding[:1024]
                            else:
                                embedding.extend([0.0] * (1024 - len(embedding)))
                        embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Embedding exception for text '{text[:50]}...': {e}")
                # 返回零向量作为后备
                embeddings.append([0.0] * 1024)
    return embeddings


# =======================
# 异步嵌入函数包装器
# =======================

class AsyncEmbeddingWrapper:
    def __init__(self, model: str = "bge-m3:latest"):
        self.model = model
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await ollama_embed_async(texts, self.model)


# =======================
# Rerank 函数（可选）
# =======================

rerank_model_func = None
if RERANK_AVAILABLE:
    try:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
        
        def rerank_model_func(query: str, docs: List[str]) -> List[str]:
            if not docs:
                return []
            pairs = [(query, doc) for doc in docs]
            scores = _reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in ranked]
    except Exception as e:
        logger.warning(f"Rerank model initialization failed: {e}")
        rerank_model_func = None


# =======================
# 主函数
# =======================
async def main():
    # 设置模型名称（确保已用 ollama pull 下载）
    LLM_MODEL = "qwen3:8b"           # 文本模型
    EMBED_MODEL = "bge-m3:latest"    # 嵌入模型
    VLM_MODEL = "llava:latest"       # 视觉语言模型

    # 创建嵌入包装器实例
    embedding_wrapper = AsyncEmbeddingWrapper(EMBED_MODEL)

    # 创建 RAGAnything 配置
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # 选择解析器：mineru 或 docling
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # LLM 函数（异步）
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await ollama_complete_async(
            model=LLM_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )

    # 视觉模型函数（支持图像输入）
    async def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        **kwargs
    ):
        return await ollama_complete_async(
            model=VLM_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            image_data=image_data,
            **kwargs
        )

    # 嵌入函数（真正的异步函数）
    async def async_embedding_func(texts: List[str]) -> List[List[float]]:
        # 过滤空文本
        filtered_texts = [text if text.strip() else " " for text in texts]
        return await embedding_wrapper.embed(filtered_texts)

    # 嵌入函数包装成 EmbeddingFunc 格式
    embedding_func = EmbeddingFunc(
        embedding_dim=1024,  # bge-m3 是 1024 维
        max_token_size=512, # 支持长文本
        func=async_embedding_func,
    )

    # 初始化 RAGAnything（不使用 rerank_model_func 参数）
    rag_args = {
        "config": config,
        "llm_model_func": llm_model_func,
        "vision_model_func": vision_model_func,
        "embedding_func": embedding_func,
    }

    # 只有在支持的情况下才添加 rerank_model_func
    if rerank_model_func is not None:
        try:
            rag = RAGAnything(**rag_args, rerank_model_func=rerank_model_func)
        except TypeError:
            logger.warning("Current RAGAnything version doesn't support rerank_model_func, initializing without it")
            rag = RAGAnything(**rag_args)
    else:
        rag = RAGAnything(**rag_args)

    # 处理文档前先清理旧存储（手动方式）
    if os.path.exists("./rag_storage"):
        try:
            shutil.rmtree("./rag_storage")
            print("🧹 已清理旧的存储目录")
        except Exception as e:
            print(f"⚠️ 清理旧目录失败: {e}")

    # 确保输出目录存在
    os.makedirs("./output", exist_ok=True)

    # 处理文档
    try:
        print("🚀 开始处理文档...")
        await rag.process_document_complete(
            file_path=r"D:\adavance\tsy\rag4chat\test.pdf",  # 修改为你的 PDF 路径
            output_dir="./output",
            parse_method="auto"
        )
        print("✅ 文档处理完成！")
    except Exception as e:
        print(f"❌ 文档处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("🔍 执行文本查询...")
        # 文本查询
        text_result = await rag.aquery(
            "文档的主要内容是什么？",
            mode="hybrid"
        )
        print("📄 文本查询结果:", text_result)
    except Exception as e:
        print(f"❌ 文本查询失败: {e}")
        # 不要让查询失败影响程序继续运行
        pass

    try:
        print("📊 执行多模态查询...")
        # 多模态查询（表格）- 修复格式
        table_md = """| 系统 | 准确率 | F1分数 |
|------|--------|-------|
| RAGAnything | 95.2% | 0.94 |
| 基准方法 | 87.3% | 0.85 |"""

        # 修复多模态内容格式
        multimodal_result = await rag.aquery_with_multimodal(
            "分析这个性能数据并解释与现有文档内容的关系",
            multimodal_content=[
                {
                    "type": "table",
                    "content": table_md,
                    "description": "性能对比结果"
                }
            ],
            mode="hybrid"
        )
        print("📊 多模态查询结果:", multimodal_result)
    except Exception as e:
        print(f"❌ 多模态查询失败: {e}")
        import traceback
        traceback.print_exc()


# 添加一个简单的测试函数
async def test_embedding():
    """测试嵌入功能"""
    print("🧪 测试嵌入功能...")
    try:
        test_texts = ["Hello world", "This is a test", ""]
        embeddings = await ollama_embed_async(test_texts, "bge-m3:latest")
        print(f"✅ 嵌入测试成功，返回 {len(embeddings)} 个向量")
        for i, emb in enumerate(embeddings):
            print(f"  文本 {i}: 维度 {len(emb)}")
    except Exception as e:
        print(f"❌ 嵌入测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 可以先运行测试
    # asyncio.run(test_embedding())
    
    # 运行主程序
    asyncio.run(main())