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
import json


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import asyncio
import base64
from typing import List, Dict, Any, Optional
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
import os
import shutil
import logging
import json

# =======================
# 新增：导入 AsyncOpenAI
# =======================
from openai import AsyncOpenAI

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
async def bailian_embed_async(texts: List[str], model: str = "text-embedding-v4") -> List[List[float]]:
    """
    异步调用阿里云百炼 Embedding API (text-embedding-v4)
    使用 OpenAI 兼容接口
    """
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置")

    # 初始化异步客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    embeddings = []
    target_dim = 1024  # text-embedding-v4 支持 dimensions=1024

    for i, text_item in enumerate(texts):
        # --- 1. 类型检查和预处理 ---
        if not isinstance(text_item, str):
            if text_item is None:
                logger.warning(f"bailian_embed_async: 文本 {i} 是 None，将其转换为空字符串。")
                text = ""
            else:
                logger.warning(f"bailian_embed_async: 文本 {i} 类型为 {type(text_item)}, 尝试转换为字符串。")
                text = str(text_item)
        else:
            text = text_item

        # 处理空字符串或纯空白字符串
        if not text or not text.strip():
            logger.debug(f"bailian_embed_async: 文本 {i} 为空，返回零向量。")
            embeddings.append([0.0] * target_dim)
            continue

        # --- 2. 发送请求 ---
        try:
            response = await client.embeddings.create(
                model=model,
                input=text,
                dimensions=target_dim,        # 指定维度
                encoding_format="float"       # 返回 float 列表
            )
            # --- 3. 解析响应 ---
            embedding = response.data[0].embedding  # 获取嵌入向量

            if not isinstance(embedding, list):
                logger.error(f"bailian_embed_async: 嵌入向量不是列表 (文本 {i})")
                embeddings.append([0.0] * target_dim)
                continue

            current_dim = len(embedding)

            if current_dim != target_dim:
                logger.warning(f"bailian_embed_async: 嵌入维度不匹配 (文本 {i}): 期望 {target_dim}, 实际 {current_dim}")
                # 填充或截断到 1024 维
                if current_dim > target_dim:
                    embedding = embedding[:target_dim]
                else:
                    embedding.extend([0.0] * (target_dim - current_dim))

            # 检查数值有效性
            import math
            if any(math.isnan(x) or math.isinf(x) for x in embedding):
                logger.error(f"bailian_embed_async: 嵌入向量包含 NaN 或 Inf (文本 {i})")
                embeddings.append([0.0] * target_dim)
                continue

            embeddings.append(embedding)
            logger.debug(f"bailian_embed_async: 成功为文本 {i} 生成 {len(embedding)} 维嵌入。")

        except Exception as e:
            logger.error(f"调用 百炼 Embedding API 时发生错误 (文本 {i}): {e}", exc_info=True)
            embeddings.append([0.0] * target_dim)

    return embeddings

async def ollama_complete_async(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    **kwargs
) -> str:
    """
    异步调用 Ollama (适用于文本 LLM)，支持文本输入
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:11434/v1/chat/completions", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Ollama LLM error {resp.status}: {text}")
            result = await resp.json()
            return result["choices"][0]["message"]["content"]


async def ollama_vision_complete_async(
    model: str,
    prompt: str,
    image_data: str,  # Base64 encoded image string
    **kwargs
) -> str:
    """
    异步调用 Ollama 视觉模型 (如 Llava)，支持图像 + 文本输入
    使用 /api/generate 端点
    """
    # 构建 payload，符合 Ollama 视觉模型的要求
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_data],  # 图像数据作为列表传递
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:11434/api/generate", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Ollama Vision Model error {resp.status}: {text}")
            result = await resp.json()
            # 视觉模型的响应直接在 'response' 字段
            return result.get("response", "")


# =======================
# 异步嵌入函数包装器（更新为使用 bailian）
# =======================
class AsyncEmbeddingWrapper:
    def __init__(self, model: str = "text-embedding-v4"):
        self.model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        return await bailian_embed_async(texts, self.model)


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
# 数据库构建和更新函数接口
# =======================

class RAGDatabaseManager:
    """RAG 数据库管理器 - 专门负责数据库构建和更新"""

    def __init__(
        self,
        working_dir: str = "./rag_storage",
        output_dir: str = "./output",
        llm_model: str = "qwen3:8b",
        embed_model: str = "text-embedding-v4",
        vision_model: str = "llava:latest",
        parser: str = "mineru"
    ):
        """
        初始化数据库管理器
        Args:
            working_dir: RAG 存储目录
            output_dir: 输出目录
            llm_model: LLM 模型名称
            embed_model: 嵌入模型名称
            vision_model: 视觉模型名称
            parser: 文档解析器
        """
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.vision_model = vision_model
        self.parser = parser
        # 创建嵌入包装器实例
        self.embedding_wrapper = AsyncEmbeddingWrapper(self.embed_model)
        # 确保目录存在
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    ##key
    async def _create_rag_instance(self):
        """创建 RAG 实例"""

        # 创建配置
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser=self.parser,
            parse_method="auto",
            enable_image_processing=True, # 保持启用图像处理
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # LLM 函数（异步）- 用于文本处理
        async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await ollama_complete_async(
                model=self.llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            )

        # 视觉模型函数（异步）- 用于图像处理，使用新的视觉模型 API
        async def vision_model_func(
            prompt,
            system_prompt=None, # 注意：Ollava 的 /api/generate 可能不直接支持 system prompt
            history_messages=[], # 注意：Ollava 的 /api/generate 可能不直接支持历史消息
            image_data=None,
            **kwargs
        ):
            # 如果提供了图像数据，则调用视觉模型
            if image_data:
                # 可以在这里组合 prompt 和 system_prompt 如果需要
                # 但标准的 /api/generate 接口可能不区分它们
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"

                return await ollama_vision_complete_async(
                    model=self.vision_model,
                    prompt=full_prompt,
                    image_data=image_data,
                    **kwargs
                )
            else:
                # 如果没有图像数据，回退到普通 LLM
                return await llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # 嵌入函数（真正的异步函数）
        async def async_embedding_func(texts: List[str]) -> List[List[float]]:
            # 过滤空文本
            filtered_texts = [text if text.strip() else " " for text in texts]
            return await self.embedding_wrapper.embed(filtered_texts)

        # 嵌入函数包装成 EmbeddingFunc 格式
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,  #  1024 维
            max_token_size=512,  # 支持长文本
            func=async_embedding_func,
        )
        # === 添加这三行，将函数保存为实例属性 ===
        self.llm_model_func = llm_model_func
        self.vision_model_func = vision_model_func
        self.embedding_func = embedding_func

        # 初始化 RAGAnything 参数
        rag_args = {
            "config": config,
            "llm_model_func": llm_model_func,
            "vision_model_func": vision_model_func, # 使用修改后的视觉模型函数
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
        return rag

    async def add_document(
        self,
        file_path: str,
        parse_method: str = "auto"
    ) -> bool:
        """
        向数据库添加单个文档
        Args:
            file_path: 文档路径
            parse_method: 解析方法
        Returns:
            bool: 是否成功添加
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return False
        # 获取文件扩展名
        file_extension = os.path.splitext(file_path)[1].lower()
        supported_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
        if file_extension not in supported_extensions:
            logger.warning(f"文件格式可能不支持: {file_extension}, 将尝试处理")
        try:
            # 创建 RAG 实例
            rag = await self._create_rag_instance()
            logger.info(f"🚀 开始处理文档: {file_path}")
            await rag.process_document_complete(
                file_path=file_path,
                output_dir=self.output_dir,
                parse_method=parse_method
            )
            logger.info(f"✅ 文档处理完成: {file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 文档处理失败: {file_path}, 错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def add_documents(
        self,
        file_paths: List[str],
        parse_method: str = "auto"
    ) -> Dict[str, bool]:
        """
        批量向数据库添加多个文档
        Args:
            file_paths: 文档路径列表
            parse_method: 解析方法
        Returns:
            Dict[str, bool]: 每个文档的处理结果
        """
        results = {}
        for file_path in file_paths:
            results[file_path] = await self.add_document(file_path, parse_method)
        return results

    async def rebuild_database(
        self,
        file_paths: List[str],
        clear_existing: bool = True
    ) -> bool:
        """
        重新构建数据库（可选清除现有数据）
        Args:
            file_paths: 要添加的文档路径列表
            clear_existing: 是否清除现有数据
        Returns:
            bool: 是否成功重建
        """
        try:
            # 如果需要清除现有数据
            if clear_existing and os.path.exists(self.working_dir):
                import shutil
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
                logger.info("🧹 已清理现有数据库")
            # 批量添加文档
            results = await self.add_documents(file_paths)
            # 检查结果
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            logger.info(f"📊 数据库重建完成: {success_count}/{total_count} 个文档成功处理")
            return success_count == total_count
        except Exception as e:
            logger.error(f"❌ 数据库重建失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """
        获取数据库信息
        Returns:
            Dict: 数据库信息
        """
        info = {
            "working_dir": self.working_dir,
            "output_dir": self.output_dir,
            "models": {
                "llm": self.llm_model,
                "embedding": self.embed_model,
                "vision": self.vision_model
            }
        }
        # 统计存储文件
        if os.path.exists(self.working_dir):
            files = os.listdir(self.working_dir)
            info["storage_files"] = files
            info["document_count"] = len([f for f in files if f.endswith('.json')])
        else:
            info["storage_files"] = []
            info["document_count"] = 0
        return info


# =======================
# 便捷函数接口
# =======================

async def add_document_to_rag(
    file_path: str,
    working_dir: str = "./rag_storage",
    output_dir: str = "./output",
    **kwargs
) -> bool:
    """
    向 RAG 数据库添加单个文档（异步接口）
    Args:
        file_path: 文档路径
        working_dir: 数据库存储目录
        output_dir: 输出目录
        **kwargs: 其他参数（llm_model, embed_model, vision_model, parser）
    Returns:
        bool: 是否成功添加
    """
    manager = RAGDatabaseManager(
        working_dir=working_dir,
        output_dir=output_dir,
        **kwargs
    )
    return await manager.add_document(file_path)


async def add_documents_to_rag(
    file_paths: List[str],
    working_dir: str = "./rag_storage",
    output_dir: str = "./output",
    **kwargs
) -> Dict[str, bool]:
    """
    批量向 RAG 数据库添加文档（异步接口）
    Args:
        file_paths: 文档路径列表
        working_dir: 数据库存储目录
        output_dir: 输出目录
        **kwargs: 其他参数
    Returns:
        Dict[str, bool]: 每个文档的处理结果
    """
    manager = RAGDatabaseManager(
        working_dir=working_dir,
        output_dir=output_dir,
        **kwargs
    )
    return await manager.add_documents(file_paths)


async def rebuild_rag_database(
    file_paths: List[str],
    working_dir: str = "./rag_storage",
    output_dir: str = "./output",
    clear_existing: bool = True,
    **kwargs
) -> bool:
    """
    重新构建 RAG 数据库（异步接口）
    Args:
        file_paths: 文档路径列表
        working_dir: 数据库存储目录
        output_dir: 输出目录
        clear_existing: 是否清除现有数据
        **kwargs: 其他参数
    Returns:
        bool: 是否成功重建
    """
    manager = RAGDatabaseManager(
        working_dir=working_dir,
        output_dir=output_dir,
        **kwargs
    )
    return await manager.rebuild_database(file_paths, clear_existing)


# =======================
# 同步接口
# =======================

def add_document_sync(
    file_path: str,
    working_dir: str = "./rag_storage",
    output_dir: str = "./output",
    **kwargs
) -> bool:
    """同步接口：添加单个文档"""
    return asyncio.run(add_document_to_rag(file_path, working_dir, output_dir, **kwargs))


def add_documents_sync(
    file_paths: List[str],
    working_dir: str = "./rag_storage",
    output_dir: str = "./output",
    **kwargs
) -> Dict[str, bool]:
    """同步接口：批量添加文档"""
    return asyncio.run(add_documents_to_rag(file_paths, working_dir, output_dir, **kwargs))


def rebuild_database_sync(
    file_paths: List[str],
    working_dir: str = "./rag_storage",
    output_dir: str = "./output",
    clear_existing: bool = True,
    **kwargs
) -> bool:
    """同步接口：重新构建数据库"""
    return asyncio.run(rebuild_rag_database(file_paths, working_dir, output_dir, clear_existing, **kwargs))


# =======================
# 使用示例
# =======================

async def main():
    # 创建数据库管理器
    db_manager = RAGDatabaseManager(
        working_dir="./rag_storage",
        output_dir="./output"
    )
    # 示例1: 添加单个文档
    # pdf_path = r"D:\adavance\tsy\rag4chat\test.pdf"
    # success = await db_manager.add_document(pdf_path)
    # # print(f"添加文档结果: {success}")

    # 示例2: 批量添加文档
    documents = [
        r"D:\adavance\tsy\rag4chat\knowladge2.docx",
        r"D:\adavance\tsy\rag4chat\knowladge3.docx",
        r"D:\adavance\tsy\rag4chat\knowladge4.docx",
        # r"D:\adavance\tsy\rag4chat\test.md" # 如果有其他文档
    ]
    results = await db_manager.add_documents(documents)
    print(f"批量添加结果: {results}")

    # # # 示例3: 获取数据库信息
    # info = db_manager.get_database_info()
    # print(f"数据库信息: {info}")

    # # 示例4: 重新构建数据库
    # rebuild_success = await db_manager.rebuild_database(documents, clear_existing=True)
    # print(f"重建数据库结果: {rebuild_success}")


if __name__ == "__main__":
    asyncio.run(main())