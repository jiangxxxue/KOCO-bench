# -*- coding: utf-8 -*-
"""
向量存储服务模块 (重构版)
此版本使用 RAGDatabaseManager 来管理文档和索引，
并使用 query_and_find_topk 进行查询。
"""
import os
import shutil
from typing import List, Optional, Dict, Any
import logging
import concurrent.futures
import functools
# 导入新的管理器和查询函数
# 假设这些在同一个项目或可通过 PYTHONPATH 找到
# from .rag_database_manager import RAGDatabaseManager # 如果在包内
# from .query_topk import query_and_find_topk # 如果在包内
# 否则，根据实际文件结构调整导入路径，例如：
from services.build_database import RAGDatabaseManager  # 假设 build_database.py 在同一目录或已导入
from services.get_top_from_rag import query_and_find_topk # 假设 get_top_from_rag.py 在同一目录或已导入

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    向量存储服务类，用于管理文档向量存储 (重构版)
    该版本委托 RAGDatabaseManager 处理存储和索引，使用 query_and_find_topk 进行查询。
    """

    # 1. 初始化向量存储服务
    def __init__(self, index_dir: str = "faiss_index", rag_working_dir: str = "./rag_storage"):
        """
        index_dir - 为了兼容性保留，但主要使用 rag_working_dir
        rag_working_dir - RAG 系统的工作目录，包含 vdb_chunks.json 等
        """
        self.rag_working_dir = rag_working_dir
        self.index_dir = index_dir  # 保留以兼容旧接口，但实际不用
        self.vdb_chunks_path = os.path.join(self.rag_working_dir, "vdb_chunks.json")

        # 初始化 RAGDatabaseManager
        self.db_manager = RAGDatabaseManager(working_dir=self.rag_working_dir)
        # 添加一个属性来指示服务是否已准备好 (例如，索引文件是否存在)
        self.is_ready = False
        # 在初始化时检查一次
        self.load_vector_store()
        logger.info(f"VectorStoreService initialized with RAG working dir: {self.rag_working_dir}")

    # 2. 更新嵌入模型 (此版本不再需要，因为由 RAGDatabaseManager 内部管理)
    def update_embedding_model(self, model_name: str) -> bool:
        """
        此版本不直接支持动态更新嵌入模型。
        模型配置应在 RAGDatabaseManager 初始化时确定。
        @return 始终返回 False，表示未执行更新。
        """
        logger.warning("update_embedding_model is not supported in this version. Configure model in RAGDatabaseManager.")
        return False

    # 3. 文本分块方法 (此版本不再需要，因为由 RAGDatabaseManager 内部处理)
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        此版本不直接提供文本分块功能。
        分块由 RAGDatabaseManager 在处理文档时内部完成。
        @return 直接返回原始 documents。
        """
        logger.warning("split_documents is handled internally by RAGDatabaseManager.")
        return documents

    # 4. 创建全新的向量库实例 (使用 RAGDatabaseManager)
    async def create_vector_store(self, document_paths: List[str]) -> bool:
        """
        使用 RAGDatabaseManager 重新构建向量库。
        document_paths - 本地文档文件路径列表 (PDF, DOCX, etc.)
        @return 是否创建/重建成功
        """
        if not document_paths:
            logger.warning("没有文档路径可以创建向量存储")
            # 更新状态
            self.is_ready = False
            return False

        logger.info(f"开始通过 RAGDatabaseManager 重新构建向量存储，文档数量: {len(document_paths)}")

        try:
            # 使用 RAGDatabaseManager 重建数据库
            # 这会清空现有数据并添加新文档
            success = await self.db_manager.add_documents(document_paths)

            if success:
                logger.info("向量存储 (通过 RAGDatabaseManager) 重建成功")
                # 更新状态
                self.is_ready = True
            else:
                logger.error("向量存储 (通过 RAGDatabaseManager) 重建失败")
                # 更新状态
                self.is_ready = False
            return success

        except Exception as e:
            logger.error(f"通过 RAGDatabaseManager 创建/重建向量存储失败: {str(e)}", exc_info=True)
            # 更新状态
            self.is_ready = False
            return False

    # 5. 保存向量存储 (此功能由 RAGDatabaseManager 内部自动处理)
    def _save_vector_store(self):
        """
        此版本中，保存由 RAGDatabaseManager 自动处理。
        """
        logger.info("向量存储的保存由 RAGDatabaseManager 自动处理。")

    # 6. 加载向量存储 (检查文件存在性并设置 is_ready)
    def load_vector_store(self) -> bool:
        """
        检查向量存储文件是否存在，以确定是否已加载或可加载。
        @return bool: 如果检测到向量存储文件则返回 True，否则返回 False。
        """
        # 检查核心的 vdb_chunks.json 文件是否存在
        if os.path.exists(self.vdb_chunks_path):
            logger.info(f"检测到向量存储文件: {self.vdb_chunks_path}")
            # 更新状态
            self.is_ready = True
            return True  # 返回 True 表示已准备好
        else:
            logger.warning(f"未检测到向量存储文件: {self.vdb_chunks_path}")
            # 更新状态
            self.is_ready = False
            return False  # 返回 False 表示未准备好

    # 7. 搜索相关文档 (核心功能，使用新的 query_and_find_topk)
    async def search_documents(self, query: str, top_k: int = 3, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        query - 查询文本
        top_k - 返回的 top K 个结果
        threshold - 相似度阈值 (在 query_and_find_topk 中可能需要调整处理)

        @return 相关文档信息列表，每个元素是一个字典，包含 'content', 'file_path', 'similarity' 等
                如果发生错误或未找到，则返回空列表。
        """
        # 检查向量存储是否已准备好
        if not self.is_ready:
            logger.warning("向量存储未准备好，无法进行搜索")
            return []

        # 检查向量存储文件是否存在（双重保险）
        if not os.path.exists(self.vdb_chunks_path):
            logger.warning("向量存储文件不存在，无法进行搜索")
            # 状态可能不同步，更新一下
            self.is_ready = False
            return []

        try:
            logger.info(f"开始搜索与 '{query}' 相关的文档...")

            # --- 关键：调用新的查询函数 ---
            # 注意：这要求 query_and_find_topk 已被修改以返回结构化数据
            # 而不是仅仅打印结果。
            # 如果 query_and_find_topk 仍然只打印，请参考之前的回答修改它。
            results = await query_and_find_topk(query, self.vdb_chunks_path, top_k)
            # print(results)

            # 确保返回的是列表
            if not isinstance(results, list):
                 logger.error(f"query_and_find_topk 返回了意外的类型: {type(results)}")
                 return []

            # 根据阈值过滤结果 (如果需要)
            # 注意：query_and_find_topk 内部计算的是余弦相似度，范围 [0, 1] (通常)
            # threshold=0.0 意味着返回所有结果
            if threshold > 0.0:
                filtered_results = [res for res in results if res.get('similarity', 1.0) >= threshold]
                logger.info(f"搜索完成，返回 {len(filtered_results)} 个相关文档 (阈值: {threshold})")
                return filtered_results
            else:
                logger.info(f"搜索完成，返回 {len(results)} 个相关文档")
                return results

        except Exception as e:
            logger.error(f"搜索文档失败: {str(e)}", exc_info=True)
            return []

    # 8. 获取文档上下文 (保持不变，处理 search_documents 的输出)
    def get_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        docs - 由 search_documents 返回的文档字典列表

        @return 合并后的上下文
        """
        if not docs:
            return ""
        # 假设每个 doc 字典都有 'content' 键
        return "\n\n".join(doc.get('content', '') for doc in docs)

    # 9. 添加单个文档到向量存储 (使用 RAGDatabaseManager)
    async def add_document(self, document_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        添加单个文档到向量存储 (通过 RAGDatabaseManager)

        @param {str} document_path - 文档文件路径 (PDF, DOCX, etc.)
        @param {Dict[str, Any]} metadata - 文档元数据 (可选，由 RAGDatabaseManager 处理)
        @return {bool} 是否添加成功
        """
        if not document_path or not os.path.exists(document_path):
            logger.warning(f"文档路径无效或文件不存在: '{document_path}'")
            return False

        try:
            logger.info(f"开始添加文档: {document_path}")

            # 使用 RAGDatabaseManager 添加单个文档
            # 注意：确保 RAGDatabaseManager.add_document 返回布尔值
            success = await self.db_manager.add_document(document_path)

            if success:
                logger.info(f"成功添加文档: {document_path}")
                # 添加文档后，索引应该已更新，标记为就绪
                self.is_ready = True
            else:
                logger.error(f"添加文档失败: {document_path}")
            return success

        except Exception as e:
            logger.error(f"添加文档 '{document_path}' 失败: {str(e)}", exc_info=True)
            return False

    # 10. 清除索引 (使用 RAGDatabaseManager 的逻辑或直接删除文件)
    def clear_index(self):
        """
        清除索引 (通过 RAGDatabaseManager 或直接文件操作)
        """
        try:
            # 方法一：直接删除 rag_working_dir 下的相关文件
            # 这是最直接和彻底的方法
            if os.path.exists(self.rag_working_dir):
                shutil.rmtree(self.rag_working_dir)
                logger.info(f"索引目录 '{self.rag_working_dir}' 已清除")
                # 重新创建空目录
                os.makedirs(self.rag_working_dir, exist_ok=True)
                # 重新初始化 db_manager 实例
                self.db_manager = RAGDatabaseManager(working_dir=self.rag_working_dir)
                logger.info("已重新初始化 RAGDatabaseManager")
            else:
                logger.warning(f"索引目录 '{self.rag_working_dir}' 不存在")

            # 清除后，状态变为未准备好
            self.is_ready = False

        except Exception as e:
            logger.error(f"清除索引失败: {str(e)}", exc_info=True)
            # 即使出错，也可能认为索引状态不确定，设为 False
            self.is_ready = False
            raise # 重新抛出异常以便调用者处理

    # --- 兼容性属性 (供旧版 UI 检查) ---
    @property
    def vector_store(self):
        """
        兼容性属性：为旧版 UI 提供检查点。
        如果服务已准备好 (is_ready=True)，则返回一个非 None 的占位符对象。
        否则返回 None。
        """
        # 注意：这个占位符对象不应该被实际调用其方法
        # 它仅仅是为了让 `if not vector_store.vector_store:` 这样的检查通过或失败
        if self.is_ready:
            # 返回一个简单的非 None 对象作为占位符
            return lambda: None # 或者 object() 或 type('Placeholder', (), {})()
        else:
            return None

# --- 示例用法 (如果需要直接运行此类) ---
# 注意：这需要 query_and_find_topk 返回结构化数据
"""
import asyncio
import os

async def example_usage():
    # --- 初始化服务 ---
    vss = VectorStoreService(rag_working_dir="./rag_storage")

    # --- 检查/加载索引 ---
    is_loaded = vss.load_vector_store() # 由内部处理，这里检查返回值
    print(f"Vector Store Loaded: {is_loaded}")

    # --- 添加文档 (如果索引为空或需要更新) ---
    # documents_to_add = [
    #     r"D:\adavance\tsy\rag4chat\test.pdf",
    #     # r"D:\adavance\tsy\rag4chat\test2.docx"
    # ]
    # success = await vss.create_vector_store(documents_to_add) # 重建
    # print(f"Create Vector Store Success: {success}")
    # 或者添加单个文档
    # success = await vss.add_document(r"D:\adavance\tsy\rag4chat\test.pdf")
    # print(f"Add Document Success: {success}")

    # --- 查询 ---
    query_text = "语义图像如何构建？"
    topk = 5
    threshold = 0.1 # 设置一个合理的阈值

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 环境变量 DASHSCOPE_API_KEY 未设置。请设置后重试。")
        return

    print(f"🔍 正在搜索与 '{query_text}' 相关的文档...")
    results = await vss.search_documents(query_text, top_k=topk, threshold=threshold)

    print(f"📄 搜索结果 ({len(results)} 个):")
    for i, res in enumerate(results):
        print(f"--- 结果 {i+1} (相似度: {res.get('similarity', 'N/A'):.4f}) ---")
        print(f"  内容预览: {res.get('content', 'N/A')[:100]}...")
        print(f"  文件路径: {res.get('file_path', 'N/A')}")
        print(f"  Chunk ID: {res.get('chunk_id', 'N/A')}")
        print("-" * 20)

    context = vss.get_context(results)
    print(f"🧾 合并上下文预览:\\n{context[:200]}...")

if __name__ == "__main__":
    asyncio.run(example_usage())
"""