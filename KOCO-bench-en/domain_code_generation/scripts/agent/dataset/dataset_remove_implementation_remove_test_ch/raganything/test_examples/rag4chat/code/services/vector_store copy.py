# -*- coding: utf-8 -*-
"""
向量存储服务模块
"""
import os
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
from utils.decorators import error_handler, log_execution

from langchain_community.vectorstores import FAISS #1024维度
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import (
    EMBEDDING_MODEL, 
    EMBEDDING_BASE_URL, 
    MAX_RETRIEVED_DOCS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    向量存储服务类，用于管理文档向量存储
    """
    # 1. 初始化向量存储服务
    def __init__(self, index_dir: str = "faiss_index"):
        """
        index_dir - 索引目录
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.vector_store = None
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=EMBEDDING_BASE_URL)
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )
    #     print(EMBEDDING_MODEL, 
    #     EMBEDDING_BASE_URL, 
    #     MAX_RETRIEVED_DOCS,#3
    #     CHUNK_SIZE,#300
    #     CHUNK_OVERLAP,#30
    #     SEPARATORS)
    
    # 2. 更新嵌入模型
    def update_embedding_model(self, model_name: str) -> bool:
        """
        model_name - 新的嵌入模型名称

        @return 是否更新成功
        """
        try:
            if self.embeddings.model != model_name:
                self.embeddings = OllamaEmbeddings(model=model_name, base_url=EMBEDDING_BASE_URL)
                logger.info(f"嵌入模型已更新为: {model_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"更新嵌入模型失败: {str(e)}")
            return False
    
    # 3. 文本分块方法
    @error_handler()
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档进行分块处理
        
        @param documents - 原始文档列表
        @return 分块后的文档列表
        """
        try:
            # 使用文本分割器进行分块
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档分块完成：原始文档数量 {len(documents)}，分块后文档数量 {len(split_docs)}")
            return split_docs
        except Exception as e:
            logger.error(f"文档分块失败: {str(e)}")
            # 如果分块失败，返回原始文档
            return documents

    # 4. 创建全新的向量库实例，会覆盖原有数据
    @error_handler()
    def create_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """
        documents - 文档列表

        @return FAISS向量存储
        """
        if not documents:
            logger.warning("没有文档可以创建向量存储")
            return None
        
        logger.info(f"开始创建向量存储，原始文档数量: {len(documents)}")
        
        try:
            # 对文档进行分块
            split_documents = self.split_documents(documents)
            
            # 使用LangChain的FAISS向量存储
            self.vector_store = FAISS.from_documents(
                split_documents,
                self.embeddings
            )
            
            # 保存向量存储
            self._save_vector_store(self.vector_store)
            print(self.vector_store)
            
            logger.info(f"向量存储创建成功，包含 {len(split_documents)} 个文档块")
            
            return self.vector_store
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {str(e)}")
            return None
    
    # 5. 保存向量存储
    def _save_vector_store(self, vector_store: FAISS):
        """
        vector_store - FAISS向量存储
        """
        try:
            vector_store.save_local(str(self.index_dir))
            logger.info(f"向量存储已保存到: {self.index_dir}")
        except Exception as e:
            logger.error(f"保存向量存储失败: {str(e)}")
    
    # 6. 加载向量存储
    @error_handler()
    def load_vector_store(self) -> Optional[FAISS]:
        """
        @return FAISS向量存储
        """
        try:
            if (self.index_dir / "index.faiss").exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("向量存储加载成功")
                return self.vector_store
            logger.warning("向量存储文件不存在")
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
        return None
    

    # 7. 搜索相关文档
    @error_handler()
    def search_documents(self, query: str, threshold: float = 0.7) -> List[Document]:
        """
        query - 查询文本
        threshold - 相似度阈值

        @return 相关文档列表
        """
        if not self.vector_store:
            self.vector_store = self.load_vector_store()
            if not self.vector_store:
                logger.warning("向量存储未初始化")
                return []
        
        try:
            # 使用LangChain的相似度搜索
            docs_and_scores = self.vector_store.similarity_search_with_score(
                query,
                k=MAX_RETRIEVED_DOCS
            )
            
            # 根据阈值过滤结果
            results = [doc for doc, score in docs_and_scores if score > threshold]
            
            logger.info(f"搜索到 {len(results)} 个相关文档，相似度阈值: {threshold}")
            return results
            
        except Exception as e:
            logger.error(f"搜索文档失败: {str(e)}")
            return []
    
    # 8. 获取文档上下文
    def get_context(self, docs: List[Document]) -> str:
        """
        docs - 文档列表

        @return 合并后的上下文
        """
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)
    

    # 9. 添加单个文档到向量存储（保留现有向量库，仅追加新的文档向量）
    @error_handler()
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        添加单个文档到向量存储
        
        @param {str} content - 文档内容
        @param {Dict[str, Any]} metadata - 文档元数据
        @return {bool} 是否添加成功
        """
        if not content:
            logger.warning("文档内容为空，无法添加")
            return False
            
        try:
            # 创建Document对象
            doc = Document(page_content=content, metadata=metadata or {})
            
            # 对文档进行分块
            split_docs = self.split_documents([doc])
            
            # 如果向量存储不存在，先初始化
            if not self.vector_store:
                self.vector_store = self.load_vector_store()
                if not self.vector_store:
                    # 如果仍不存在，使用当前文档创建新的向量存储
                    self.vector_store = self.create_vector_store([doc])
                    return True
            
            # 为已存在的向量存储添加文档块
            self.vector_store.add_documents(split_docs)
            
            # 保存更新后的向量存储
            self._save_vector_store(self.vector_store)
            
            logger.info(f"成功添加文档，标题: {metadata.get('source', '未知') if metadata else '未知'}，分块数量: {len(split_docs)}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            return False
    

    # 10. 清除索引（删除所有索引文件）
    def clear_index(self):
        try:
            for file in self.index_dir.glob("*"):
                file.unlink()
            self.vector_store = None
            logger.info("索引已清除")
        except Exception as e:
            logger.error(f"清除索引失败: {str(e)}")
            raise 