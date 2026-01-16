# -*- coding: utf-8 -*-
# @Time    : 2025/08/06 10:00:00
# @Author  : nyzhhd
# @File    : get_response.py
# @Description: 智能体响应处理模块

from datetime import datetime
import logging
import re
import asyncio
from typing import Optional, AsyncGenerator
from config.settings import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    AVAILABLE_EMBEDDING_MODELS,
    MAX_HISTORY_TURNS
)
# RAGAgent: 用于处理用户输入和生成响应的智能体，封装模型交互逻辑。
from models.agent import RAGAgent
# ChatHistoryManager: 管理对话历史
from utils.chat_history import ChatHistoryManager
# DocumentProcessor: 处理用户上传的文档
from utils.document_processor import DocumentProcessor
# VectorStoreService: 向量数据库服务，用于文档索引与检索
from services.vector_store import VectorStoreService

from utils.decorators import error_handler, log_execution

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class agent_response:
    """
    RAG应用主类
    """
    def __init__(self):
        """
        @description 初始化应用
        """
        self.chat_history = ChatHistoryManager()  # 创建聊天历史管理器
        self.document_processor = DocumentProcessor()  # 创建文档处理器
        self.vector_store = VectorStoreService()  # 创建向量存储服务
        self.rag_enabled = True
        self.model = DEFAULT_MODEL
        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
        self.max_turns = MAX_HISTORY_TURNS
        self.human_content = None
        self.agent = RAGAgent(self.model) 
        logger.info("应用初始化成功")
    
    def update(self, rag_enabled , model, similarity_threshold, max_turns = 5):
        self.rag_enabled = rag_enabled
        
        if(self.model != model):
            self.model = model
            self.agent = RAGAgent(self.model)
            logger.info("更新模型成功")
        self.similarity_threshold = similarity_threshold
        self.max_turns = max_turns
        self.human_content = None
        logger.info("更新应用参数成功")
    
    async def process_user_input(self, prompt: str):
        """
        prompt - 用户输入的提示文本

        1️⃣ RAG模式：检索相关文档→获取上下文→调用模型
        2️⃣ 普通模式：直接调用模型
        """
        
        self.chat_history.add_message("user", prompt)  # 将用户消息添加到聊天历史
        if self.rag_enabled:
            await self._process_rag_query(prompt)  # 如果启用RAG，处理RAG查询
        else:
            await self._process_simple_query(prompt)  # 否则处理简单查询
    
    async def process_user_input_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        流式处理用户输入
        
        Args:
            prompt (str): 用户输入
            
        Yields:
            str: 响应的文本片段
        """
        self.chat_history.add_message("user", prompt)  # 将用户消息添加到聊天历史
        
        if self.rag_enabled:
            async for chunk in self._process_rag_query_stream(prompt):
                yield chunk
        else:
            async for chunk in self._process_simple_query_stream(prompt):
                yield chunk
    
    # 5. 处理RAG查询
    async def _process_rag_query(self, prompt: str):
        """
        prompt - 用户输入的提示文本
        """
        top_k = 3
        docs = await self.vector_store.search_documents(  
            prompt,
            top_k,
            self.similarity_threshold
        )
        logger.info(f"检索到的文档数: {len(docs)}")  
        # 获取文档上下文
        context = self.vector_store.get_context(docs)  
        # 创建RAG代理
         
        # 运行代理获取响应
        response = self.agent.run(  
            prompt, 
            context=context,
            history = self.chat_history.get_formatted_history(2),
            isapp = True
        )
        # 处理响应
        await self._process_response(response, docs)  
    
    # 5. 流式处理RAG查询
    async def _process_rag_query_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        流式处理RAG查询
        
        Args:
            prompt (str): 用户输入
            
        Yields:
            str: 响应的文本片段
        """
        top_k = 3
        docs = await self.vector_store.search_documents(  
            prompt,
            top_k,
            self.similarity_threshold
        )
        logger.info(f"检索到的文档数: {len(docs)}")  
        # 获取文档上下文
        context = self.vector_store.get_context(docs)
        history = self.chat_history.get_formatted_history(2)
         
        # 流式运行代理获取响应
        full_response = ""
        async for chunk in self.agent.run_stream(
            prompt, 
            context=context,
            history=history,
            isapp=True
        ):
            full_response += chunk
            yield chunk            
            
        # 处理响应
        await self._process_response(full_response, docs)

    # 6. 处理简单查询
    async def _process_simple_query(self, prompt: str):
        """
        prompt - 用户输入的提示文本
        """
        history = self.chat_history.get_formatted_history(2)
        response = self.agent.run(prompt, context = None, history = history, isapp = True)  
        # 处理响应
        await self._process_response(response)  
    
    # 6. 流式处理简单查询
    async def _process_simple_query_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        流式处理简单查询
        
        Args:
            prompt (str): 用户输入
            
        Yields:
            str: 响应的文本片段
        """
        history = self.chat_history.get_formatted_history(2)
        
        # 流式运行代理获取响应
        full_response = ""
        async for chunk in self.agent.run_stream(prompt, context=None, history=history, isapp=True):
            full_response += chunk
            yield chunk
            
        # 处理响应
        await self._process_response(full_response)
    
    # 7. 处理Agent的响应
    async def _process_response(self, response: str, docs=None):
        """
        response - 模型的原始响应
        docs - 检索到的文档（可选）
        """
        # 7.1 处理响应中的思考过程
        think_pattern = r'<think>([\s\S]*?)</think>'  # 定义思考过程的正则表达式模式
        think_match = re.search(think_pattern, response)  # 搜索思考过程
        if think_match:
            think_content = think_match.group(1).strip()  # 提取思考内容
            response_wo_think = re.sub(think_pattern, '', response).strip()  # 移除思考部分
        else:
            think_content = None
            response_wo_think = response
        
        human_pattern = r'<human>([\s\S]*?)</human>'  # 定义人类的正则表达式模式
        human_match = re.search(human_pattern, response_wo_think)  # 搜索human标签
        if not human_match:
            human_pattern = r'human>([\s\S]*?)</human>'  # 定义人类的正则表达式模式
            human_match = re.search(human_pattern, response_wo_think)  # 搜索human标签
        # if not human_match:
        #     human_pattern = r'human([\s\S]*?)human'  # 定义人类的正则表达式模式
        #     human_match = re.search(human_pattern, response_wo_think)  # 搜索human标签
        if human_match:
            human_content = human_match.group(1).strip()  # 提取human内容
            response_wo_think = re.sub(human_pattern, '', response_wo_think).strip()  # 移除human标签
        else:
            human_content = None
            response_wo_think = response_wo_think
        
        # 关键：正确设置human_content属性
        if human_content:
            self.human_content = human_content
        else:
            self.human_content = None  # 确保设置为None而不是保持旧值
        
        # 7.2 保存响应到历史
        self.chat_history.add_message("assistant", response_wo_think)  # 添加助手回复
        if think_content:
            self.chat_history.add_message("assistant_think", think_content)  # 添加思考过程
        if docs:
            doc_contents = [doc.get('content', '') for doc in docs] # 使用 .get() 避免 KeyError
            # doc_contents = [doc.page_content for doc in docs]  # 提取文档内容
            self.chat_history.add_message("retrieved_doc", doc_contents)  # 添加检索到的文档
        
        # 保存最后的响应，供外部获取
        self.last_response = response_wo_think

    # 获取最后的响应
    def get_last_response(self):
        """获取最后一次的AI响应"""
        return getattr(self, 'last_response', "抱歉，我没有理解您的问题。")
    
    # 清除响应历史
    def clear_response(self):
        """清除响应历史"""
        if hasattr(self, 'last_response'):
            delattr(self, 'last_response')
    
    # 获取历史记录
    def get_history(self):
        """获取历史记录"""
        return self.chat_history.get_formatted_history(self.max_turns)
    
    def get_history_summary(self):
        """获取历史记录简介"""
        response = self.agent.summary_chat_histroy(self.get_history())
        # print("history: ",response)
        think_pattern = r'<think>([\s\S]*?)</think>'  # 定义思考过程的正则表达式模式
        think_match = re.search(think_pattern, response)  # 搜索思考过程
        if think_match:
            think_content = think_match.group(1).strip()  # 提取思考内容
            summary_histroy = re.sub(think_pattern, '', response).strip()  # 移除思考部分
        else:
            think_content = None
            summary_histroy = response
        return summary_histroy

    # 同步运行方法
    async def run(self, prompt): 
        if prompt:
            await self.process_user_input(prompt)
            return self.get_last_response()
        return "请输入有效的问题"
    
    # 异步流式运行方法
    async def run_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        异步流式运行应用
        
        Args:
            prompt (str): 用户输入
            
        Yields:
            str: 响应的文本片段
        """
        if prompt:
            async for chunk in self.process_user_input_stream(prompt):
                yield chunk
        else:
            yield "请输入有效的问题"
    
    # 判断是否需要转到人类,需要转人工就返回true，并且返回历史记录
    def is_to_human(self):
        if not self.human_content:
            return False, None
        else:
            return True, self.get_history_summary()
                
        
if __name__ == "__main__":
    app = agent_response()  # 创建应用实例
    response = asyncio.run(app.run('你好')) # 运行应用
    print(f"响应: {response}")