# -*- coding: utf-8 -*-
# @Time    : 2025/08/06 10:00:00
# @Author  : nyzhhd
# @File    : app.py
# @Description: 主应用文件

# streamlit run app.py --server.port 6006

import streamlit as st

# 在 app.py 最顶部（所有 import 之前）加入
# import os, tempfile
# # 指定一个你确定有读写权限的目录
# os.environ["STREAMLIT_TEMP_DIR"] = r"D:\streamlit_tmp"   # 任意有效路径
# # 如果目录不存在就自动建
# os.makedirs(os.environ["STREAMLIT_TEMP_DIR"], exist_ok=True)
# tempfile.tempdir = os.environ["STREAMLIT_TEMP_DIR"]

from datetime import datetime
import logging
import re
import asyncio
from config.settings import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    AVAILABLE_EMBEDDING_MODELS
)
# 假设这些模块存在并正确实现
from models.agent import RAGAgent
from utils.chat_history import ChatHistoryManager
from utils.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from utils.ui_components import UIComponents
from utils.decorators import error_handler, log_execution

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class App:
    """
    RAG应用主类
    """

    def __init__(self):
        """
        @description 初始化应用
        """
        self._init_session_state()  # 初始化会话状态
        self.chat_history = ChatHistoryManager()  # 创建聊天历史管理器
        self.document_processor = DocumentProcessor()  # 创建文档处理器
        self.vector_store = VectorStoreService()  # 创建向量存储服务
        logger.info("应用初始化成功")

    # 1. 初始化会话状态
    @error_handler(show_error=False)
    def _init_session_state(self):
        """确保所有 session_state 变量都被初始化"""
        defaults = {
            'model_version': DEFAULT_MODEL,
            'processed_documents': [],
            'similarity_threshold': DEFAULT_SIMILARITY_THRESHOLD,
            'rag_enabled': True,
            'embedding_model': EMBEDDING_MODEL,
            'thinking': False,  # 初始化 thinking 状态
            'user_input': "",   # 初始化输入框状态 (虽然不直接修改，但初始化是好习惯)
            '_input_to_process': "" # 用于暂存待处理的用户输入
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    # 2. 渲染侧边栏
    @error_handler()
    @log_execution
    def render_sidebar(self):
        # 更新模型选择和嵌入模型选择
        st.session_state.model_version, new_embedding_model = UIComponents.render_model_selection(
            AVAILABLE_MODELS,
            st.session_state.model_version,
            AVAILABLE_EMBEDDING_MODELS,
            st.session_state.embedding_model
        )

        # 检查嵌入模型是否更改
        previous_embedding_model = st.session_state.embedding_model
        st.session_state.embedding_model = new_embedding_model

        # 更新RAG设置
        st.session_state.rag_enabled, st.session_state.similarity_threshold = UIComponents.render_rag_settings(
            st.session_state.rag_enabled,
            st.session_state.similarity_threshold,
            DEFAULT_SIMILARITY_THRESHOLD
        )

        # 更新向量存储服务的嵌入模型
        if previous_embedding_model != st.session_state.embedding_model:
            if self.vector_store.update_embedding_model(st.session_state.embedding_model):
                # 如果向量存储已存在，则提示用户可能需要重新处理文档
                if len(st.session_state.processed_documents) > 0:
                    st.sidebar.info(
                        f"⚠️ 嵌入模型已更改为 {st.session_state.embedding_model}，您可能需要重新处理文档以使用新的嵌入模型。")

        # 渲染聊天统计
        UIComponents.render_chat_stats(self.chat_history)

    # 3. 渲染文档上传区域
    @error_handler()
    @log_execution
    def render_document_upload(self):
        all_docs, self.vector_store = UIComponents.render_document_upload(
            self.document_processor,
            self.vector_store,
            st.session_state.processed_documents
        )

    # 4. 处理用户输入 (启动思考流程)
    @error_handler()
    @log_execution
    async def _handle_user_input_and_thinking(self, prompt: str):
        """
        处理用户输入并设置思考状态
        """
        self.chat_history.add_message("user", prompt)  # 将用户消息添加到聊天历史
        st.session_state.thinking = True  # 设置思考状态
        st.session_state._input_to_process = prompt # 将输入暂存
        st.rerun()  # 刷新页面以显示思考状态

    # 5. 处理用户输入 (实际处理逻辑)
    @error_handler()
    @log_execution
    async def process_user_input(self, prompt: str):
        """
        prompt - 用户输入的提示文本

        1️⃣ RAG模式：检索相关文档→获取上下文→调用模型
        2️⃣ 普通模式：直接调用模型
        """
        # self.chat_history.add_message("user", prompt)  # 用户消息已在 _handle_user_input_and_thinking 中添加
        if st.session_state.rag_enabled:
            await self._process_rag_query(prompt)  # 如果启用RAG，处理RAG查询
        else:
            await self._process_simple_query(prompt)  # 否则处理简单查询

        # 处理完成后清除状态
        st.session_state.thinking = False
        st.session_state._input_to_process = ""
        st.rerun() # 刷新以显示回复

    # 6. 处理RAG查询
    @error_handler()
    @log_execution
    async def _process_rag_query(self, prompt: str):
        """
        prompt - 用户输入的提示文本
        """
        top_k = 3
        with st.spinner("🤔正在评估查询..."):
            # 搜索相关文档
            docs = await self.vector_store.search_documents(
                prompt,
                top_k,
                st.session_state.similarity_threshold
            )
            logger.info(f"检索到的文档数: {len(docs)}")
            # 获取文档上下文
            context = self.vector_store.get_context(docs)
            # 创建RAG代理
            agent = RAGAgent(st.session_state.model_version)
            # 运行代理获取响应
            response = agent.run(
                prompt,
                context=context
            )
            # 处理响应
            await self._process_response(response, docs)

    # 7. 处理简单查询
    @error_handler()
    @log_execution
    async def _process_simple_query(self, prompt: str):
        """
        prompt - 用户输入的提示文本
        """
        with st.spinner("🤖 思考中..."):
            # 创建RAG代理
            agent = RAGAgent(st.session_state.model_version)
            # 运行代理获取响应
            response = agent.run(prompt)
            # 处理响应
            await self._process_response(response)

    # 8. 处理Agent的响应
    async def _process_response(self, response: str, docs=None):
        """
        response - 模型的原始响应
        docs - 检索到的文档（可选）
        """
        # 8.1 处理响应中的思考过程
        think_pattern = r'<think>([\s\S]*?)</think>'  # 定义思考过程的正则表达式模式
        think_match = re.search(think_pattern, response)  # 搜索思考过程
        if think_match:
            think_content = think_match.group(1).strip()  # 提取思考内容
            response_wo_think = re.sub(think_pattern, '', response).strip()  # 移除思考部分
        else:
            think_content = None
            response_wo_think = response

        # 8.2 保存响应到历史
        self.chat_history.add_message("assistant", response_wo_think)  # 添加助手回复
        if think_content:
            self.chat_history.add_message("assistant_think", think_content)  # 添加思考过程
        if docs:
            doc_contents = [doc.get('content', '') for doc in docs]  # 使用 .get() 避免 KeyError
            self.chat_history.add_message("retrieved_doc", doc_contents)  # 添加检索到的文档

    # 入口处：运行应用
    @error_handler()
    @log_execution
    async def run(self):
        st.set_page_config(page_title="🐋 你的智能运维客服", layout="wide")
        UIComponents.inject_custom_css()  # 注入自定义CSS

        st.title("🐋 你的智能运维客服")  # 设置应用标题
        st.info("🤖 帮助你快速解决问题。")  # 显示模型信息

        self.render_sidebar()  # 渲染侧边栏
        self.render_document_upload()  # 渲染文档上传区域

        # --- 渲染聊天历史 (在输入框上方) ---
        UIComponents.render_chat_history(self.chat_history)

        # --- 渲染思考中提示 ---
        if st.session_state.thinking:
             # 使用一个独立的容器来显示思考提示，避免影响聊天历史的滚动
            with st.container():
                st.markdown('<div class="thinking-message">⏳ 正在思考中，请稍候...</div>', unsafe_allow_html=True)
                UIComponents.scroll_to_bottom() # 滚动到底部显示思考提示

        # --- 渲染底部输入区域 ---
        # 调用 UIComponents 的方法来渲染输入框，它会处理自身的交互
        UIComponents.render_input_area()
        # 注意：render_input_area 内部通过回调处理了输入提交和清空，我们在这里不需要直接获取返回值

        # --- 处理用户输入逻辑 ---
        # 1. 如果有待处理的输入且未在思考中，则启动处理流程
        pending_input = st.session_state.get('_input_to_process', '')
        if pending_input and not st.session_state.thinking:
            await self._handle_user_input_and_thinking(pending_input)

        # 2. 如果正在思考中，则继续处理
        elif st.session_state.thinking and pending_input:
             await self.process_user_input(pending_input)

        mode_description = ""
        if st.session_state.rag_enabled:
            mode_description += "📚 可以询问上传文档的内容。"
        else:
            mode_description += "💬 直接与模型交流。"
        mode_description += " 🔍 支持转人工。"
        mode_description += " 🌤️ 可以进行天气查询。"

        st.info(mode_description)  # 显示模式描述


if __name__ == "__main__":
    app = App()  # 创建应用实例
    asyncio.run(app.run())  # 运行应用
