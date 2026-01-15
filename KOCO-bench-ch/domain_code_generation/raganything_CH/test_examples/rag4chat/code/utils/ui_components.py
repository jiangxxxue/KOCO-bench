"""
UI组件模块，包含所有Streamlit UI渲染逻辑
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Any
import logging
from utils.document_processor import DocumentProcessor # 假设存在
from services.vector_store import VectorStoreService   # 假设存在
from langchain.schema import Document                  # 假设存在
from config.settings import AVAILABLE_EMBEDDING_MODELS # 假设存在
import concurrent.futures
import functools
import asyncio
import base64
import re  # 用于正则表达式匹配图片路径

logger = logging.getLogger(__name__)

def convert_local_images_to_base64(markdown_text: str) -> str:
    """
    遍历 Markdown 文本，查找本地图片路径，并将其替换为 Base64 嵌码。
    注意：此函数假设图片路径是 Windows 风格的绝对路径。
    """
    # 定义一个内部函数来处理图片转换
    def _replace_image_path(match):
        alt_text = match.group(1)  # 获取 alt text
        image_path = match.group(2)  # 获取图片路径

        # 简单检查是否为本地路径 (可以根据需要调整判断逻辑)
        # 这里假设包含盘符（如 D:\）或以 \ 或 / 开头的是本地路径
        if re.match(r'^[A-Za-z]:[\\\/]|^[\\\/]', image_path):
            try:
                # 读取图片文件并转换为 Base64
                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

                # 简单推断 MIME 类型 (可以根据文件扩展名更精确地判断)
                # 这里只处理常见的 JPEG 和 PNG
                mime_type = "image/jpeg"
                if image_path.lower().endswith(".png"):
                    mime_type = "image/png"
                elif image_path.lower().endswith(".gif"):
                    mime_type = "image/gif"
                # 可以根据需要添加更多类型

                # 返回新的 Markdown 图片标签 (使用 Base64)
                return f'![{alt_text}](data:{mime_type};base64,{encoded_string})'
            except FileNotFoundError:
                # 如果文件未找到，返回原始 Markdown 或一个错误提示
                st.warning(f"图片文件未找到，无法嵌入: {image_path}")
                return match.group(0)  # 返回原始匹配内容
            except Exception as e:
                # 处理其他可能的错误（如读取权限问题）
                st.error(f"处理图片 '{image_path}' 时出错: {e}")
                return match.group(0)  # 返回原始匹配内容
        else:
            # 如果不是本地路径（例如已经是 URL），则不修改
            return match.group(0)

    # 使用正则表达式查找所有 Markdown 图片语法 ![alt](path)
    # 这个正则表达式会匹配 ![...](...) 并捕获 alt text 和 path
    pattern = r'!\[(.*?)\]\((.*?)\)'
    # 使用 re.sub 和回调函数 _replace_image_path 来替换匹配项
    corrected_markdown_text = re.sub(pattern, _replace_image_path, markdown_text)

    return corrected_markdown_text


def run_async_in_thread(coro):
    """
    在一个新线程中运行协程，并等待其完成。
    这避免了在已有事件循环（如 Streamlit 的）中直接操作循环的问题。
    返回协程的结果或引发异常。
    """
    def _run_in_thread():
        # 在新线程中创建并运行一个新的事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 在这个新循环中运行协程直到完成
            return loop.run_until_complete(coro)
        finally:
            # 清理：关闭新创建的循环
            loop.close()
            asyncio.set_event_loop(None)  # 重置线程的事件循环

    # 使用 ThreadPoolExecutor 在新线程中执行 _run_in_thread 函数
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务并等待结果
        future = executor.submit(_run_in_thread)
        # 阻塞等待结果（这发生在主线程，但不会阻塞 Streamlit 的事件循环太久，因为工作在后台线程）
        # 如果协程内部有异常，future.result() 会重新抛出它
        return future.result()


class UIComponents:
    """UI组件类，封装了所有Streamlit UI渲染逻辑"""

    @staticmethod
    def inject_custom_css():
        """注入自定义CSS样式"""
        st.markdown("""
            <style>
            /* 全局样式 */
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            /* 聊天容器样式 */
            .chat-container {
                height: calc(100vh - 250px); /* 调整高度以适应输入框 */
                overflow-y: auto;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
                margin-bottom: 20px;
            }
            
            /* 消息气泡容器 */
            .message-bubble {
                display: flex;
                align-items: flex-start;
                margin: 15px 0;
                animation: fadeIn 0.3s ease-in;
            }
            
            /* 用户消息 */
            .user-message {
                flex-direction: row-reverse;
            }
            
            /* 助手消息 */
            .assistant-message {
                flex-direction: row;
            }
            
            /* 系统消息 */
            .system-message {
                justify-content: center;
                width: 100%;
            }
            
            /* 头像样式 */
            .avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                margin: 0 10px;
                flex-shrink: 0;
            }
            
            .user-avatar {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
            }
            
            .assistant-avatar {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: white;
            }
            
            .system-avatar {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #333;
            }
            
            /* 消息内容样式 */
            .message-content {
                max-width: 70%;
                padding: 15px 20px;
                border-radius: 20px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                line-height: 1.5;
                position: relative;
                word-wrap: break-word;
            }
            
            .user-message .message-content {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                border-bottom-right-radius: 5px;
            }
            
            .assistant-message .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-bottom-left-radius: 5px;
            }
            
            .system-message .message-content {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                color: #333;
                text-align: center;
                font-size: 14px;
                max-width: 80%;
            }
            
            /* 思考中消息样式 */
            .thinking-message {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 15px 20px;
                border-radius: 20px;
                margin: 15px 0;
                text-align: center;
                animation: pulse 2s infinite;
                max-width: 300px;
                margin: 15px auto;
            }
            
            /* 底部输入区域样式 */
            .input-container {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(255, 255, 255, 0.95);
                padding: 20px;
                backdrop-filter: blur(10px);
                border-top: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.1);
                z-index: 1000;
            }
            
            /* 输入框样式 */
            .stTextInput > div > div > input {
                padding: 15px 20px;
                border-radius: 25px;
                border: 2px solid #e0e0e0;
                font-size: 16px;
                transition: border-color 0.3s ease;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: #4facfe;
                box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
            }
            
            /* 发送按钮样式 */
            .stButton > button {
                height: 50px;
                border-radius: 25px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                font-weight: bold;
                transition: transform 0.2s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            /* 滚动条样式 */
            .chat-container::-webkit-scrollbar {
                width: 8px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
            }
            
            .chat-container::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 255, 255, 0.5);
            }
            
            /* 动画效果 */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.7; }
                100% { opacity: 1; }
            }
            
            /* 侧边栏样式 */
            [data-testid="stSidebar"] {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
            }
            
            [data-testid="stSidebar"] .stMarkdown {
                color: white;
            }
            
            /* 扩展器样式 */
            .streamlit-expanderHeader {
                background: rgba(255, 255, 255, 0.1) !important;
                border-radius: 10px !important;
                margin: 5px 0 !important;
            }
            
            .streamlit-expanderContent {
                background: rgba(255, 255, 255, 0.05) !important;
                border-radius: 10px !important;
                padding: 15px !important;
            }

            /* 无消息提示 */
            .no-messages {
                text-align: center;
                color: #999;
                font-style: italic;
                margin-top: 50px;
            }
                    
            .chat-container:empty {
                 height: auto;
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def scroll_to_bottom():
        """自动滚动到底部"""
        st.components.v1.html(
            """
            <script>
                var chatContainer = parent.document.querySelector('.chat-container');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            </script>
            """,
            height=0
        )

    # 1.渲染模型选择组件
    @staticmethod
    def render_model_selection(available_models: List[str], current_model: str, embedding_models: List[str],
                               current_embedding_model: str) -> Tuple[str, str]:
        """
        available_models - 可用模型列表
        current_model - 当前选中的模型
        embedding_models - 可用嵌入模型列表
        current_embedding_model - 当前选中的嵌入模型

        @return (用户选择的模型, 用户选择的嵌入模型)
        """
        st.sidebar.header("⚙️ 设置")

        new_model = st.sidebar.selectbox(
            "选择模型",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="选择要使用的语言模型"
        )

        new_embedding_model = st.sidebar.selectbox(
            "嵌入模型",
            options=embedding_models,
            index=embedding_models.index(current_embedding_model) if current_embedding_model in embedding_models else 0,
            help="选择用于文档嵌入的模型"
        )

        return new_model, new_embedding_model

    # 2. 渲染RAG设置组件
    @staticmethod
    def render_rag_settings(rag_enabled: bool, similarity_threshold: float, default_threshold: float) -> Tuple[bool, float]:
        """
        rag_enabled - 是否启用RAG
        similarity_threshold - 相似度阈值
        default_threshold - 默认相似度阈值

        @return (是否启用RAG, 相似度阈值)
        """
        st.sidebar.subheader("RAG设置")

        new_rag_enabled = st.sidebar.checkbox(
            "启用RAG",
            value=rag_enabled,
            help="启用检索增强生成功能，使用上传的文档增强回答"
        )

        new_similarity_threshold = st.sidebar.slider(
            "相似度阈值",
            min_value=0.0,
            max_value=1.0,
            value=similarity_threshold,
            step=0.05,
            help="调整检索相似度阈值，值越高要求匹配度越精确"
        )

        # 将重置相似度阈值按钮样式更改为容器宽度
        if st.sidebar.button("重置相似度阈值", use_container_width=True):
            new_similarity_threshold = default_threshold

        return new_rag_enabled, new_similarity_threshold

    # 3. 渲染聊天统计信息
    @staticmethod
    def render_chat_stats(chat_history):
        """
        chat_history - 聊天历史管理器
        """
        st.sidebar.header("💬 对话历史")
        stats = chat_history.get_stats()
        st.sidebar.info(f"总对话数: {stats['total_messages']} 用户消息: {stats['user_messages']}")

        if st.sidebar.button("📥 导出对话历史", use_container_width=True):
            csv = chat_history.export_to_csv()
            if csv:
                st.sidebar.download_button(
                    label="下载CSV文件",
                    data=csv,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # 假设导出服务存在
        try:
            from services.export_file import mdcontent2docx, mdcontent2pdf
            if st.sidebar.button("✅ 导出当前内容为PDF", use_container_width=True):
                last_assistant_message = None
                for message in reversed(chat_history.history): # 从后往前找更高效
                    if message.get('role') == 'assistant':
                        last_assistant_message = message.get('content', '')
                        break
                if last_assistant_message:
                    mdcontent2pdf(last_assistant_message, './now_content.pdf')
                    st.sidebar.success("PDF导出成功!")
                    # st.rerun() # 通常不需要rerun来显示成功消息

            if st.sidebar.button("🚀 导出当前内容为DOCX", use_container_width=True):
                last_assistant_message = None
                for message in reversed(chat_history.history):
                    if message.get('role') == 'assistant':
                        last_assistant_message = message.get('content', '')
                        break
                if last_assistant_message:
                    mdcontent2docx(last_assistant_message, './now_content.docx')
                    st.sidebar.success("DOCX导出成功!")
                    # st.rerun()
        except ImportError:
             st.sidebar.warning("导出服务未配置")

        if st.sidebar.button("✨ 清空对话", use_container_width=True):
            chat_history.clear_history()
            st.rerun()

    # 4. 渲染文档上传组件
    @staticmethod
    def render_document_upload(
            document_processor: DocumentProcessor,
            vector_store: VectorStoreService,
            processed_documents: List[str]
    ) -> Tuple[List[Document], VectorStoreService]:
        """
        document_processor - 文档处理器
        vector_store - 向量存储服务
        processed_documents - 已处理文档列表

        @return (all_docs, vector_store)
        """
        with st.expander("📁 上传用于构建知识库的分析文档", expanded=not bool(processed_documents)):
            uploaded_files = st.file_uploader(
                "上传PDF、TXT、DOCX、MD文件",
                type=["pdf", "txt", "docx", "md"],
                accept_multiple_files=True
            )

            if not vector_store.vector_store:
                st.warning("⚠️ 请在侧边栏配置向量存储以启用文档处理。")

            all_docs = []
            if uploaded_files:
                if st.button("处理文档"):
                    with st.spinner("正在处理文档..."):
                        for uploaded_file in uploaded_files:
                            if uploaded_file.name not in processed_documents:
                                try:
                                    # 统一处理所有文件类型
                                    print(uploaded_file)
                                    result = document_processor.process_file(uploaded_file)

                                    if isinstance(result, list):
                                        # 结果是Document列表(PDF文档)
                                        all_docs.extend(result)
                                    else:
                                        # 结果是文本内容(TXT、DOCX等)
                                        doc = Document(
                                            page_content=result,
                                            metadata={"source": uploaded_file.name}
                                        )
                                        all_docs.append(doc)

                                    processed_documents.append(uploaded_file.name)
                                    st.success(f"✅ 已处理: {uploaded_file.name}")
                                except Exception as e:
                                    st.error(f"❌ 处理失败: {uploaded_file.name} - {str(e)}")
                            else:
                                st.warning(f"⚠️ 已存在: {uploaded_file.name}")

                    if all_docs:
                        document_paths = []
                        for doc in all_docs:
                            # --- 关键修改：从 Document 对象中提取路径 ---
                            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                                source_path = doc.metadata.get('source')
                                if source_path and isinstance(source_path, str):
                                    document_paths.append(source_path)
                                else:
                                    st.warning(f"⚠️ 无法从文档对象获取文件路径: {doc}")
                                    logger.warning(f"无法从文档对象获取文件路径: {doc}")
                            else:
                                st.warning(f"⚠️ 文档对象缺少有效的 metadata: {doc}")
                                logger.warning(f"文档对象缺少有效的 metadata: {doc}")
                        # --- 新增结束 ---

                        if not document_paths:
                            st.error("❌ 没有有效的文档路径可供处理。")
                            logger.error("没有有效的文档路径可供处理。")
                            return all_docs, vector_store # 返回当前状态
                        with st.spinner("正在构建向量索引..."):
                            success = run_async_in_thread(vector_store.create_vector_store(document_paths))

            # 显示已处理文档列表
            if processed_documents:
                st.subheader("已处理文档")
                for doc in processed_documents:
                    st.markdown(f"- {doc}")

                if st.button("清除所有文档"):
                    with st.spinner("正在清除向量索引..."):
                        vector_store.clear_index()
                        processed_documents.clear()
                    st.success("✅ 所有文档已清除")
                    st.rerun()

            return all_docs, vector_store

    # 5. 渲染聊天历史
    @staticmethod
    def render_chat_history(chat_history):
        """
        chat_history - 聊天历史管理器
        """
        # 创建聊天容器
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)

            # 渲染每条消息
            for message in chat_history.history:
                role = message.get('role', '')
                content = message.get('content', '')

                if role == "assistant_think":
                    with st.expander("💡 查看推理过程 <think> ... </think>"):
                        st.markdown(content)
                elif role == "retrieved_doc":
                    with st.expander(f"📖 查看本次召回的文档块", expanded=False):
                        if isinstance(content, list):
                            for idx, doc in enumerate(content, 1):
                                st.markdown(f"**文档块{idx}:**\n{doc}")
                        else:
                            st.markdown(content)
                else:
                    corrected_markdown_content = convert_local_images_to_base64(content)

                    # 根据角色渲染不同样式的消息
                    if role == "user":
                        st.markdown(f'''
                            <div class="message-bubble user-message">
                                <div class="avatar user-avatar">👤</div>
                                <div class="message-content">{corrected_markdown_content}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    elif role == "assistant":
                        st.markdown(f'''
                            <div class="message-bubble assistant-message">
                                <div class="avatar assistant-avatar">🤖</div>
                                <div class="message-content">{corrected_markdown_content}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    elif role == "system":
                        st.markdown(f'''
                            <div class="message-bubble system-message">
                                <div class="avatar system-avatar">ℹ️</div>
                                <div class="message-content">{corrected_markdown_content}</div>
                            </div>
                        ''', unsafe_allow_html=True)
                    # 注意：role == "thinking" 的消息由 app.py 的 run 方法直接渲染，不在这里处理

            # 如果聊天历史为空，显示提示信息
            if not chat_history.history:
                st.markdown('<div class="no-messages">暂无聊天记录</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # 自动滚动到底部
        UIComponents.scroll_to_bottom()

    # 6. 渲染输入区域（使用回调函数实现清空）
    @staticmethod
    def render_input_area() -> None:
        """
        渲染底部输入框。
        使用回调函数处理发送逻辑，避免直接修改 session_state 导致的错误。
        """
        # --- 定义回调函数 ---
        def on_send_click():
            """
            当发送按钮被点击时调用的回调函数。
            它会将当前输入保存到 session_state 的一个临时变量中，
            然后清空输入框。
            """
            # 1. 获取当前输入框的值
            current_input = st.session_state.get("user_input", "").strip()

            # 2. 如果输入不为空，则保存到临时变量 `_input_to_process`
            # app.py 会从这个变量读取待处理的输入
            if current_input:
                st.session_state._input_to_process = current_input

            # 3. 清空输入框 (这是安全的，因为在回调函数中)
            st.session_state.user_input = ""

        # --- 渲染 UI ---
        # 创建固定在底部的输入区域
        input_container = st.container()
        with input_container:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)

            # 使用columns来更好地控制布局
            col1, col2 = st.columns([4, 1])
            with col1:
                # 修复 label 为空的警告 - 提供一个描述性的 label
                st.text_input(
                    "用户输入框",  # <-- 非空 label
                    placeholder="请输入您的问题...",
                    label_visibility="collapsed",  # 隐藏 label 但保留其可访问性
                    key="user_input"  # 关键：为 widget 指定一个 key
                )
            with col2:
                # 发送按钮，绑定 on_click 回调
                st.button(
                    "发送",
                    use_container_width=True,
                    key="send_button",
                    on_click=on_send_click  # 绑定回调函数
                )

            st.markdown('</div>', unsafe_allow_html=True)

        # 注意：不再返回用户输入，也不直接在这里修改 st.session_state.user_input
        # 输入的获取和清空由 app.py 的 run 方法处理

# 确保 __name__ == "__main__" 不会执行任何代码，因为这是被导入的模块
if __name__ == "__main__":
    pass  # UIComponents 通常不直接运行
