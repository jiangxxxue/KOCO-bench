"""
UI组件模块，包含所有Streamlit UI渲染逻辑
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Any
import logging
from utils.document_processor import DocumentProcessor
from services.vector_store import VectorStoreService
from langchain.schema import Document
from config.settings import AVAILABLE_EMBEDDING_MODELS
import concurrent.futures
import functools
import asyncio

logger = logging.getLogger(__name__)

import streamlit as st
import base64
import re # 用于正则表达式匹配图片路径

def convert_local_images_to_base64(markdown_text: str) -> str:
    """
    遍历 Markdown 文本，查找本地图片路径，并将其替换为 Base64 嵌码。
    注意：此函数假设图片路径是 Windows 风格的绝对路径。
    """
    # 定义一个内部函数来处理图片转换
    def _replace_image_path(match):
        alt_text = match.group(1) # 获取 alt text
        image_path = match.group(2) # 获取图片路径
        
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
                return match.group(0) # 返回原始匹配内容
            except Exception as e:
                # 处理其他可能的错误（如读取权限问题）
                st.error(f"处理图片 '{image_path}' 时出错: {e}")
                return match.group(0) # 返回原始匹配内容
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
            asyncio.set_event_loop(None) # 重置线程的事件循环

    # 使用 ThreadPoolExecutor 在新线程中执行 _run_in_thread 函数
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务并等待结果
        future = executor.submit(_run_in_thread)
        # 阻塞等待结果（这发生在主线程，但不会阻塞 Streamlit 的事件循环太久，因为工作在后台线程）
        # 如果协程内部有异常，future.result() 会重新抛出它
        return future.result()
    
class UIComponents:
    """UI组件类，封装了所有Streamlit UI渲染逻辑"""
    
    # 1.渲染模型选择组件
    @staticmethod
    def render_model_selection(available_models: List[str], current_model: str, embedding_models: List[str], current_embedding_model: str) -> Tuple[str, str]:
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
        from services.export_file import mdcontent2docx, mdcontent2pdf
        if st.sidebar.button("✅ 导出当前内容为PDF", use_container_width=True):
            # print(chat_history)
            last_assistant_message = None

            for message in chat_history.history:
                role = message.get('role', '') # 安全地获取 'role' 键的值
                content = message.get('content', '') # 安全地获取 'content' 键的值

                if role == 'assistant':
                    last_assistant_message = content # 更新为最新的 assistant 消息
            
            mdcontent2pdf(last_assistant_message, './now_content.pdf')
            st.rerun()

        if st.sidebar.button("🚀 导出当前内容为DOCX", use_container_width=True):
            last_assistant_message = None

            for message in chat_history.history:
                role = message.get('role', '') # 安全地获取 'role' 键的值
                content = message.get('content', '') # 安全地获取 'content' 键的值

                if role == 'assistant':
                    last_assistant_message = content # 更新为最新的 assistant 消息
            mdcontent2docx(last_assistant_message, './now_content.docx')
            st.rerun()

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
        processed_documents - 已处理的文档列表

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
                        # 你需要根据 Document 对象的实际结构来获取路径
                        # 常见的是在 metadata['source'] 中
                        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                            source_path = doc.metadata.get('source')
                            if source_path and isinstance(source_path, str):
                                document_paths.append(source_path)
                            else:
                                # 处理无法获取路径的情况
                                st.warning(f"⚠️ 无法从文档对象获取文件路径: {doc}")
                                logger.warning(f"无法从文档对象获取文件路径: {doc}")
                        else:
                            # 处理没有 metadata 或 metadata 不是字典的情况
                            st.warning(f"⚠️ 文档对象缺少有效的 metadata: {doc}")
                            logger.warning(f"文档对象缺少有效的 metadata: {doc}")
                    # --- 新增结束 ---
                    
                    if not document_paths:
                        st.error("❌ 没有有效的文档路径可供处理。")
                        logger.error("没有有效的文档路径可供处理。")
                        return # 或其他错误处理
                    with st.spinner("正在构建向量索引..."):
                        import asyncio 
                        # vector_store.vector_store = vector_store.create_vector_store(all_docs)
                        # success = asyncio.run(vector_store.create_vector_store(document_paths))
                        # 获取当前运行的事件循环
                        # loop = asyncio.get_event_loop()
                        # 使用 run_until_complete 在当前循环中等待协程完成
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
        for message in chat_history.history:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == "assistant_think":
                with st.expander("💡 查看推理过程 <think> ... </think>"):
                    st.markdown(content)
            elif role == "retrieved_doc":
                with st.expander(f"🔎 查看本次召回的文档块", expanded=False):
                    if isinstance(content, list):
                        for idx, doc in enumerate(content, 1):
                            st.markdown(f"**文档块{idx}:**\n{doc}")
                    else:
                        st.markdown(content)
            else:
                # with st.chat_message(role):
                #     st.write(content)
                print(content)
                corrected_markdown_content = convert_local_images_to_base64(content)

                # 2. 推送到前端 (Streamlit 聊天框)
                with st.chat_message("assistant"): 
                    st.markdown(corrected_markdown_content)

                original_markdown_content = """
                ### 3.2.2 多通道语义图像

                在目标检测算法FFTNet中，便通过灰度语义图像构建了高斯椭圆组成的Heatmap来指示像素属于目标中心点的概率，以及该目标的回归尺度。

                #### 1. 固定部件

                1.  初始化一个长宽为 `width/8` 和 `height/8` 的灰度图像张量。
                2.  遍历标准图标注中的每一个固定部件，根据其边界框生成目标的高斯椭圆。
                3.  将每个固定部件的高斯椭圆添加在对应类别通道的灰度语义图像的对应位置上。

                ![图1：高斯椭圆构造效果](C:/Users/wcz13/Pictures/881.png)
                """

                # --- 修正并推送 ---
                # 1. 修正 Markdown 内容
                corrected_markdown_content = convert_local_images_to_base64(original_markdown_content)

                # 2. 推送到前端 (Streamlit 聊天框)
                with st.chat_message("assistant"): 
                    st.markdown(corrected_markdown_content)