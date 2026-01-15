"""RAG 核心类 - 使用已存在的 LightRAG 实例"""
import asyncio
import aiohttp
import os
import json
from typing import List, Dict, Any, Optional
from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from config import LLM_MODEL, EMBED_MODEL, VLM_MODEL, WORKING_DIR, OUTPUT_DIR, DB_INFO_FILE

async def ollama_complete_async(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    image_data: Optional[str] = None,
    **kwargs
) -> str:
    """异步调用 Ollama"""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for msg in history_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

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
    """异步获取嵌入向量"""
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
                        print(f"⚠️ Embedding warning for text '{text[:50]}...': {text_resp}")
                        embeddings.append([0.0] * 1024)
                    else:
                        result = await resp.json()
                        embedding = result["embedding"]
                        if len(embedding) != 1024:
                            print(f"⚠️ Embedding dimension mismatch: expected 1024, got {len(embedding)}")
                            if len(embedding) > 1024:
                                embedding = embedding[:1024]
                            else:
                                embedding.extend([0.0] * (1024 - len(embedding)))
                        embeddings.append(embedding)
            except Exception as e:
                print(f"⚠️ Embedding exception for text '{text[:50]}...': {e}")
                embeddings.append([0.0] * 1024)
    return embeddings

class RAGCore:
    def __init__(self):
        self.rag = None
        self.lightrag_instance = None
        
    async def initialize_with_existing_lightrag(self):
        """使用已存在的 LightRAG 实例初始化"""
        print("🔄 初始化 RAG 系统（使用已存在的 LightRAG 实例）...")
        
        # 检查是否存在已构建的数据库
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR, exist_ok=True)
            
        # 检查是否已有 LightRAG 数据
        has_existing_data = (
            os.path.exists(WORKING_DIR) and 
            os.listdir(WORKING_DIR) and
            any(f for f in os.listdir(WORKING_DIR) if f.endswith('.json') or f.endswith('.graphml'))
        )
        
        if has_existing_data:
            print("✅ 发现已存在的 LightRAG 实例，正在加载...")
        else:
            print("⚠️ 未找到已存在的 LightRAG 实例，将创建新实例")
        
        # 创建 LightRAG 实例
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # 注意：这里使用同步版本，因为 LightRAG 可能期望同步函数
            return asyncio.run(ollama_complete_async(
                model=LLM_MODEL,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **kwargs
            ))

        def embedding_func(texts: List[str]) -> List[List[float]]:
            # 注意：这里使用同步版本
            return asyncio.run(ollama_embed_async(texts, EMBED_MODEL))

        # 创建 LightRAG 实例
        self.lightrag_instance = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func,
            )
        )
        
        # 定义视觉模型函数
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
        
        # 使用已存在的 LightRAG 实例初始化 RAGAnything
        self.rag = RAGAnything(
            lightrag=self.lightrag_instance,
            vision_model_func=vision_model_func,
        )
        
        print("✅ RAG 系统初始化完成")
        
    async def build_database(self, file_path: str):
        """构建数据库"""
        if self.rag is None:
            await self.initialize_with_existing_lightrag()
            
        # 支持的文件格式
        supported_formats = ['.pdf', '.docx', '.txt', '.md', '.html', '.epub']
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in supported_formats:
            supported_list = ', '.join(supported_formats)
            raise ValueError(f"不支持的文件格式: {file_extension}。支持的格式: {supported_list}")
            
        # 确保目录存在
        os.makedirs(WORKING_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 处理文档
        print(f"📄 处理文档: {file_path}")
        print(f"📝 文件类型: {file_extension}")
        
        await self.rag.process_document_complete(
            file_path=file_path,
            output_dir=OUTPUT_DIR,
            parse_method="auto"
        )
        print("✅ 数据库构建完成")
        
        # 保存数据库状态
        db_info = {
            "status": "built",
            "file_path": file_path,
            "file_type": file_extension,
            "build_time": asyncio.get_event_loop().time()
        }
        with open(DB_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(db_info, f, ensure_ascii=False, indent=2)
        
    async def load_database(self):
        """加载已存在的数据库"""
        if not os.path.exists(DB_INFO_FILE):
            raise Exception("数据库不存在，请先运行 build_database.py 构建数据库")
            
        if self.rag is None:
            await self.initialize_with_existing_lightrag()
            
        print("✅ 数据库加载完成")
        
    async def query(self, question: str, mode: str = "hybrid"):
        """文本查询"""
        if self.rag is None:
            await self.load_database()
            
        try:
            return await self.rag.aquery(question, mode=mode)
        except Exception as e:
            print(f"⚠️ 查询出现异常，尝试备用方法: {e}")
            # 备用查询方法
            if self.lightrag_instance:
                return await self.lightrag_instance.aquery(question, mode=mode)
            raise e
        
    async def multimodal_query(self, question: str, multimodal_content: List[Dict], mode: str = "hybrid"):
        """多模态查询"""
        if self.rag is None:
            await self.load_database()
            
        return await self.rag.aquery_with_multimodal(
            question,
            multimodal_content=multimodal_content,
            mode=mode
        )