import asyncio
import os
import base64
# --- 导入你自己的函数 ---
# 注意：不要从当前文件导入自己！确保 build_database.py 是独立的文件
from build_database import (
    ollama_complete_async,
    ollama_vision_complete_async,
    bailian_embed_async,
)
from raganything import RAGAnything
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
import logging

# --- 设置日志 ---
# logging.basicConfig(level=logging.DEBUG) # 调试时可以启用
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 定义你自己的 LLM Model Func (适配 LightRAG/RAGAnything) ---
async def my_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:
        response = await ollama_complete_async(
            model="qwen3:8b",
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
        logger.debug(f"LLM Response: {response}")
        return response
    except Exception as e:
        logger.error(f"LLM Model Func Error: {e}", exc_info=True) # exc_info=True 打印堆栈
        raise

# --- 定义你自己的 Embedding Func (适配 LightRAG) ---
async def my_embedding_func(texts):
    try:
        # 确保传入正确的模型名
        embeddings = await bailian_embed_async(texts, model="text-embedding-v4")
        logger.debug(f"Generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding Func Error: {e}", exc_info=True)
        target_dim = 1024
        return [[0.0] * target_dim for _ in texts]

async def load_existing_rag():
    lightrag_working_dir = "./rag_storage"

    if os.path.exists(lightrag_working_dir) and os.listdir(lightrag_working_dir):
        print("✅ 发现已存在的 LightRAG 存储目录，正在加载...")
    else:
        print("⚠️ 未找到已存在的 LightRAG 存储目录或目录为空。")

    try:
        lightrag_instance = LightRAG(
            working_dir=lightrag_working_dir,
            llm_model_func=my_llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=512,
                func=my_embedding_func,
            )
        )
        print("✅ LightRAG 实例创建成功。")
    except Exception as e:
        print(f"❌ 创建或加载 LightRAG 实例时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return

    # --- 定义视觉模型函数用于 RAGAnything ---
    async def my_vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=[],
        image_data=None,
        **kwargs
    ):
        if image_data:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            try:
                response = await ollama_vision_complete_async(
                    model="llava:latest",
                    prompt=full_prompt,
                    image_data=image_data,
                    **kwargs
                )
                logger.debug(f"Vision Model Response: {response}")
                return response
            except Exception as e:
                logger.error(f"Vision Model Func Error: {e}", exc_info=True)
                raise
        else:
            logger.info("No image data provided, falling back to LLM.")
            return await my_llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    try:
        # --- 传递实例属性 ---
        rag = RAGAnything(
            lightrag=lightrag_instance,
            vision_model_func=my_vision_model_func,
            # llm_model_func=my_llm_model_func, # 通常由 lightrag_instance 提供
            # embedding_func=lightrag_instance.embedding_func, # 通常由 lightrag_instance 提供
        )
        print("✅ RAGAnything 实例创建成功。")
    except Exception as e:
        print(f"❌ 创建 RAGAnything 实例时出错: {e}")
        import traceback
        print(traceback.format_exc())
        return

    # --- 示例查询 ---
    print("\n--- 执行文本查询 ---")
    try:
        text_query = "rag4chat项目的主要功能是什么？"
        # --- 使用 aquery 并传递 mode ---
        text_result = await rag.aquery(
            text_query,
            mode="hybrid"
        )
        print(f"📝 文本查询 '{text_query}' 的结果:")
        # 根据 RAGAnything 返回结构打印结果
        # 常见的可能是直接返回字符串或包含 'response' 键的字典
        if isinstance(text_result, dict):
             print(text_result.get('response', str(text_result)))
        else:
             print(str(text_result))

    except Exception as e:
        import traceback
        print(f"❌ 文本查询执行失败: {e}")
        print(traceback.format_exc()) # 关键：打印完整堆栈

    # print("\n--- 执行图文查询 (请确保有图片) ---")
    # image_path = r"D:\adavance\tsy\rag4chat\output\test\auto\images\1b6f94a003aca5e16796a25d2e6c97b3d90f875c433d22aeb65552e3b8420e7e.jpg"
    # if os.path.exists(image_path):
    #     try:
    #         with open(image_path, "rb") as image_file:
    #             image_data = base64.b64encode(image_file.read()).decode('utf-8')

    #         vision_query = "请描述这张图片的内容。"

    #         # --- 尝试使用 aquery 并传递图片 ---
    #         # 这是关键部分，需要查阅 RAGAnything 文档确认如何传入图片
    #         # 常见的方式可能是通过 kwarg 传递 images=[base64_str]
    #         vision_result = await rag.aquery(
    #             vision_query,
    #             image_data=[image_data],
    #             mode="hybrid" # 如果支持，可以指定模式
    #         )
    #         print(f"🖼️ 图文查询 '{vision_query}' 的结果:")
    #         if isinstance(vision_result, dict):
    #              print(vision_result.get('response', str(vision_result)))
    #         else:
    #              print(str(vision_result))

    #     except Exception as e:
    #         import traceback
    #         print(f"❌ 图文查询执行失败: {e}")
    #         print(traceback.format_exc())
    # else:
    #     print(f"⚠️ 图片文件 {image_path} 不存在，跳过图文查询。")


if __name__ == "__main__":
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 环境变量 DASHSCOPE_API_KEY 未设置。")
    else:
        asyncio.run(load_existing_rag())