import asyncio
import os
import base64
from typing import List

# 假设你的类和函数都在一个模块中，比如 rag_manager.py
from build_database import RAGDatabaseManager  # 替换为你的实际文件名

async def test_llm_model():
    db_manager = RAGDatabaseManager(llm_model="qwen3:8b")
    await db_manager._create_rag_instance()
    llm_func = db_manager.llm_model_func  # 获取函数引用

    response = await llm_func(
        prompt="请用中文介绍一下人工智能。",
        system_prompt="你是一个 helpful assistant."
    )
    print("LLM Model Response:")
    print(response)

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def test_vision_model():
    db_manager = RAGDatabaseManager(vision_model="llava:latest")
    await db_manager._create_rag_instance()
    vision_func = db_manager.vision_model_func

    # 准备图像（请替换为你的测试图片路径）
    image_path = r"D:\adavance\tsy\rag4chat\output\test\auto\images\1b6f94a003aca5e16796a25d2e6c97b3d90f875c433d22aeb65552e3b8420e7e.jpg"  # 比如一只猫、狗或文档截图
    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found!")
        return

    image_data = image_to_base64(image_path)

    response = await vision_func(
        prompt="请描述这张图片的内容。",
        image_data=image_data
    )
    print("Vision Model Response:")
    print(response)

async def test_embedding_func():
    db_manager = RAGDatabaseManager(embed_model="text-embedding-v4")
    await db_manager._create_rag_instance()
    embed_func = db_manager.embedding_func.func  # 提取真正的异步函数

    texts = [
        "人工智能是计算机科学的一个分支。",
        "LLaVA 是一个多模态大模型。",
        ""
    ]

    embeddings = await embed_func(texts)
    print("Embedding Test Results:")
    for i, emb in enumerate(embeddings):
        print(f"Text {i}: {texts[i][:30]}... -> Embedding Dim: {emb}")

async def main():
    print("\n🧪 Testing LLM Model...")
    await test_llm_model()

    print("\n🖼️ Testing Vision Model...")
    await test_vision_model()

    print("\n📊 Testing Embedding Function...")
    await test_embedding_func()

if __name__ == "__main__":
    asyncio.run(main())

