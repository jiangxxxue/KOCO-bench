import asyncio
import json
import os
import numpy as np
import base64
import io
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI

# =======================
# 工具函数：调用阿里云百炼 Embedding API（异步）
# =======================
async def bailian_embed_async(text: str, model: str = "text-embedding-v4") -> List[float]:
    """
    异步调用阿里云百炼 Embedding API (text-embedding-v4)
    使用 OpenAI 兼容接口
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置")

    # 创建 AsyncOpenAI 客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" # 注意：移除了末尾空格
    )

    target_dim = 1024
    try:
        # 调用 Embeddings API
        response = await client.embeddings.create(
            model=model,
            input=text,
            dimensions=target_dim,
            encoding_format="float"
        )
        # 提取嵌入向量
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"❌ 调用 Embedding API 时出错: {e}")
        # 根据需求返回零向量或重新抛出异常
        # 这里选择返回零向量以避免中断
        return [0.0] * target_dim

# =======================
# 工具函数：解码 Base64 字符串为 NumPy 数组
# =======================
# =======================
# 工具函数：解码 Base64 字符串为 NumPy 数组 (修正版 - 支持 pickle)
# =======================
# =======================
# 工具函数：解码 Base64 字符串为 NumPy 数组 (修正版 - 直接解释为 float32)
# =======================
def decode_base64_vector_matrix(base64_str: str, num_vectors: int, vector_dim: int = 1024, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    将 Base64 编码的原始 float32 向量数据解码为 NumPy 数组。
    
    假设原始数据是 num_vectors 个 vector_dim 维的 dtype 类型向量的连续二进制数据。
    """
    try:
        # 1. Base64 解码
        decoded_bytes = base64.b64decode(base64_str)
        print(f"✅ 解码 matrix 字符串为 {len(decoded_bytes)} 字节。")

        # 2. 验证字节长度是否匹配预期
        expected_bytes = num_vectors * vector_dim * dtype().itemsize # itemsize for float32 is 4
        if len(decoded_bytes) != expected_bytes:
            print(f"⚠️ 警告: 解码后的字节数 ({len(decoded_bytes)}) 与预期 ({expected_bytes}) 不符。")
            # 可以选择返回空数组或尝试处理
            # return np.array([]) # 或者抛出异常

        # 3. 将字节重新解释为 NumPy 数组
        # 注意：这里假设数据是以小端序 (little-endian) 存储的
        array_flat = np.frombuffer(decoded_bytes, dtype=dtype)
        
        # 4. 重塑为 (num_vectors, vector_dim) 的矩阵
        array_matrix = array_flat.reshape((num_vectors, vector_dim))
        
        print(f"✅ 将字节重塑为 NumPy 数组，形状为 {array_matrix.shape}，数据类型为 {array_matrix.dtype}。")
        return array_matrix

    except Exception as e:
        print(f"❌ 解码/重塑 matrix 时出错: {e}")
        import traceback
        traceback.print_exc()
        # 根据需要处理异常，例如返回 None 或抛出特定异常
        raise # 重新抛出异常以便调用者处理

# =======================
# 余弦相似度计算
# =======================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个 NumPy 向量之间的余弦相似度"""
    # 确保是 1D 向量
    a = a.ravel()
    b = b.ravel()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0 # 避免除以零
    return np.dot(a, b) / (norm_a * norm_b)

# =======================
# 加载 vdb_chunks.json 数据
# =======================
def load_vdb_chunks(file_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    从指定路径加载 vdb_chunks.json 文件，
    并解码 'matrix' 字段中的 Base64 编码的 NumPy 数组。
    返回 (data 列表, 解码后的矩阵 NumPy 数组)。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks_data = data.get("data", [])
        raw_matrix_string = data.get("matrix", "") # 注意：这里是单个字符串
        
        print(f"✅ 从 data 列表加载了 {len(chunks_data)} 个 chunks。")

        if not raw_matrix_string:
             print("⚠️ 警告: JSON 文件中 'matrix' 字段为空或缺失。")
             return chunks_data, np.array([]) # 返回空数组

        # 解码 Base64 编码的 NumPy 矩阵
        try:
            # 使用函数解码 matrix 字符串 (仅 Base64 解码 + NumPy 加载)
            num_chunks = len(chunks_data)
            # 使用修正后的函数解码 matrix 字符串 (Base64 解码 -> 字节 -> float32 数组 -> reshape)
            matrix_data_np: np.ndarray = decode_base64_vector_matrix(raw_matrix_string, num_vectors=num_chunks, vector_dim=1024, dtype=np.float32)
        except Exception as e:
             print(f"❌ 解码 'matrix' 字段失败: {e}")
             return chunks_data, np.array([]) # 返回空数组

        # 验证矩阵形状
        if matrix_data_np.size > 0:
            expected_rows = len(chunks_data)
            actual_rows, actual_cols = matrix_data_np.shape
            if actual_rows != expected_rows:
                print(f"⚠️ 警告: 矩阵行数 ({actual_rows}) 与 data 列表长度 ({expected_rows}) 不匹配。")
                # 可以选择截断或填充，这里简单警告
            if actual_cols != 1024: # 假设维度是 1024，根据你的 embedding_dim 调整
                 print(f"⚠️ 警告: 矩阵列数 ({actual_cols}) 不是 1024。")
            else:
                 print(f"✅ 矩阵形状一致: {matrix_data_np.shape} 对应 {expected_rows} 个 chunks。")
        else:
             print("⚠️ 警告: 解码后的矩阵为空。")

        return chunks_data, matrix_data_np

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        return [], np.array([])
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 无法解析 JSON 文件 {file_path}: {e}")
        return [], np.array([])
    except Exception as e:
        print(f"❌ 加载数据时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return [], np.array([])

# =======================
# 主查询和相似度计算函数
# =======================
async def query_and_find_topk(query_text: str, vdb_file_path: str = "./rag_storage/vdb_chunks.json", topk: int = 5):
    """
    查询文本，并从 vdb_chunks.json 中找到 topk 个最相似的条目
    使用预先计算并解码好的 matrix 向量
    """
    print(f"🔍 正在从: {vdb_file_path} 加载数据...")
    
    # 1. 加载数据和预计算的向量 (已解码)
    chunks_data, matrix_data_np = load_vdb_chunks(vdb_file_path)
    
    if not chunks_data or matrix_data_np.size == 0:
        print("⚠️ 警告: 未加载到任何数据或向量。")
        return

    num_chunks = len(chunks_data)
    num_vectors = matrix_data_np.shape[0] if matrix_data_np.size > 0 else 0
    print(f"✅ 成功加载 {num_chunks} 个条目和对应的向量矩阵 ({num_vectors} 行)。")

    # 2. 获取查询文本的 embedding (使用 AsyncOpenAI API)
    print("🔍 正在获取查询文本的 Embedding (通过阿里云百炼 API)...")
    try:
        query_embedding_list = await bailian_embed_async(query_text)
        # 转换为 NumPy 数组以便计算，确保形状为 (1024,)
        query_embedding_np = np.array(query_embedding_list, dtype=np.float32).ravel() # .ravel() 确保是 1D
        print(f"✅ 查询 Embedding 形状: {query_embedding_np.shape}")
    except ValueError as e:
        print(f"❌ Embedding 错误: {e}")
        return
    except Exception as e:
        print(f"❌ 获取查询 Embedding 时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 计算所有条目的相似度 (使用预计算并解码的向量矩阵)
    print("🔍 正在计算相似度...")
    similarities_and_chunks: List[Tuple[float, Dict[str, Any]]] = []

    # 使用矩阵运算计算所有相似度会更高效，但循环也适用且更直观
    # 假设 matrix_data_np 形状是 (N, 1024)
    for i in range(matrix_data_np.shape[0]):
        try:
            # 取出第 i 个 chunk 的 embedding 向量
            content_embedding_np = matrix_data_np[i] # 形状 (1024,)
            # 计算与查询向量的余弦相似度
            sim = cosine_similarity(query_embedding_np, content_embedding_np)
            # 将相似度和对应的 chunk 信息存入列表
            # 注意：确保索引 i 在 chunks_data 范围内（虽然加载时已对齐）
            if i < len(chunks_data):
                 similarities_and_chunks.append((sim, chunks_data[i]))
            else:
                 print(f"❌ 索引不匹配: 矩阵行 {i} 没有对应的数据块。")
        except Exception as e:
            print(f"❌ 处理条目索引 {i} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not similarities_and_chunks:
        print("⚠️ 警告: 未能计算任何相似度。")
        return

    # 4. 排序并获取 Top-K
    print(f"🔍 正在排序并获取 Top-{topk}...")
    # 按相似度降序排列
    similarities_and_chunks.sort(key=lambda x: x[0], reverse=True)
    topk_results = similarities_and_chunks[:topk]

    # 5. 输出结果
    print("\n--- 📋 查询结果 (Top-K) ---")
    print(f"🔍 查询文本: {query_text}\n")
    # print(topk_results)
    for i, (similarity, chunk) in enumerate(topk_results):
        content = chunk.get("content", "N/A")
        file_path = chunk.get("file_path", "N/A")
        chunk_id = chunk.get("__id__", "N/A")
        created_at = chunk.get("__created_at__", "N/A")
        print(f"--- 🏆 Top {i+1} (相似度: {similarity:.4f}) ---")
        print(f"🆔 ID: {chunk_id}")
        print(f"🕒 创建时间: {created_at}")
        print(f"📁 文件路径: {file_path}")
        # 打印内容的前 N 个字符
        print(f"📄 内容预览: {content[:500]}...\n")

    formatted_results = []
    for similarity, chunk in topk_results:
            # 构造一个新的字典，包含原始 chunk 信息和相似度
            result_item = {
                "similarity": similarity,
                "content": chunk.get("content", ""),
                "file_path": chunk.get("file_path", ""),
                "chunk_id": chunk.get("__id__", ""),
                # 可以添加其他需要的字段
            }
            formatted_results.append(result_item)
    
    print("\n--- 查询结果 (Top-K) ---")
    print(f"查询文本: {query_text}\n")
    for i, item in enumerate(formatted_results): # 使用 formatted_results 打印
        # 从 item 字典中获取信息
        content = item.get("content", "N/A")
        file_path = item.get("file_path", "N/A")
        chunk_id = item.get("chunk_id", "N/A")
        similarity_score = item.get("similarity", 0.0) # 获取相似度
        print(f"--- Top {i+1} (相似度: {similarity_score:.4f}) ---")
        print(f"ID: {chunk_id}")
        print(f"文件路径: {file_path}")
        print(f"内容: {content[:500]}...\n")
        
    # 返回格式化后的字典列表
    return formatted_results
    

# =======================
# 入口函数
# =======================
async def main():
    # --- 配置 ---
    query_text = "语义图像？"
    vdb_file_path = "./rag_storage/vdb_chunks.json" # 请确保路径正确
    topk = 5
    # --- 配置结束 ---

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 环境变量 DASHSCOPE_API_KEY 未设置。请设置后重试。")
        return # 优雅退出

    result = await query_and_find_topk(query_text, vdb_file_path, topk)
    return result

# =======================
# 运行
# =======================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断。")
    except Exception as e:
        print(f"❌ 程序执行过程中发生未处理的错误: {e}")
        import traceback
        traceback.print_exc()
