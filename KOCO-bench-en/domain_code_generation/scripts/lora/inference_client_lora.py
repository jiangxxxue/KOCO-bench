#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_client_lora.py — LoRA 推理客户端

通过 HTTP 请求调用 LoRA 推理服务器进行代码生成
与 inference_client.py 类似，但专门用于 LoRA 服务器
"""

import json
import argparse
import requests
import time
from typing import List, Dict, Any
from pathlib import Path


class LoRAInferenceClient:
    """LoRA 推理客户端"""
    
    def __init__(self, server_url: str):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，例如 http://localhost:8000
        """
        self.server_url = server_url.rstrip('/')
        self.health_url = f"{self.server_url}/health"
        self.generate_url = f"{self.server_url}/generate"
    
    def check_health(self) -> Dict[str, Any]:
        """检查服务器健康状态"""
        try:
            response = requests.get(self.health_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"服务器健康检查失败: {e}")
    
    def generate(
        self,
        prompts: List[str],
        num_completions: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[List[str]]:
        """
        生成代码补全
        
        Args:
            prompts: 提示列表
            num_completions: 每个提示生成的补全数量
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: Top-p采样
            do_sample: 是否采样
        
        Returns:
            补全列表，外层对应每个提示，内层对应每个提示的多个补全
        """
        payload = {
            "prompts": prompts,
            "num_completions": num_completions,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=300  # 5分钟超时
            )
            response.raise_for_status()
            result = response.json()
            return result["completions"]
        
        except requests.exceptions.Timeout:
            raise RuntimeError("请求超时")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"请求失败: {e}")
        except Exception as e:
            raise RuntimeError(f"生成失败: {e}")


def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl_data(data: List[Dict[str, Any]], file_path: str):
    """保存 JSONL 数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def format_prompt(prompt_data):
    """
    将 prompt 数据格式化为字符串
    
    Args:
        prompt_data: 可以是字符串或对话列表
    
    Returns:
        格式化后的字符串
    """
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        # 对话列表格式：[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        formatted_parts = []
        for message in prompt_data:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")
        return "\n\n".join(formatted_parts)
    else:
        return str(prompt_data)


def process_dataset(
    client: LoRAInferenceClient,
    input_file: str,
    output_file: str,
    model_name: str,
    num_completions: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 1,
):
    """
    处理数据集
    
    Args:
        client: 推理客户端
        input_file: 输入文件路径
        output_file: 输出文件路径
        model_name: 模型名称（用于输出目录）
        num_completions: 每个任务生成的补全数量
        max_tokens: 最大生成token数
        temperature: 温度参数
        top_p: Top-p采样
        batch_size: 批处理大小
    """
    # 加载数据
    print(f"📂 加载数据: {input_file}")
    data = load_jsonl_data(input_file)
    print(f"  找到 {len(data)} 个任务")
    
    # 构建输出路径
    output_path = Path(output_file)
    if not output_path.is_absolute():
        # 如果是相对路径，基于输入文件的目录
        input_path = Path(input_file)
        base_dir = input_path.parent
        
        # 创建模型输出目录
        model_dir = base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        output_filename = input_path.stem + "_output" + input_path.suffix
        output_path = model_dir / output_filename
    
    print(f"📂 输出文件: {output_path}")
    print()
    
    # 批量处理
    total = len(data)
    processed = 0
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        # 直接使用 prompt（可以是字符串或对话列表，服务器端会处理格式化）
        batch_prompts = [item["prompt"] for item in batch]
        
        print(f"🔄 处理批次 {i // batch_size + 1} / {(total + batch_size - 1) // batch_size}")
        print(f"   任务 {i + 1}-{min(i + len(batch), total)} / {total}")
        
        try:
            # 生成
            start_time = time.time()
            completions = client.generate(
                prompts=batch_prompts,
                num_completions=num_completions,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed = time.time() - start_time
            
            # 保存结果
            for j, item in enumerate(batch):
                item["completions"] = completions[j]
            
            processed += len(batch)
            print(f"   ✅ 完成 ({elapsed:.2f}s) - 进度: {processed}/{total}")
            
        except Exception as e:
            print(f"   ❌ 失败: {e}")
            # 保存空结果
            for item in batch:
                item["completions"] = [""] * num_completions
        
        print()
    
    # 保存输出
    print(f"💾 保存结果: {output_path}")
    save_jsonl_data(data, str(output_path))
    print()
    print("✅ 处理完成！")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 代码补全推理客户端 - 通过 HTTP 请求调用推理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python inference_client_lora.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl

  # 生成多个补全
  python inference_client_lora.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl \\
    --num_completions 4 \\
    --temperature 0.2

  # 使用批处理
  python inference_client_lora.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl \\
    --batch_size 4
        """
    )
    
    # 服务器配置
    parser.add_argument(
        "--server_url",
        type=str,
        required=True,
        help="推理服务器地址，例如 http://localhost:8000"
    )
    
    # 文件路径
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入 JSONL 文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出 JSONL 文件路径（默认: 自动生成）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lora-model",
        help="模型名称（用于输出目录）"
    )
    
    # 生成参数
    parser.add_argument(
        "--num_completions",
        type=int,
        default=1,
        help="每个任务生成的补全数量 (默认: 1)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="最大生成token数 (默认: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数 (默认: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p采样 (默认: 0.95)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批处理大小 (默认: 1)"
    )
    
    args = parser.parse_args()
    
    # 设置输出文件
    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_output{input_path.suffix}")
    
    print("=" * 60)
    print("🚀 LoRA 推理客户端")
    print("=" * 60)
    print(f"服务器: {args.server_url}")
    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"模型名称: {args.model_name}")
    print(f"补全数量: {args.num_completions}")
    print(f"批处理大小: {args.batch_size}")
    print("=" * 60)
    print()
    
    # 创建客户端
    client = LoRAInferenceClient(args.server_url)
    
    # 健康检查
    print("🔍 检查服务器健康状态...")
    try:
        health = client.check_health()
        print(f"✅ 服务器正常")
        print(f"  - 状态: {health['status']}")
        print(f"  - 基础模型: {health['base_model']}")
        print(f"  - LoRA adapter: {health['lora_adapter']}")
        print(f"  - 设备: {health['device']}")
        print()
    except Exception as e:
        print(f"❌ 服务器不可用: {e}")
        return 1
    
    # 处理数据集
    try:
        process_dataset(
            client=client,
            input_file=args.input_file,
            output_file=args.output_file,
            model_name=args.model_name,
            num_completions=args.num_completions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )
        return 0
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

