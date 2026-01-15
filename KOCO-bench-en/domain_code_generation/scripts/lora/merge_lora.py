#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_lora.py — 将 LoRA adapter 合并到基础模型

用途：
1. 将 LoRA adapter 权重合并到基础模型，生成完整的模型
2. 合并后的模型可以直接使用 transformers 加载，无需 peft 库
3. 便于部署和分发

使用示例：
python merge_lora.py \
  --base_model /path/to/Qwen2.5-Coder-7B-Instruct \
  --lora_adapter ./models/qwen2.5-coder-7b-verl-lora \
  --output_dir ./models/qwen2.5-coder-7b-verl-merged
"""

import os
import sys
import torch
import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(
    base_model_path: str,
    lora_adapter_path: str,
    output_dir: str,
    device: str = "auto",
    torch_dtype: str = "auto",
):
    """
    合并 LoRA 权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA adapter 路径
        output_dir: 输出目录
        device: 设备
        torch_dtype: 数据类型
    """
    print("="*60)
    print("🔧 开始合并 LoRA 权重")
    print("="*60)
    
    # 处理 torch_dtype
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"\n📦 加载基础模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # 避免 flash-attention GLIBC 问题
    )
    
    print(f"\n🔧 加载 LoRA adapter: {lora_adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=dtype,
    )
    
    print("\n🔄 合并权重中...")
    merged_model = model.merge_and_unload()
    
    print(f"\n💾 保存合并后的模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,  # 使用 safetensors 格式
    )
    
    # 保存 tokenizer
    print("💾 保存 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_adapter_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(output_dir)
    
    # 保存配置信息
    print("💾 保存合并配置...")
    merge_info = {
        "base_model": base_model_path,
        "lora_adapter": lora_adapter_path,
        "merged_at": __import__('datetime').datetime.now().isoformat(),
        "device": str(device),
        "dtype": str(dtype),
    }
    
    import json
    with open(os.path.join(output_dir, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(merge_info, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("✅ 合并完成！")
    print("="*60)
    print(f"\n📁 模型保存位置: {output_dir}")
    print("\n💡 使用方法:")
    print("```python")
    print("from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print("```")
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基础模型")
    parser.add_argument(
        "--base_model",
        "-b",
        type=str,
        required=True,
        help="基础模型路径",
    )
    parser.add_argument(
        "--lora_adapter",
        "-l",
        type=str,
        required=True,
        help="LoRA adapter 路径",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备: auto, cuda:0, cpu 等（默认: auto）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="数据类型（默认: auto）",
    )
    
    args = parser.parse_args()
    
    # 验证路径
    if not Path(args.base_model).exists():
        print(f"❌ 错误: 基础模型路径不存在: {args.base_model}")
        return 1
    
    if not Path(args.lora_adapter).exists():
        print(f"❌ 错误: LoRA adapter 路径不存在: {args.lora_adapter}")
        return 1
    
    # 执行合并
    try:
        merge_lora_weights(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_adapter,
            output_dir=args.output_dir,
            device=args.device,
            torch_dtype=args.dtype,
        )
        return 0
    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

