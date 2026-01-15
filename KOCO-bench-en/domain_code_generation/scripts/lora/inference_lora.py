#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_lora.py — LoRA 模型推理示例

演示如何加载并使用训练好的 LoRA adapter 进行代码生成
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_lora_model(
    base_model_path: str,
    lora_adapter_path: str,
    device: str = "auto",
    torch_dtype: str = "auto",
):
    """
    加载基础模型 + LoRA adapter
    
    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA adapter 路径
        device: 设备，"auto" 或 "cuda:0" 等
        torch_dtype: 数据类型，"auto", "bfloat16", "float16" 等
    
    Returns:
        model, tokenizer
    """
    print(f"📦 加载基础模型: {base_model_path}")
    
    # 处理 torch_dtype
    if torch_dtype == "auto":
        dtype = "auto"
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # 避免 flash-attention GLIBC 问题
    )
    
    print(f"🔧 加载 LoRA adapter: {lora_adapter_path}")
    
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=dtype,
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        lora_adapter_path,
        trust_remote_code=True,
    )
    
    print("✅ 模型加载完成！")
    print(f"   - 设备: {model.device}")
    print(f"   - 数据类型: {model.dtype}")
    
    return model, tokenizer


def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
):
    """
    生成代码
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数
        top_p: nucleus sampling
        top_k: top-k sampling
        repetition_penalty: 重复惩罚
        do_sample: 是否采样
    
    Returns:
        生成的文本
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model, tokenizer):
    """交互式模式"""
    print("\n" + "="*60)
    print("🤖 进入交互式代码生成模式")
    print("="*60)
    print("输入代码提示，模型将自动补全")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空历史")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("\n📝 请输入代码提示: ").strip()
            
            if not prompt:
                continue
                
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 退出交互模式")
                break
                
            if prompt.lower() == 'clear':
                print("\033c", end="")  # 清屏
                continue
            
            print("\n🔄 生成中...")
            generated = generate_code(model, tokenizer, prompt)
            
            print("\n" + "─"*60)
            print("📄 生成结果:")
            print("─"*60)
            print(generated)
            print("─"*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 退出交互模式")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def batch_inference(model, tokenizer, prompts_file: str, output_file: str):
    """批量推理"""
    print(f"\n📂 从文件读取提示: {prompts_file}")
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"📊 共 {len(prompts)} 个提示")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n🔄 [{i}/{len(prompts)}] 生成中...")
        print(f"提示: {prompt[:50]}...")
        
        generated = generate_code(model, tokenizer, prompt)
        results.append({
            "prompt": prompt,
            "generated": generated,
        })
    
    # 保存结果
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 批量推理完成！结果保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LoRA 模型推理")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基础模型路径",
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="LoRA adapter 路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "single", "batch"],
        help="推理模式: interactive（交互）, single（单次）, batch（批量）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="单次推理的提示（mode=single 时使用）",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="批量推理的提示文件（mode=batch 时使用）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./inference_results.json",
        help="批量推理的输出文件",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="nucleus sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备: auto, cuda:0, cpu 等",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="数据类型",
    )
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_lora_model(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        device=args.device,
        torch_dtype=args.dtype,
    )
    
    # 根据模式执行
    if args.mode == "interactive":
        interactive_mode(model, tokenizer)
    
    elif args.mode == "single":
        if not args.prompt:
            print("❌ single 模式需要提供 --prompt 参数")
            return 1
        
        print(f"\n📝 提示: {args.prompt}")
        print("\n🔄 生成中...")
        
        generated = generate_code(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        print("\n" + "─"*60)
        print("📄 生成结果:")
        print("─"*60)
        print(generated)
        print("─"*60)
    
    elif args.mode == "batch":
        if not args.prompts_file:
            print("❌ batch 模式需要提供 --prompts_file 参数")
            return 1
        
        batch_inference(model, tokenizer, args.prompts_file, args.output_file)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

