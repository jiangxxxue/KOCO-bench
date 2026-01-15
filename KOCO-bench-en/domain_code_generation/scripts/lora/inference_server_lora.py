#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_server_lora.py — LoRA 模型推理服务器

基于 FastAPI 提供代码生成服务，支持 LoRA adapter 加载
"""

import os
import sys
import argparse
import torch
import uvicorn
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


# ========================================
# 全局变量
# ========================================
model = None
tokenizer = None
generation_config = None
base_model_path = None
lora_adapter_path = None


# ========================================
# 请求/响应模型
# ========================================

class GenerationRequest(BaseModel):
    """生成请求"""
    prompts: List[Any] = Field(..., description="输入提示列表（可以是字符串或对话列表）")
    num_completions: int = Field(1, ge=1, le=10, description="每个提示生成的补全数量")
    max_tokens: int = Field(512, ge=1, le=4096, description="生成的最大token数")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Top-p采样")
    do_sample: bool = Field(True, description="是否使用采样")


class GenerationResponse(BaseModel):
    """生成响应"""
    completions: List[List[str]] = Field(..., description="生成结果，外层列表对应输入提示，内层列表对应每个提示的多个补全")
    model: str = Field(..., description="使用的模型")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field("healthy", description="服务状态")
    model: str = Field(..., description="加载的模型")
    base_model: str = Field(..., description="基础模型路径")
    lora_adapter: str = Field(..., description="LoRA adapter路径")
    device: str = Field(..., description="设备信息")


# ========================================
# 模型加载
# ========================================

def load_lora_model(
    base_model: str,
    lora_adapter: str,
    device: str = "auto",
    torch_dtype: str = "bfloat16",
    max_context_len: int = 4096,
):
    """
    加载基础模型 + LoRA adapter
    
    Args:
        base_model: 基础模型路径
        lora_adapter: LoRA adapter 路径
        device: 设备，"auto" 或 "cuda:0" 等
        torch_dtype: 数据类型
        max_context_len: 最大上下文长度
    
    Returns:
        model, tokenizer, generation_config
    """
    print(f"📦 加载基础模型: {base_model}")
    
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
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # 避免 flash-attention 问题
    )
    
    print(f"📦 加载 LoRA adapter: {lora_adapter}")
    
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(
        base,
        lora_adapter,
        torch_dtype=dtype,
    )
    
    # 合并 LoRA 权重以提高推理速度（可选）
    # model = model.merge_and_unload()
    
    print(f"📦 加载 tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left",  # 批量生成时需要左填充
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 生成配置
    gen_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print("✅ 模型加载完成")
    print(f"  - 基础模型: {base_model}")
    print(f"  - LoRA adapter: {lora_adapter}")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")
    
    return model, tokenizer, gen_config


# ========================================
# 生成函数
# ========================================

def format_prompt(prompt_data: Any) -> str:
    """
    格式化 prompt 为字符串
    
    Args:
        prompt_data: 可以是字符串或对话列表
    
    Returns:
        格式化后的字符串
    """
    global tokenizer
    
    if isinstance(prompt_data, str):
        return prompt_data
    elif isinstance(prompt_data, list):
        # 对话列表格式：[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        # 使用 tokenizer 的 apply_chat_template 方法（如果可用）
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                return tokenizer.apply_chat_template(
                    prompt_data,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # 如果失败，回退到简单格式化
                pass
        
        # 简单格式化
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
        return "\n\n".join(formatted_parts) + "\n\nAssistant: "
    else:
        return str(prompt_data)


def generate_completions(
    prompts: List[Any],
    num_completions: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> List[List[str]]:
    """
    批量生成代码补全
    
    Args:
        prompts: 提示列表（可以是字符串或对话列表）
        num_completions: 每个提示生成的补全数量
        max_tokens: 最大生成token数
        temperature: 温度参数
        top_p: Top-p采样
        do_sample: 是否采样
    
    Returns:
        补全列表，外层对应每个提示，内层对应每个提示的多个补全
    """
    global model, tokenizer, generation_config
    
    if model is None or tokenizer is None:
        raise RuntimeError("模型未加载")
    
    results = []
    
    for prompt_data in prompts:
        prompt_completions = []
        
        # 格式化 prompt
        prompt = format_prompt(prompt_data)
        
        # 为每个补全单独生成
        for _ in range(num_completions):
            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # 解码
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            prompt_completions.append(completion)
        
        results.append(prompt_completions)
    
    return results


# ========================================
# FastAPI 应用
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载模型
    global model, tokenizer, generation_config, base_model_path, lora_adapter_path
    
    print("=" * 60)
    print("🚀 启动 LoRA 推理服务器")
    print("=" * 60)
    
    model, tokenizer, generation_config = load_lora_model(
        base_model=base_model_path,
        lora_adapter=lora_adapter_path,
        device=app.state.device,
        torch_dtype=app.state.torch_dtype,
        max_context_len=app.state.max_context_len,
    )
    
    print("=" * 60)
    print("✅ 服务器启动完成")
    print("=" * 60)
    
    yield
    
    # 关闭时清理
    print("🛑 关闭服务器...")


app = FastAPI(
    title="LoRA 代码生成推理服务器",
    description="支持 LoRA adapter 的代码补全推理服务",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    device_name = str(next(model.parameters()).device)
    
    return HealthResponse(
        status="healthy",
        model=f"{base_model_path} + {lora_adapter_path}",
        base_model=base_model_path,
        lora_adapter=lora_adapter_path,
        device=device_name,
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """代码生成端点"""
    try:
        completions = generate_completions(
            prompts=request.prompts,
            num_completions=request.num_completions,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample,
        )
        
        return GenerationResponse(
            completions=completions,
            model=f"{base_model_path} + {lora_adapter_path}",
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="LoRA 代码补全推理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 启动服务器（默认端口 8000）
  python inference_server_lora.py \\
    --base_model /path/to/base/model \\
    --lora_adapter ../models/qwen2.5-coder-7b-verl-lora

  # 指定端口
  python inference_server_lora.py \\
    --base_model /path/to/base/model \\
    --lora_adapter ../models/qwen2.5-coder-7b-verl-lora \\
    --port 8001

  # 测试健康检查
  curl http://localhost:8000/health

  # 测试生成
  curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompts": ["def hello():\\n    "], "num_completions": 1}'
"""
    )
    
    # 模型参数
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_adapter",
        type=str,
        required=True,
        help="LoRA adapter 路径"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备 (默认: auto)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="模型数据类型 (默认: bfloat16)"
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=4096,
        help="最大上下文长度 (默认: 4096)"
    )
    
    # 服务器参数
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)"
    )
    
    args = parser.parse_args()
    
    # 设置全局变量
    global base_model_path, lora_adapter_path
    base_model_path = args.base_model
    lora_adapter_path = args.lora_adapter
    
    # 保存参数到 app.state
    app.state.device = args.device
    app.state.torch_dtype = args.torch_dtype
    app.state.max_context_len = args.max_context_len
    
    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

