#!/usr/bin/env python3
"""
代码补全推理服务器
使用 FastAPI 提供 HTTP API，只加载一次模型，持续提供推理服务
"""

import json
import torch
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from transformers import AutoTokenizer, AutoModelForCausalLM

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# 请求/响应模型
# ========================================

class GenerationRequest(BaseModel):
    """生成请求"""
    prompts: List[Any] = Field(..., description="待生成的 prompts 列表（支持字符串或对话格式）")
    num_completions: int = Field(1, description="每个 prompt 生成的补全数量")
    max_tokens: int = Field(512, description="每个补全的最大 token 数")
    temperature: float = Field(0.2, description="采样温度")
    top_p: float = Field(0.95, description="Top-p 采样参数")

class GenerationResponse(BaseModel):
    """生成响应"""
    completions: List[List[str]] = Field(..., description="生成的补全列表")
    model: str = Field(..., description="使用的模型")
    status: str = Field("success", description="状态")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    model: str = ""
    device: str = ""

# ========================================
# 全局变量（模型和 tokenizer）
# ========================================

model = None
tokenizer = None
model_name = ""
device = ""

# ========================================
# FastAPI 应用
# ========================================

app = FastAPI(
    title="代码补全推理服务",
    description="提供代码补全推理的 HTTP API 服务",
    version="1.0.0"
)

@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 健康检查"""
    return HealthResponse(
        status="healthy",
        model=model_name,
        device=device
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点"""
    return HealthResponse(
        status="healthy",
        model=model_name,
        device=device
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    生成代码补全
    
    请求示例:
    {
        "prompts": ["def hello():\\n    "],  # 字符串格式
        或
        "prompts": [[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]],  # 对话格式
        "num_completions": 2,
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.95
    }
    """
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        logger.info(f"收到生成请求: {len(request.prompts)} 个 prompts")
        
        all_completions = []
        
        for i, prompt in enumerate(request.prompts):
            if (i + 1) % 10 == 0:
                logger.info(f"正在处理 {i+1}/{len(request.prompts)}...")
            
            # 处理 prompt 格式
            if isinstance(prompt, list):
                # 对话格式：使用 chat template
                if hasattr(tokenizer, 'apply_chat_template'):
                    try:
                        formatted_prompt = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    except Exception as e:
                        logger.warning(f"样本 {i} 使用 chat template 失败: {e}，回退到拼接消息")
                        # 回退：简单拼接消息内容
                        formatted_prompt = "\n\n".join(msg.get("content", "") for msg in prompt if msg.get("content"))
                else:
                    # 没有 chat template，简单拼接
                    formatted_prompt = "\n\n".join(msg.get("content", "") for msg in prompt if msg.get("content"))
            else:
                # 字符串格式：直接使用
                formatted_prompt = prompt
            
            # Tokenize 输入
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,  # 使用固定的最大长度
            ).to(model.device)
            
            input_len = inputs.input_ids.shape[1]
            
            # 生成多个补全
            completions = []
            for _ in range(request.num_completions):
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature if request.temperature > 0 else 1.0,
                        top_p=request.top_p,
                        do_sample=request.temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.0,
                    )
                
                # 只提取新生成的部分
                generated_tokens = outputs[0][input_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                completions.append(generated_text)
            
            all_completions.append(completions)
        
        logger.info(f"✓ 生成完成")
        
        return GenerationResponse(
            completions=all_completions,
            model=model_name,
            status="success"
        )
    
    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# 模型加载
# ========================================

def load_model(model_path: str, max_context_len: int = 4096):
    """加载模型和 tokenizer"""
    global model, tokenizer, model_name, device
    
    logger.info("="*60)
    logger.info("正在启动推理服务器...")
    logger.info("="*60)
    logger.info(f"模型路径: {model_path}")
    logger.info(f"最大上下文长度: {max_context_len}")
    
    # 加载 tokenizer
    logger.info("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✓ Tokenizer 加载完成")
    
    # 加载模型
    logger.info("正在加载模型...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"使用精度: {torch_dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    device = str(model.device)
    model_name = Path(model_path).name
    
    logger.info(f"✓ 模型加载完成")
    logger.info(f"设备: {device}")
    logger.info(f"模型名称: {model_name}")
    logger.info("="*60)
    logger.info("✅ 服务器准备就绪！")
    logger.info("="*60)

# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="代码补全推理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 启动服务器（默认端口 8000）
  python inference_server.py --model_path ../models/qwen2.5-coder-7b-verl-ntp

  # 指定端口
  python inference_server.py \\
    --model_path ../models/qwen2.5-coder-7b-verl-ntp \\
    --port 8001

  # 测试健康检查
  curl http://localhost:8000/health

  # 测试生成
  curl -X POST http://localhost:8000/generate \\
    -H "Content-Type: application/json" \\
    -d '{"prompts": ["def hello():\\n    "], "num_completions": 1}'
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="本地模型路径"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口（默认: 8000）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址（默认: 0.0.0.0）"
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=4096,
        help="最大上下文长度（默认: 4096）"
    )
    
    args = parser.parse_args()
    
    # 验证模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"❌ 模型路径不存在: {model_path}")
        return
    
    # 加载模型
    load_model(str(model_path), args.max_context_len)
    
    # 启动服务器
    logger.info(f"正在启动服务器: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()

