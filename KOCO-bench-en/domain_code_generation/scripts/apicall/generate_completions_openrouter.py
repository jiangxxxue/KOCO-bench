#!/usr/bin/env python3
"""
使用 OpenRouter API 生成代码补全
"""

import json
import os
import argparse
import time
from typing import List, Dict, Any
import logging
from openai import OpenAI

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_completions_openrouter(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    num_completions: int = 1,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    delay: float = 0.5,
    debug: bool = False
) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    使用 OpenRouter API 生成多个补全
    
    Args:
        client: OpenAI 客户端
        model: 模型名称
        messages: 消息列表
        num_completions: 生成数量
        max_tokens: 最大 token 数
        temperature: 温度参数
        top_p: top_p 参数
        delay: 每次调用之间的延迟
        debug: 是否启用调试模式
        
    Returns:
        (生成的文本列表, usage 信息列表)
    """
    completions = []
    usage_list = []
    
    for i in range(num_completions):
        try:
            if debug:
                logger.info(f"  📤 发送请求 {i+1}/{num_completions}")
                logger.info(f"     模型: {model}")
                logger.info(f"     消息数: {len(messages)}")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # 提取 usage 信息
            usage_info = {}
            if hasattr(response, 'usage') and response.usage:
                usage_dict = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else vars(response.usage)
                usage_info = {
                    'prompt_tokens': usage_dict.get('prompt_tokens', 0),
                    'completion_tokens': usage_dict.get('completion_tokens', 0),
                    'total_tokens': usage_dict.get('total_tokens', 0)
                }
            
            if debug:
                logger.info(f"  📥 收到响应")
                logger.info(f"     Response ID: {response.id}")
                logger.info(f"     Model: {response.model}")
                logger.info(f"     Choices: {len(response.choices)}")
                if usage_info:
                    logger.info(f"     Usage: input={usage_info.get('prompt_tokens', 0)}, output={usage_info.get('completion_tokens', 0)}, total={usage_info.get('total_tokens', 0)}")
            
            # 获取响应内容
            choice = response.choices[0]
            message = choice.message
            content = message.content
            
            # 特殊处理：检查是否有推理内容（o1/o3/o4 系列模型）
            # 这些模型可能会返回空的 content，但在其他字段中包含实际内容
            if content is None or (isinstance(content, str) and len(content.strip()) == 0):
                logger.warning(f"  ⚠️  候选 {i+1}/{num_completions} content 为空或 None")
                logger.warning(f"     finish_reason: {choice.finish_reason}")
                
                # 尝试获取完整的 message 信息
                message_dict = message.model_dump() if hasattr(message, 'model_dump') else vars(message)
                
                # 检查是否因为 token 限制导致（推理模型常见问题）
                if choice.finish_reason == "length":
                    logger.error(f"     ❌ Token 限制！推理模型消耗了所有 tokens")
                    logger.error(f"     建议：增加 --max_tokens 参数（推荐 8192 或更高）")
                    
                    # 检查是否有 reasoning 字段
                    if hasattr(message, 'reasoning') and message.reasoning:
                        reasoning_len = len(message.reasoning)
                        logger.warning(f"     ℹ️  发现推理内容（长度: {reasoning_len}），但实际代码未生成")
                
                # 记录完整信息用于调试
                logger.debug(f"     完整 message 对象: {json.dumps(message_dict, ensure_ascii=False, indent=2)}")
                
                # 检查是否有 refusal 字段
                if hasattr(message, 'refusal') and message.refusal:
                    logger.error(f"     ❌ 模型拒绝响应: {message.refusal}")
                    content = ""
                else:
                    content = "" if content is None else content
            
            content = content.strip() if content else ""
            
            if not content:
                logger.warning(f"  ⚠️  候选 {i+1}/{num_completions} 内容为空")
            
            completions.append(content)
            usage_list.append(usage_info)
            logger.info(f"  ✅ 候选 {i+1}/{num_completions} 生成成功 (长度: {len(content)})")
            
        except Exception as e:
            logger.error(f"  ❌ 候选 {i+1}/{num_completions} 生成失败: {e}")
            completions.append("")
            usage_list.append({})
        
        # 延迟避免限流
        if i < num_completions - 1:
            time.sleep(delay)
    
    return completions, usage_list


def main():
    parser = argparse.ArgumentParser(description="使用 OpenRouter API 生成代码补全")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENROUTER_API_KEY"),
                        help="OpenRouter API Key")
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenRouter API 基础 URL")
    parser.add_argument("--model", type=str, default="qwen/qwen2.5-coder-7b-instruct",
                        help="模型名称")
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入 JSONL 文件")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出 JSONL 文件")
    parser.add_argument("--num_completions", type=int, default=1,
                        help="每个样本生成数量")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="top_p 参数")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="API 调用延迟（秒）")
    parser.add_argument("--debug", action="store_true",
                        help="启用调试模式，显示详细请求/响应信息")
    
    args = parser.parse_args()
    
    # 验证 API Key
    if not args.api_key:
        logger.error("❌ 错误: 未设置 OPENROUTER_API_KEY")
        logger.error("请运行: export OPENROUTER_API_KEY='your-api-key'")
        return 1
    
    # 加载数据
    logger.info(f"📖 加载数据: {args.input_file}")
    input_data = load_jsonl(args.input_file)
    logger.info(f"✅ 加载了 {len(input_data)} 个样本")
    
    # 初始化客户端
    logger.info(f"🔧 初始化 OpenRouter 客户端")
    logger.info(f"   模型: {args.model}")
    logger.info(f"   每个样本生成: {args.num_completions} 个候选")
    
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # 生成代码
    logger.info(f"🚀 开始生成...")
    logger.info("")
    
    results = []
    success_count = 0
    fail_count = 0
    total_prompt_tokens_all = 0
    total_completion_tokens_all = 0
    total_tokens_all = 0
    
    for i, sample in enumerate(input_data, 1):
        function_name = sample.get('function_name', f'function_{i}')
        logger.info(f"[{i}/{len(input_data)}] 处理: {function_name}")
        
        if 'prompt' not in sample:
            logger.warning(f"  ⚠️  跳过: 缺少 prompt 字段")
            sample['completions'] = []
            results.append(sample)
            continue
        
        # 生成补全
        completions, usage_list = generate_completions_openrouter(
            client=client,
            model=args.model,
            messages=sample['prompt'],
            num_completions=args.num_completions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            delay=args.delay,
            debug=args.debug
        )
        
        # 统计
        success = sum(1 for c in completions if c)
        fail = len(completions) - success
        success_count += success
        fail_count += fail
        
        # 计算总 token 使用量
        total_prompt_tokens = sum(u.get('prompt_tokens', 0) for u in usage_list)
        total_completion_tokens = sum(u.get('completion_tokens', 0) for u in usage_list)
        total_tokens = sum(u.get('total_tokens', 0) for u in usage_list)
        
        # 累加到全局统计
        total_prompt_tokens_all += total_prompt_tokens
        total_completion_tokens_all += total_completion_tokens
        total_tokens_all += total_tokens
        
        # 保存结果
        result = sample.copy()
        result['completions'] = completions
        result['usage'] = {
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens,
            'per_completion': usage_list  # 每个补全的详细 usage
        }
        results.append(result)
        
        logger.info(f"  📊 成功: {success}/{len(completions)}")
        if total_tokens > 0:
            logger.info(f"  🔢 Token 使用: input={total_prompt_tokens}, output={total_completion_tokens}, total={total_tokens}")
        logger.info("")
    
    # 保存结果
    logger.info(f"💾 保存结果: {args.output_file}")
    save_jsonl(results, args.output_file)
    
    # 显示统计
    total = success_count + fail_count
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 生成统计")
    logger.info("=" * 60)
    logger.info(f"总样本数: {len(input_data)}")
    logger.info(f"总生成数: {total}")
    logger.info(f"✅ 成功: {success_count}")
    logger.info(f"❌ 失败: {fail_count}")
    if total > 0:
        logger.info(f"成功率: {success_count/total*100:.1f}%")
    logger.info("")
    logger.info("🔢 Token 使用统计:")
    logger.info(f"  输入 tokens: {total_prompt_tokens_all:,}")
    logger.info(f"  输出 tokens: {total_completion_tokens_all:,}")
    logger.info(f"  总计 tokens: {total_tokens_all:,}")
    if len(input_data) > 0:
        logger.info(f"  平均每样本: {total_tokens_all/len(input_data):.1f} tokens")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

