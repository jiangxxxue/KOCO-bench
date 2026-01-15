#!/usr/bin/env python3
"""
代码补全推理客户端
通过 HTTP 请求调用推理服务器生成代码补全
"""

import json
import requests
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================
# 辅助函数
# ========================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过第 {line_num} 行（JSON 解析错误）: {e}")
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存为 JSONL 文件"""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"✓ 结果已保存到 {file_path}")

def prepare_prompts(data: List[Dict[str, Any]], prompt_field: str = "prompt") -> List[Any]:
    """
    准备推理的 prompts
    
    支持的输入格式：
    1. {"prompt": [{"role": "user", "content": "..."}]} - 对话格式（保持原样发送）
    2. {"prompt": "def hello():\\n    "} - 直接代码前缀
    3. {"prefix": "def hello():\\n    "} - 使用 prefix 字段
    
    注意：对话格式会原样发送到服务器，由服务器端使用 tokenizer.apply_chat_template 处理
    """
    prompts = []
    
    for i, sample in enumerate(data):
        try:
            prompt_value = sample.get(prompt_field) or sample.get("prefix") or sample.get("code")
            
            if prompt_value is None:
                logger.warning(f"样本 {i} 缺少有效的 prompt 字段，使用空字符串")
                prompts.append("")
                continue
            
            # 格式1: 对话格式 - 直接发送原始格式到服务器
            # 服务器端会使用 tokenizer.apply_chat_template 处理
            if isinstance(prompt_value, list):
                prompts.append(prompt_value)
            
            # 格式2: 直接的代码字符串
            elif isinstance(prompt_value, str):
                prompts.append(prompt_value)
            
            else:
                logger.warning(f"样本 {i} 的 prompt 格式不支持，使用空字符串")
                prompts.append("")
        
        except Exception as e:
            logger.error(f"处理样本 {i} 时出错: {e}")
            prompts.append("")
    
    return prompts

def check_server_health(server_url: str, max_retries: int = 5, retry_delay: int = 2) -> bool:
    """
    检查服务器是否健康
    
    Args:
        server_url: 服务器 URL
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    
    Returns:
        服务器是否健康
    """
    health_url = f"{server_url}/health"
    
    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ 服务器健康检查通过")
                logger.info(f"  模型: {data.get('model', 'N/A')}")
                logger.info(f"  设备: {data.get('device', 'N/A')}")
                return True
            else:
                logger.warning(f"服务器返回异常状态码: {response.status_code}")
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                logger.warning(f"无法连接到服务器，{retry_delay} 秒后重试... ({i+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"❌ 无法连接到服务器: {server_url}")
                return False
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    return False

def generate_completions(
    prompts: List[Any],
    server_url: str,
    num_completions: int = 1,
    max_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    batch_size: int = 1
) -> List[List[str]]:
    """
    通过 HTTP 请求生成代码补全
    
    Args:
        prompts: 待生成的 prompts 列表
        server_url: 服务器 URL
        num_completions: 每个 prompt 生成的补全数量
        max_tokens: 每个补全的最大 token 数
        temperature: 采样温度
        top_p: Top-p 采样参数
        batch_size: 批次大小（每次请求的 prompts 数量）
    
    Returns:
        生成的补全列表
    """
    generate_url = f"{server_url}/generate"
    all_completions = []
    
    total = len(prompts)
    logger.info(f"开始生成 {total} 个样本的补全（批次大小: {batch_size}）...")
    
    # 分批处理
    for i in range(0, total, batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"正在处理批次 {batch_num}/{total_batches} ({len(batch_prompts)} 个 prompts)...")
        
        try:
            # 构造请求
            request_data = {
                "prompts": batch_prompts,
                "num_completions": num_completions,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            # 发送请求
            response = requests.post(
                generate_url,
                json=request_data,
                timeout=600  # 10 分钟超时
            )
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                batch_completions = result["completions"]
                all_completions.extend(batch_completions)
                logger.info(f"✓ 批次 {batch_num} 完成")
            else:
                logger.error(f"❌ 批次 {batch_num} 失败: {response.status_code}")
                logger.error(f"错误信息: {response.text}")
                # 添加空补全
                all_completions.extend([[""] * num_completions] * len(batch_prompts))
        
        except requests.exceptions.Timeout:
            logger.error(f"❌ 批次 {batch_num} 超时")
            all_completions.extend([[""] * num_completions] * len(batch_prompts))
        except Exception as e:
            logger.error(f"❌ 批次 {batch_num} 出错: {e}")
            all_completions.extend([[""] * num_completions] * len(batch_prompts))
    
    logger.info(f"✓ 生成完成")
    return all_completions

# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="代码补全推理客户端 - 通过 HTTP 请求调用推理服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python inference_client.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl

  # 生成多个补全
  python inference_client.py \\
    --server_url http://localhost:8000 \\
    --input_file ../data/algorithm_methods_data.jsonl \\
    --output_file ../data/algorithm_methods_output.jsonl \\
    --num_completions 4 \\
    --temperature 0.2

  # 使用批处理
  python inference_client.py \\
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
        default="http://localhost:8000",
        help="推理服务器 URL（默认: http://localhost:8000）"
    )
    
    # 数据文件
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入 JSONL 文件（每行包含 prompt/prefix/code 字段）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出 JSONL 文件（不指定则自动生成）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="模型显示名称（用于输出目录）"
    )
    
    # 生成参数
    parser.add_argument(
        "--num_completions",
        type=int,
        default=1,
        help="每个 prompt 生成的补全数量（默认: 1）"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="每个补全的最大 token 数（默认: 512）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="采样温度（0=贪婪解码，默认: 0.2）"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p 采样参数（默认: 0.95）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批次大小（每次请求的 prompts 数量，默认: 1）"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 检查服务器健康状态
    logger.info("="*60)
    logger.info("正在连接推理服务器...")
    logger.info("="*60)
    logger.info(f"服务器 URL: {args.server_url}")
    
    if not check_server_health(args.server_url):
        logger.error("❌ 服务器不可用，请先启动推理服务器")
        logger.error(f"   python inference_server.py --model_path <模型路径>")
        return
    
    # 自动生成输出文件路径
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # 从服务器获取模型名称
        try:
            response = requests.get(f"{args.server_url}/health")
            model_name = response.json().get("model", "model")
        except:
            model_name = args.model_name or "model"
        
        # 在输入文件的同级目录下创建模型子目录
        input_dir = input_file.parent
        model_output_dir = input_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 输出文件名格式: 移除 .jsonl 后缀，添加 _output.jsonl
        if input_file.suffix == '.jsonl':
            base_name = input_file.stem
            output_file = model_output_dir / f"{base_name}_output.jsonl"
        else:
            output_file = model_output_dir / f"{input_file.stem}_output{input_file.suffix}"
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("代码补全推理")
    logger.info("="*60)
    logger.info(f"服务器: {args.server_url}")
    logger.info(f"输入文件: {args.input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"生成数量: {args.num_completions}")
    logger.info(f"最大 tokens: {args.max_tokens}")
    logger.info(f"温度: {args.temperature}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info("="*60)
    
    # 加载输入数据
    logger.info(f"正在加载输入数据...")
    input_data = load_jsonl(args.input_file)
    logger.info(f"✓ 加载了 {len(input_data)} 个样本")
    
    if len(input_data) == 0:
        logger.error("❌ 输入文件为空")
        return
    
    # 准备 prompts
    logger.info(f"正在准备 prompts...")
    prompts = prepare_prompts(input_data, "prompt")
    logger.info(f"✓ 准备了 {len(prompts)} 个 prompts")
    
    # 显示第一个样本示例
    if prompts and prompts[0]:
        logger.info(f"\n第一个 prompt 示例:")
        logger.info("="*60)
        # 处理对话格式（list）和字符串格式
        first_prompt = prompts[0]
        if isinstance(first_prompt, list):
            # 对话格式：显示简要信息
            logger.info(f"对话格式，包含 {len(first_prompt)} 条消息")
            for msg in first_prompt[:2]:  # 只显示前两条消息
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                logger.info(f"  [{role}]: {content[:100]}..." if len(content) > 100 else f"  [{role}]: {content}")
        else:
            # 字符串格式：显示前 300 字符
            logger.info(first_prompt[:300] + ("..." if len(first_prompt) > 300 else ""))
        logger.info("="*60 + "\n")
    
    # 生成补全
    try:
        all_completions = generate_completions(
            prompts,
            args.server_url,
            num_completions=args.num_completions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size
        )
        
        # 组装结果
        logger.info("正在组装输出结果...")
        results = []
        for sample, completions in zip(input_data, all_completions):
            result = sample.copy()
            result["completions"] = completions
            result["server"] = args.server_url
            result["num_completions"] = len(completions)
            results.append(result)
        
        # 保存结果
        save_jsonl(results, str(output_file))
        
        # 统计信息
        logger.info("\n" + "="*60)
        logger.info(f"✓ 成功生成 {len(results)} 个样本的补全")
        logger.info(f"✓ 每个样本生成 {args.num_completions} 个补全")
        logger.info(f"✓ 总共生成 {len(results) * args.num_completions} 个补全")
        logger.info("="*60)
        
        # 显示第一个补全示例
        if results and results[0]["completions"] and results[0]["completions"][0]:
            first_completion = results[0]["completions"][0]
            logger.info(f"\n第一个补全示例:")
            logger.info("="*60)
            logger.info(first_completion[:300] + ("..." if len(first_completion) > 300 else ""))
            logger.info("="*60 + "\n")
    
    except Exception as e:
        logger.error(f"❌ 生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

