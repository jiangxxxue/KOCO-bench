#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_metrics.py - 聚合多个测试实例的评估指标

计算多个测试实例的综合 pass@1 和 avg_pass_ratio
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_metrics(file_path: str) -> Dict[str, Any]:
    """加载单个 metrics 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_test_examples(model_dir: str) -> List[str]:
    """
    自动发现目录下的所有测试实例
    
    通过扫描 *result.metrics.json 文件来提取测试实例名称
    
    Args:
        model_dir: 模型输出目录路径
    
    Returns:
        测试实例名称列表
    """
    import re
    
    model_path = Path(model_dir)
    examples = []
    
    # 查找所有 metrics 文件
    pattern = "*result.metrics.json"
    for metrics_file in model_path.glob(pattern):
        filename = metrics_file.name
        
        # 尝试匹配 algorithm_methods_data_{example}_result.metrics.json 格式
        match = re.match(r'algorithm_methods_data_(.+?)_result\.metrics\.json', filename)
        if match:
            example_name = match.group(1)
            examples.append(example_name)
            continue
        
        # 尝试匹配其他格式: {prefix}_{example}_result.metrics.json
        # 去掉 _result.metrics.json 后缀，取最后一个下划线后的部分
        base_name = filename.replace('_result.metrics.json', '')
        if '_' in base_name:
            # 取最后一个下划线后的部分作为 example 名称
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2:
                example_name = parts[1]
                examples.append(example_name)
    
    # 去重并排序
    examples = sorted(set(examples))
    
    return examples


def aggregate_metrics(
    model_dir: str,
    test_examples: List[str] = None,
    framework: str = None,
) -> Dict[str, Any]:
    """
    聚合多个测试实例的指标
    
    Args:
        model_dir: 模型输出目录路径
        test_examples: 测试实例名称列表（可选，不指定则自动发现目录下所有实例）
        framework: 框架名称（可选，用于文件名匹配）
    
    Returns:
        聚合后的指标字典
    """
    model_path = Path(model_dir)
    
    # 如果未指定 test_examples，则自动发现
    if test_examples is None or len(test_examples) == 0:
        test_examples = discover_test_examples(model_dir)
        if not test_examples:
            raise ValueError(f"在目录 {model_dir} 中未找到任何 metrics 文件")
        print(f"📋 自动发现 {len(test_examples)} 个测试实例: {', '.join(test_examples)}")
    
    # 收集所有实例的指标
    all_metrics = []
    total_functions = 0
    total_tests = 0
    total_passed = 0
    weighted_pass_at_1 = 0.0  # 修改：使用加权 pass@1
    weighted_avg_pass_ratio = 0.0
    
    missing_files = []
    
    for example in test_examples:
        # 构建 metrics 文件路径
        if framework:
            metrics_file = model_path / f"algorithm_methods_data_{example}_result.metrics.json"
        else:
            # 尝试查找匹配的文件
            pattern = f"*{example}*result.metrics.json"
            matches = list(model_path.glob(pattern))
            if matches:
                metrics_file = matches[0]
            else:
                metrics_file = model_path / f"algorithm_methods_data_{example}_result.metrics.json"
        
        # 加载指标
        if not metrics_file.exists():
            print(f"⚠️  警告: 未找到 {example} 的 metrics 文件: {metrics_file}")
            missing_files.append(example)
            continue
        
        try:
            metrics = load_metrics(str(metrics_file))
            all_metrics.append({
                'example': example,
                'metrics': metrics
            })
            
            # 累加统计
            num_funcs = metrics.get('total_functions', 0)
            total_functions += num_funcs
            total_tests += metrics.get('total_tests', 0)
            total_passed += metrics.get('total_passed', 0)
            
            # 修改：加权平均 pass@1（按函数数量加权）
            pass_at_1 = metrics.get('pass_at_k', {}).get('pass@1', 0.0)
            weighted_pass_at_1 += pass_at_1 * num_funcs
            
            # 加权平均 avg_pass_ratio（按函数数量加权）
            avg_ratio = metrics.get('avg_pass_ratio', 0.0)
            weighted_avg_pass_ratio += avg_ratio * num_funcs
            
            print(f"✓ {example}: {num_funcs} 函数, "
                  f"pass@1={pass_at_1:.4f}, "
                  f"avg_pass_ratio={avg_ratio:.4f}")
        
        except Exception as e:
            print(f"❌ 错误: 无法加载 {example} 的 metrics: {e}")
            missing_files.append(example)
    
    if not all_metrics:
        raise ValueError("没有找到任何有效的 metrics 文件")
    
    # 计算综合指标
    # 修改：pass@1 使用加权平均（而不是简单的 total_passed / total_functions）
    aggregate_pass_at_1 = weighted_pass_at_1 / total_functions if total_functions > 0 else 0.0
    
    # avg_pass_ratio: 加权平均
    aggregate_avg_pass_ratio = weighted_avg_pass_ratio / total_functions if total_functions > 0 else 0.0
    
    result = {
        'model_dir': str(model_path),
        'test_examples': test_examples,
        'valid_examples': [m['example'] for m in all_metrics],
        'missing_examples': missing_files,
        'aggregate_metrics': {
            'total_functions': total_functions,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'pass@1': aggregate_pass_at_1,
            'avg_pass_ratio': aggregate_avg_pass_ratio,
        },
        'individual_metrics': all_metrics
    }
    
    return result


def print_summary(result: Dict[str, Any]):
    """打印汇总结果"""
    print("\n" + "=" * 70)
    print("📊 综合指标汇总")
    print("=" * 70)
    print(f"模型目录: {result['model_dir']}")
    print(f"测试实例: {', '.join(result['test_examples'])}")
    
    if result['missing_examples']:
        print(f"⚠️  缺失实例: {', '.join(result['missing_examples'])}")
    
    print("\n" + "-" * 70)
    print("综合结果:")
    print("-" * 70)
    
    agg = result['aggregate_metrics']
    print(f"总函数数:     {agg['total_functions']}")
    print(f"总测试数:     {agg['total_tests']}")
    print(f"通过函数数:   {agg['total_passed']}")
    print(f"pass@1:       {agg['pass@1']:.4f} ({agg['pass@1']*100:.2f}%)")
    print(f"avg_pass_ratio: {agg['avg_pass_ratio']:.4f}")
    
    print("\n" + "-" * 70)
    print("各实例详情:")
    print("-" * 70)
    for item in result['individual_metrics']:
        example = item['example']
        m = item['metrics']
        print(f"\n{example}:")
        print(f"  函数数: {m.get('total_functions', 0)}")
        print(f"  通过数: {m.get('total_passed', 0)}")
        print(f"  pass@1: {m.get('pass_at_k', {}).get('pass@1', 0.0):.4f}")
        print(f"  avg_pass_ratio: {m.get('avg_pass_ratio', 0.0):.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="聚合多个测试实例的评估指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 自动发现所有测试实例（推荐）
  python aggregate_metrics.py \\
    --model_dir scripts/data/verl/qwen2.5-coder-32b-instruct-simple

  # 指定特定测试实例
  python aggregate_metrics.py \\
    --model_dir scripts/data/verl/qwen2.5-coder-32b-instruct-simple \\
    --test_examples prime ARES LUFFY PURE

  # 指定框架名称
  python aggregate_metrics.py \\
    --model_dir scripts/data/verl/qwen2.5-coder-7b-lora \\
    --test_examples prime ARES LUFFY PURE \\
    --framework verl

  # 保存结果到文件
  python aggregate_metrics.py \\
    --model_dir scripts/data/verl/qwen2.5-coder-32b-instruct-simple \\
    --output aggregate_result.json

输出说明:
  - pass@1: 所有实例中通过的函数数 / 总函数数
  - avg_pass_ratio: 所有实例的 avg_pass_ratio 按函数数加权平均
        """
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='模型输出目录路径'
    )
    parser.add_argument(
        '--test_examples',
        type=str,
        nargs='+',
        default=None,
        help='测试实例名称列表（空格分隔）。不指定则自动发现目录下所有实例'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default=None,
        help='框架名称（可选）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出 JSON 文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    try:
        # 聚合指标
        result = aggregate_metrics(
            model_dir=args.model_dir,
            test_examples=args.test_examples,
            framework=args.framework,
        )
        
        # 打印汇总
        print_summary(result)
        
        # 保存到文件
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {output_path}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

