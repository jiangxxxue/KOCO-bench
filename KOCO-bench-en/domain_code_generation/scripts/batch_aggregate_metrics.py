#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_aggregate_metrics.py - 批量聚合多个模型的评估指标

一次性处理多个模型目录，生成汇总表格
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# 导入 aggregate_metrics 模块
from aggregate_metrics import aggregate_metrics, discover_test_examples


def batch_aggregate(
    base_dir: str,
    model_names: List[str],
    test_examples: List[str] = None,
    framework: str = None,
    output_csv: str = None,
) -> List[Dict[str, Any]]:
    """
    批量聚合多个模型的指标
    
    Args:
        base_dir: 基础目录路径
        model_names: 模型名称列表
        test_examples: 测试实例列表（可选，不指定则自动发现每个模型目录下的实例）
        framework: 框架名称
        output_csv: 输出 CSV 文件路径（可选）
    
    Returns:
        所有模型的聚合结果列表
    """
    base_path = Path(base_dir)
    all_results = []
    
    # 用于 CSV 输出的测试实例列表（在自动发现模式下会动态收集）
    all_test_examples = set()
    
    print("=" * 80)
    print("📊 批量聚合评估指标")
    print("=" * 80)
    print(f"基础目录: {base_dir}")
    print(f"模型数量: {len(model_names)}")
    if test_examples:
        print(f"测试实例: {', '.join(test_examples)}")
    else:
        print("测试实例: (自动发现)")
    print("=" * 80)
    print()
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n[{i}/{len(model_names)}] 处理模型: {model_name}")
        print("-" * 80)
        
        model_dir = base_path / model_name
        
        if not model_dir.exists():
            print(f"⚠️  警告: 模型目录不存在: {model_dir}")
            continue
        
        try:
            result = aggregate_metrics(
                model_dir=str(model_dir),
                test_examples=test_examples,
                framework=framework,
            )
            
            # 添加模型名称
            result['model_name'] = model_name
            all_results.append(result)
            
            # 收集所有测试实例（用于 CSV 输出）
            for item in result['individual_metrics']:
                all_test_examples.add(item['example'])
            
            # 打印简要结果
            agg = result['aggregate_metrics']
            print(f"✓ pass@1: {agg['pass@1']:.4f}, avg_pass_ratio: {agg['avg_pass_ratio']:.4f}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            continue
    
    if not all_results:
        raise ValueError("没有成功处理任何模型")
    
    # 生成汇总表格
    print("\n" + "=" * 80)
    print("📋 汇总表格")
    print("=" * 80)
    print()
    
    # 表头
    header = f"{'模型名称':<30} {'总函数数':>10} {'通过数':>10} {'pass@1':>12} {'avg_pass_ratio':>15}"
    print(header)
    print("-" * 80)
    
    # 数据行
    for result in all_results:
        model_name = result['model_name']
        agg = result['aggregate_metrics']
        
        row = (f"{model_name:<30} "
               f"{agg['total_functions']:>10} "
               f"{agg['total_passed']:>10} "
               f"{agg['pass@1']:>12.4f} "
               f"{agg['avg_pass_ratio']:>15.4f}")
        print(row)
    
    print("=" * 80)
    
    # 保存 CSV
    if output_csv:
        # 使用指定的 test_examples 或收集到的所有 test_examples
        csv_examples = test_examples if test_examples else sorted(all_test_examples)
        save_csv(all_results, output_csv, csv_examples)
        print(f"\n💾 CSV 已保存到: {output_csv}")
    
    return all_results


def save_csv(results: List[Dict[str, Any]], output_path: str, test_examples: List[str]):
    """保存结果为 CSV 文件"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ['model_name', 'total_functions', 'total_passed', 'pass@1', 'avg_pass_ratio']
        
        # 为每个测试实例添加列
        for example in test_examples:
            header.extend([
                f'{example}_functions',
                f'{example}_passed',
                f'{example}_pass@1',
                f'{example}_avg_pass_ratio'
            ])
        
        writer.writerow(header)
        
        # 写入数据
        for result in results:
            agg = result['aggregate_metrics']
            row = [
                result['model_name'],
                agg['total_functions'],
                agg['total_passed'],
                f"{agg['pass@1']:.4f}",
                f"{agg['avg_pass_ratio']:.4f}",
            ]
            
            # 添加每个实例的详细数据
            individual_dict = {item['example']: item['metrics'] 
                             for item in result['individual_metrics']}
            
            for example in test_examples:
                if example in individual_dict:
                    m = individual_dict[example]
                    row.extend([
                        m.get('total_functions', 0),
                        m.get('total_passed', 0),
                        f"{m.get('pass_at_k', {}).get('pass@1', 0.0):.4f}",
                        f"{m.get('avg_pass_ratio', 0.0):.4f}",
                    ])
                else:
                    row.extend(['', '', '', ''])
            
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="批量聚合多个模型的评估指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 自动发现所有测试实例（推荐）
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/verl \\
    --model_names qwen2.5-coder-7b qwen2.5-coder-32b qwen2.5-coder-7b-lora

  # 指定特定测试实例
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/verl \\
    --model_names qwen2.5-coder-7b qwen2.5-coder-32b qwen2.5-coder-7b-lora \\
    --test_examples prime ARES LUFFY PURE

  # 保存为 CSV
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/verl \\
    --model_names qwen2.5-coder-7b qwen2.5-coder-32b \\
    --output_csv verl_rl_comparison.csv

  # 使用通配符（需要 shell 支持）
  python batch_aggregate_metrics.py \\
    --base_dir scripts/data/verl \\
    --model_names qwen2.5-coder-*
        """
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='基础目录路径'
    )
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        required=True,
        help='模型名称列表（空格分隔）'
    )
    parser.add_argument(
        '--test_examples',
        type=str,
        nargs='+',
        default=None,
        help='测试实例名称列表（空格分隔）。不指定则自动发现每个模型目录下的实例'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default=None,
        help='框架名称（可选）'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='输出 CSV 文件路径（可选）'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        default=None,
        help='输出 JSON 文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    try:
        # 批量聚合
        results = batch_aggregate(
            base_dir=args.base_dir,
            model_names=args.model_names,
            test_examples=args.test_examples,
            framework=args.framework,
            output_csv=args.output_csv,
        )
        
        # 保存 JSON
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 JSON 已保存到: {args.output_json}")
        
        print("\n✅ 批量聚合完成！")
        return 0
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

