#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_cross_framework.py - 跨框架聚合评估指标

聚合多个框架的所有测试实例，计算综合 pass@1 和 avg_pass_ratio
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# 导入现有模块
from aggregate_metrics import aggregate_metrics, discover_test_examples


def discover_frameworks(data_dir: str) -> List[str]:
    """
    自动发现数据目录下的所有框架
    
    Args:
        data_dir: 数据目录路径
    
    Returns:
        框架名称列表
    """
    data_path = Path(data_dir)
    frameworks = []
    
    for item in data_path.iterdir():
        if item.is_dir():
            frameworks.append(item.name)
    
    return sorted(frameworks)


def aggregate_cross_framework(
    model_name: str,
    data_dir: str,
    frameworks: List[str] = None,
) -> Dict[str, Any]:
    """
    跨框架聚合评估指标
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录路径（如 scripts/data）
        frameworks: 框架名称列表（可选，不指定则自动发现）
    
    Returns:
        聚合后的指标字典
    """
    data_path = Path(data_dir)
    
    # 如果未指定框架，则自动发现
    if frameworks is None or len(frameworks) == 0:
        frameworks = discover_frameworks(data_dir)
        if not frameworks:
            raise ValueError(f"在目录 {data_dir} 中未找到任何框架目录")
        print(f"📋 自动发现 {len(frameworks)} 个框架: {', '.join(frameworks)}")
    
    # 收集每个框架的结果
    all_framework_results = []
    total_functions = 0
    total_tests = 0
    total_passed = 0
    weighted_pass_at_1 = 0.0
    weighted_avg_pass_ratio = 0.0
    
    missing_frameworks = []
    valid_frameworks = []
    
    for framework in frameworks:
        model_dir = data_path / framework / model_name
        
        if not model_dir.exists():
            print(f"⚠️  警告: 框架 {framework} 下未找到模型 {model_name}: {model_dir}")
            missing_frameworks.append(framework)
            continue
        
        # 检查是否有 metrics 文件
        test_examples = discover_test_examples(str(model_dir))
        if not test_examples:
            print(f"⚠️  警告: 框架 {framework} 模型 {model_name} 下无测试实例")
            missing_frameworks.append(framework)
            continue
        
        try:
            # 聚合该框架下的所有实例
            result = aggregate_metrics(
                model_dir=str(model_dir),
                test_examples=test_examples,
                framework=framework,
            )
            
            agg = result['aggregate_metrics']
            num_funcs = agg['total_functions']
            
            # 累加统计
            total_functions += num_funcs
            total_tests += agg['total_tests']
            total_passed += agg['total_passed']
            
            # 加权累加
            weighted_pass_at_1 += agg['pass@1'] * num_funcs
            weighted_avg_pass_ratio += agg['avg_pass_ratio'] * num_funcs
            
            # 保存框架结果
            framework_result = {
                'framework': framework,
                'model_dir': str(model_dir),
                'test_examples': test_examples,
                'metrics': agg,
                'individual_metrics': result['individual_metrics']
            }
            all_framework_results.append(framework_result)
            valid_frameworks.append(framework)
            
            print(f"\n✅ {framework}: {len(test_examples)} 实例, {num_funcs} 函数")
            print(f"   pass@1: {agg['pass@1']:.4f}, avg_pass_ratio: {agg['avg_pass_ratio']:.4f}")
            
        except Exception as e:
            print(f"❌ 框架 {framework} 处理失败: {e}")
            missing_frameworks.append(framework)
            continue
    
    if not all_framework_results:
        raise ValueError("没有成功处理任何框架")
    
    # 计算综合指标
    aggregate_pass_at_1 = weighted_pass_at_1 / total_functions if total_functions > 0 else 0.0
    aggregate_avg_pass_ratio = weighted_avg_pass_ratio / total_functions if total_functions > 0 else 0.0
    
    result = {
        'model_name': model_name,
        'data_dir': str(data_path),
        'frameworks': frameworks,
        'valid_frameworks': valid_frameworks,
        'missing_frameworks': missing_frameworks,
        'aggregate_metrics': {
            'total_frameworks': len(valid_frameworks),
            'total_functions': total_functions,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'pass@1': aggregate_pass_at_1,
            'avg_pass_ratio': aggregate_avg_pass_ratio,
        },
        'framework_metrics': all_framework_results
    }
    
    return result


def print_summary(result: Dict[str, Any]):
    """打印汇总结果"""
    print("\n" + "=" * 80)
    print("📊 跨框架综合指标汇总")
    print("=" * 80)
    print(f"模型名称: {result['model_name']}")
    print(f"数据目录: {result['data_dir']}")
    print(f"框架列表: {', '.join(result['frameworks'])}")
    
    if result['missing_frameworks']:
        print(f"⚠️  缺失框架: {', '.join(result['missing_frameworks'])}")
    
    print("\n" + "-" * 80)
    print("综合结果:")
    print("-" * 80)
    
    agg = result['aggregate_metrics']
    print(f"有效框架数:   {agg['total_frameworks']}")
    print(f"总函数数:     {agg['total_functions']}")
    print(f"总测试数:     {agg['total_tests']}")
    print(f"通过函数数:   {agg['total_passed']}")
    print(f"pass@1:       {agg['pass@1']:.4f} ({agg['pass@1']*100:.2f}%)")
    print(f"avg_pass_ratio: {agg['avg_pass_ratio']:.4f}")
    
    print("\n" + "-" * 80)
    print("各框架详情:")
    print("-" * 80)
    
    # 表头
    header = f"{'框架':<25} {'实例数':>8} {'函数数':>10} {'通过数':>10} {'pass@1':>12} {'avg_pass_ratio':>15}"
    print(header)
    print("-" * 80)
    
    for fw_result in result['framework_metrics']:
        framework = fw_result['framework']
        m = fw_result['metrics']
        num_examples = len(fw_result['test_examples'])
        
        row = (f"{framework:<25} "
               f"{num_examples:>8} "
               f"{m['total_functions']:>10} "
               f"{m['total_passed']:>10} "
               f"{m['pass@1']:>12.4f} "
               f"{m['avg_pass_ratio']:>15.4f}")
        print(row)
    
    print("=" * 80)


def save_csv(result: Dict[str, Any], output_path: str):
    """保存结果为 CSV 文件"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入汇总信息
        writer.writerow(['# 跨框架聚合结果'])
        writer.writerow(['模型名称', result['model_name']])
        agg = result['aggregate_metrics']
        writer.writerow(['总框架数', agg['total_frameworks']])
        writer.writerow(['总函数数', agg['total_functions']])
        writer.writerow(['总通过数', agg['total_passed']])
        writer.writerow(['综合 pass@1', f"{agg['pass@1']:.4f}"])
        writer.writerow(['综合 avg_pass_ratio', f"{agg['avg_pass_ratio']:.4f}"])
        writer.writerow([])
        
        # 写入框架详情表头
        writer.writerow(['框架', '实例数', '函数数', '通过数', 'pass@1', 'avg_pass_ratio'])
        
        # 写入框架数据
        for fw_result in result['framework_metrics']:
            m = fw_result['metrics']
            writer.writerow([
                fw_result['framework'],
                len(fw_result['test_examples']),
                m['total_functions'],
                m['total_passed'],
                f"{m['pass@1']:.4f}",
                f"{m['avg_pass_ratio']:.4f}",
            ])


def main():
    parser = argparse.ArgumentParser(
        description="跨框架聚合评估指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 自动发现所有框架（推荐）
  python aggregate_cross_framework.py \\
    --model_name qwen2.5-coder-7b-instruct \\
    --data_dir scripts/data

  # 指定特定框架
  python aggregate_cross_framework.py \\
    --model_name qwen2.5-coder-7b-instruct \\
    --data_dir scripts/data \\
    --frameworks verl open-r1 smolagents

  # 保存结果
  python aggregate_cross_framework.py \\
    --model_name qwen2.5-coder-7b-instruct \\
    --data_dir scripts/data \\
    --output cross_framework_result.json \\
    --output_csv cross_framework_result.csv

输出说明:
  - pass@1: 所有框架所有实例中通过的函数数 / 总函数数（加权平均）
  - avg_pass_ratio: 所有框架所有实例的 avg_pass_ratio 按函数数加权平均
        """
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='模型名称（如 qwen2.5-coder-7b-instruct）'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='数据目录路径（如 scripts/data）'
    )
    parser.add_argument(
        '--frameworks',
        type=str,
        nargs='+',
        default=None,
        help='框架名称列表（空格分隔）。不指定则自动发现目录下所有框架'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出 JSON 文件路径（可选）'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='输出 CSV 文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    try:
        # 跨框架聚合
        result = aggregate_cross_framework(
            model_name=args.model_name,
            data_dir=args.data_dir,
            frameworks=args.frameworks,
        )
        
        # 打印汇总
        print_summary(result)
        
        # 保存到 JSON 文件
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 JSON 结果已保存到: {output_path}")
        
        # 保存到 CSV 文件
        if args.output_csv:
            save_csv(result, args.output_csv)
            print(f"💾 CSV 结果已保存到: {args.output_csv}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

