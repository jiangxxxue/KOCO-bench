#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_metrics.py - Aggregate evaluation metrics from multiple test instances

Calculate comprehensive pass@1 and avg_pass_ratio across multiple test instances
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_metrics(file_path: str) -> Dict[str, Any]:
    """Load a single metrics file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_test_examples(model_dir: str) -> List[str]:
    """
    Automatically discover all test instances in the directory
    
    Extract test instance names by scanning *result.metrics.json files
    
    Args:
        model_dir: Model output directory path
    
    Returns:
        List of test instance names
    """
    import re
    
    model_path = Path(model_dir)
    examples = []
    
    # Find all metrics files
    pattern = "*result.metrics.json"
    for metrics_file in model_path.glob(pattern):
        filename = metrics_file.name
        
        # Try to match algorithm_methods_data_{example}_result.metrics.json format
        match = re.match(r'algorithm_methods_data_(.+?)_result\.metrics\.json', filename)
        if match:
            example_name = match.group(1)
            examples.append(example_name)
            continue
        
        # Try to match other formats: {prefix}_{example}_result.metrics.json
        # Remove _result.metrics.json suffix, take the part after the last underscore
        base_name = filename.replace('_result.metrics.json', '')
        if '_' in base_name:
            # Take the part after the last underscore as example name
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2:
                example_name = parts[1]
                examples.append(example_name)
    
    # Remove duplicates and sort
    examples = sorted(set(examples))
    
    return examples


def aggregate_metrics(
    model_dir: str,
    test_examples: List[str] = None,
    framework: str = None,
) -> Dict[str, Any]:
    """
    Aggregate metrics from multiple test instances
    
    Args:
        model_dir: Model output directory path
        test_examples: List of test instance names (optional, auto-discover all instances in directory if not specified)
        framework: Framework name (optional, used for filename matching)
    
    Returns:
        Aggregated metrics dictionary
    """
    model_path = Path(model_dir)
    
    # Auto-discover test_examples if not specified
    if test_examples is None or len(test_examples) == 0:
        test_examples = discover_test_examples(model_dir)
        if not test_examples:
            raise ValueError(f"No metrics files found in directory {model_dir}")
        print(f"📋 Auto-discovered {len(test_examples)} test instances: {', '.join(test_examples)}")
    
    # Collect metrics from all instances
    all_metrics = []
    total_functions = 0
    total_tests = 0
    total_passed = 0
    weighted_pass_at_1 = 0.0  # Use weighted pass@1
    weighted_avg_pass_ratio = 0.0
    
    missing_files = []
    
    for example in test_examples:
        # Build metrics file path
        if framework:
            metrics_file = model_path / f"algorithm_methods_data_{example}_result.metrics.json"
        else:
            # Try to find matching file
            pattern = f"*{example}*result.metrics.json"
            matches = list(model_path.glob(pattern))
            if matches:
                metrics_file = matches[0]
            else:
                metrics_file = model_path / f"algorithm_methods_data_{example}_result.metrics.json"
        
        # Load metrics
        if not metrics_file.exists():
            print(f"⚠️  Warning: Metrics file not found for {example}: {metrics_file}")
            missing_files.append(example)
            continue
        
        try:
            metrics = load_metrics(str(metrics_file))
            all_metrics.append({
                'example': example,
                'metrics': metrics
            })
            
            # Accumulate statistics
            num_funcs = metrics.get('total_functions', 0)
            total_functions += num_funcs
            total_tests += metrics.get('total_tests', 0)
            total_passed += metrics.get('total_passed', 0)
            
            # Weighted average pass@1 (weighted by function count)
            pass_at_1 = metrics.get('pass_at_k', {}).get('pass@1', 0.0)
            weighted_pass_at_1 += pass_at_1 * num_funcs
            
            # Weighted average avg_pass_ratio (weighted by function count)
            avg_ratio = metrics.get('avg_pass_ratio', 0.0)
            weighted_avg_pass_ratio += avg_ratio * num_funcs
            
            print(f"✓ {example}: {num_funcs} functions, "
                  f"pass@1={pass_at_1:.4f}, "
                  f"avg_pass_ratio={avg_ratio:.4f}")
        
        except Exception as e:
            print(f"❌ Error: Cannot load metrics for {example}: {e}")
            missing_files.append(example)
    
    if not all_metrics:
        raise ValueError("No valid metrics files found")
    
    # Calculate comprehensive metrics
    # pass@1 uses weighted average (instead of simple total_passed / total_functions)
    aggregate_pass_at_1 = weighted_pass_at_1 / total_functions if total_functions > 0 else 0.0
    
    # avg_pass_ratio: weighted average
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
    """Print summary results"""
    print("\n" + "=" * 70)
    print("📊 Comprehensive Metrics Summary")
    print("=" * 70)
    print(f"Model directory: {result['model_dir']}")
    print(f"Test instances: {', '.join(result['test_examples'])}")
    
    if result['missing_examples']:
        print(f"⚠️  Missing instances: {', '.join(result['missing_examples'])}")
    
    print("\n" + "-" * 70)
    print("Comprehensive Results:")
    print("-" * 70)
    
    agg = result['aggregate_metrics']
    print(f"Total functions:     {agg['total_functions']}")
    print(f"Total tests:         {agg['total_tests']}")
    print(f"Passed functions:    {agg['total_passed']}")
    print(f"pass@1:              {agg['pass@1']:.4f} ({agg['pass@1']*100:.2f}%)")
    print(f"avg_pass_ratio:     {agg['avg_pass_ratio']:.4f}")
    
    print("\n" + "-" * 70)
    print("Instance Details:")
    print("-" * 70)
    for item in result['individual_metrics']:
        example = item['example']
        m = item['metrics']
        print(f"\n{example}:")
        print(f"  Functions: {m.get('total_functions', 0)}")
        print(f"  Passed:    {m.get('total_passed', 0)}")
        print(f"  pass@1:    {m.get('pass_at_k', {}).get('pass@1', 0.0):.4f}")
        print(f"  avg_pass_ratio: {m.get('avg_pass_ratio', 0.0):.4f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics from multiple test instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Auto-discover all test instances (recommended)
  python aggregate_metrics.py \\
    --model_dir scripts/data/your_framework/your_model

  # Specify specific test instances
  python aggregate_metrics.py \\
    --model_dir scripts/data/your_framework/your_model \\
    --test_examples example1 example2 example3

  # Specify framework name
  python aggregate_metrics.py \\
    --model_dir scripts/data/your_framework/your_model \\
    --test_examples example1 example2 example3 \\
    --framework your_framework

  # Save results to file
  python aggregate_metrics.py \\
    --model_dir scripts/data/your_framework/your_model \\
    --output aggregate_result.json

Output description:
  - pass@1: Number of passed functions across all instances / Total functions
  - avg_pass_ratio: avg_pass_ratio of all instances weighted by function count
        """
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help='Model output directory path'
    )
    parser.add_argument(
        '--test_examples',
        type=str,
        nargs='+',
        default=None,
        help='List of test instance names (space-separated). If not specified, auto-discover all instances in directory'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default=None,
        help='Framework name (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        # Aggregate metrics
        result = aggregate_metrics(
            model_dir=args.model_dir,
            test_examples=args.test_examples,
            framework=args.framework,
        )
        
        # Print summary
        print_summary(result)
        
        # Save to file
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_path}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

