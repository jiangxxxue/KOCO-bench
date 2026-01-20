#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_cross_framework.py - Cross-framework aggregation of evaluation metrics

Aggregate all test instances across multiple frameworks, calculate comprehensive pass@1 and avg_pass_ratio
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import existing modules
from aggregate_metrics import aggregate_metrics, discover_test_examples


def discover_frameworks(data_dir: str) -> List[str]:
    """
    Automatically discover all frameworks in the data directory
    
    Args:
        data_dir: Data directory path
    
    Returns:
        List of framework names
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
    Aggregate evaluation metrics across frameworks
    
    Args:
        model_name: Model name
        data_dir: Data directory path (e.g., scripts/data)
        frameworks: List of framework names (optional, auto-discover if not specified)
    
    Returns:
        Aggregated metrics dictionary
    """
    data_path = Path(data_dir)
    
    # Auto-discover frameworks if not specified
    if frameworks is None or len(frameworks) == 0:
        frameworks = discover_frameworks(data_dir)
        if not frameworks:
            raise ValueError(f"No framework directories found in {data_dir}")
        print(f"📋 Auto-discovered {len(frameworks)} frameworks: {', '.join(frameworks)}")
    
    # Collect results for each framework
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
            print(f"⚠️  Warning: Model {model_name} not found under framework {framework}: {model_dir}")
            missing_frameworks.append(framework)
            continue
        
        # Check if metrics files exist
        test_examples = discover_test_examples(str(model_dir))
        if not test_examples:
            print(f"⚠️  Warning: No test instances found for framework {framework} model {model_name}")
            missing_frameworks.append(framework)
            continue
        
        try:
            # Aggregate all instances under this framework
            result = aggregate_metrics(
                model_dir=str(model_dir),
                test_examples=test_examples,
                framework=framework,
            )
            
            agg = result['aggregate_metrics']
            num_funcs = agg['total_functions']
            
            # Accumulate statistics
            total_functions += num_funcs
            total_tests += agg['total_tests']
            total_passed += agg['total_passed']
            
            # Weighted accumulation
            weighted_pass_at_1 += agg['pass@1'] * num_funcs
            weighted_avg_pass_ratio += agg['avg_pass_ratio'] * num_funcs
            
            # Save framework result
            framework_result = {
                'framework': framework,
                'model_dir': str(model_dir),
                'test_examples': test_examples,
                'metrics': agg,
                'individual_metrics': result['individual_metrics']
            }
            all_framework_results.append(framework_result)
            valid_frameworks.append(framework)
            
            print(f"\n✅ {framework}: {len(test_examples)} instances, {num_funcs} functions")
            print(f"   pass@1: {agg['pass@1']:.4f}, avg_pass_ratio: {agg['avg_pass_ratio']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to process framework {framework}: {e}")
            missing_frameworks.append(framework)
            continue
    
    if not all_framework_results:
        raise ValueError("No frameworks were successfully processed")
    
    # Calculate comprehensive metrics
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
    """Print summary results"""
    print("\n" + "=" * 80)
    print("📊 Cross-Framework Comprehensive Metrics Summary")
    print("=" * 80)
    print(f"Model name: {result['model_name']}")
    print(f"Data directory: {result['data_dir']}")
    print(f"Framework list: {', '.join(result['frameworks'])}")
    
    if result['missing_frameworks']:
        print(f"⚠️  Missing frameworks: {', '.join(result['missing_frameworks'])}")
    
    print("\n" + "-" * 80)
    print("Comprehensive Results:")
    print("-" * 80)
    
    agg = result['aggregate_metrics']
    print(f"Valid frameworks:   {agg['total_frameworks']}")
    print(f"Total functions:     {agg['total_functions']}")
    print(f"Total tests:        {agg['total_tests']}")
    print(f"Passed functions:   {agg['total_passed']}")
    print(f"pass@1:             {agg['pass@1']:.4f} ({agg['pass@1']*100:.2f}%)")
    print(f"avg_pass_ratio:     {agg['avg_pass_ratio']:.4f}")
    
    print("\n" + "-" * 80)
    print("Framework Details:")
    print("-" * 80)
    
    # Table header
    header = f"{'Framework':<25} {'Instances':>8} {'Functions':>10} {'Passed':>10} {'pass@1':>12} {'avg_pass_ratio':>15}"
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
    """Save results as CSV file"""
    import csv
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write summary information
        writer.writerow(['# Cross-Framework Aggregation Results'])
        writer.writerow(['Model Name', result['model_name']])
        agg = result['aggregate_metrics']
        writer.writerow(['Total Frameworks', agg['total_frameworks']])
        writer.writerow(['Total Functions', agg['total_functions']])
        writer.writerow(['Total Passed', agg['total_passed']])
        writer.writerow(['Comprehensive pass@1', f"{agg['pass@1']:.4f}"])
        writer.writerow(['Comprehensive avg_pass_ratio', f"{agg['avg_pass_ratio']:.4f}"])
        writer.writerow([])
        
        # Write framework details header
        writer.writerow(['Framework', 'Instances', 'Functions', 'Passed', 'pass@1', 'avg_pass_ratio'])
        
        # Write framework data
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
        description="Aggregate evaluation metrics across frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Auto-discover all frameworks (recommended)
  python aggregate_cross_framework.py \\
    --model_name your_model \\
    --data_dir scripts/data

  # Specify specific frameworks
  python aggregate_cross_framework.py \\
    --model_name your_model \\
    --data_dir scripts/data \\
    --frameworks framework1 framework2 framework3

  # Save results
  python aggregate_cross_framework.py \\
    --model_name your_model \\
    --data_dir scripts/data \\
    --output cross_framework_result.json \\
    --output_csv cross_framework_result.csv

Output description:
  - pass@1: Number of passed functions across all frameworks and instances / Total functions (weighted average)
  - avg_pass_ratio: avg_pass_ratio of all frameworks and instances weighted by function count
        """
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Model name (e.g., your_model)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Data directory path (e.g., scripts/data)'
    )
    parser.add_argument(
        '--frameworks',
        type=str,
        nargs='+',
        default=None,
        help='List of framework names (space-separated). If not specified, auto-discover all frameworks in directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (optional)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default=None,
        help='Output CSV file path (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        # Cross-framework aggregation
        result = aggregate_cross_framework(
            model_name=args.model_name,
            data_dir=args.data_dir,
            frameworks=args.frameworks,
        )
        
        # Print summary
        print_summary(result)
        
        # Save to JSON file
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 JSON results saved to: {output_path}")
        
        # Save to CSV file
        if args.output_csv:
            save_csv(result, args.output_csv)
            print(f"💾 CSV results saved to: {args.output_csv}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

