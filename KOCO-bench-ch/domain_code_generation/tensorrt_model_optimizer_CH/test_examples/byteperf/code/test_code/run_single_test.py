#!/usr/bin/env python3
# Copyright 2024 ByteMLPerf. All rights reserved.

"""
运行单个测试文件的脚本
"""

import unittest
import sys
import os
import time

def run_single_test(test_file):
    """运行单个测试文件"""
    print(f"运行测试: {test_file}")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 导入测试模块
        test_module = __import__(test_file[:-3])  # 移除.py扩展名
        
        # 创建测试套件
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_module)
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "success": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "duration": duration,
            "result": result
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ 运行测试时出错: {e}")
        return {
            "success": False,
            "tests_run": 0,
            "failures": 0,
            "errors": 1,
            "duration": duration,
            "error": str(e)
        }

def main():
    """主函数"""
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("ByteMLPerf 单个测试运行器")
    print("=" * 50)
    print(f"测试目录: {script_dir}")
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python run_single_test.py <test_file>")
        print("可用的测试文件:")
        for file in os.listdir("."):
            if file.startswith("test_") and file.endswith(".py"):
                print(f"  - {file}")
        sys.exit(1)
    
    test_file = sys.argv[1]
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        sys.exit(1)
    
    # 运行测试
    result = run_single_test(test_file)
    
    # 打印结果
    print(f"\n{'='*50}")
    print("测试结果")
    print(f"{'='*50}")
    
    status = "✅ 通过" if result["success"] else "❌ 失败"
    duration = result["duration"]
    tests_run = result["tests_run"]
    failures = result["failures"]
    errors = result["errors"]
    
    print(f"状态: {status}")
    print(f"测试数: {tests_run}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    print(f"耗时: {duration:.2f}秒")
    
    if not result["success"] and "error" in result:
        print(f"错误: {result['error']}")
    
    # 返回适当的退出码
    if result["success"]:
        print(f"\n🎉 测试通过!")
        sys.exit(0)
    else:
        print(f"\n❌ 测试失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()