import unittest
import torch
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import accuracy_reward


class TestAccuracyReward(unittest.TestCase):
    """测试accuracy_reward函数 - CANOE文档第1节描述的答案正确性验证"""
    
    def setUp(self):
        # 清除DEBUG_MODE环境变量
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_accuracy_reward_basic_string_matching(self):
        """测试基本的字符串匹配功能"""
        # 创建测试数据
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer><answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        # 调用真实的accuracy_reward函数
        rewards = accuracy_reward(completions, solution)
        
        # 验证输出
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 答案正确
        self.assertEqual(rewards[1], 1.0)  # 答案正确
    
    def test_accuracy_reward_incorrect_answer(self):
        """测试错误答案的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>思考过程</think><long_answer>详细答案</long_answer><answer>99</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 第一个答案正确
        self.assertEqual(rewards[1], 0.0)  # 第二个答案错误
    
    def test_accuracy_reward_without_tags(self):
        """测试没有answer标签的情况"""
        completions = [
            [{"role": "assistant", "content": "42"}],
            [{"role": "assistant", "content": "100"}],
        ]
        solution = ["42", "99"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 答案正确
        self.assertEqual(rewards[1], 0.0)  # 答案错误
    
    def test_accuracy_reward_with_whitespace(self):
        """测试带有空白字符的答案"""
        completions = [
            [{"role": "assistant", "content": "<answer>  42  </answer>"}],
            [{"role": "assistant", "content": "<answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>  100  </answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # 由于strip()处理，空白字符应该被忽略
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_accuracy_reward_mixed_format(self):
        """测试混合格式（有标签和无标签）"""
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            [{"role": "assistant", "content": "100"}],
        ]
        solution = ["42", "<answer>100</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_accuracy_reward_multiline_content(self):
        """测试多行内容"""
        completions = [
            [{"role": "assistant", "content": "<think>思考\n过程</think><long_answer>详细\n答案</long_answer><answer>42</answer>"}],
        ]
        solution = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_accuracy_reward_empty_answer(self):
        """测试空答案的情况"""
        completions = [
            [{"role": "assistant", "content": "<answer></answer>"}],
            [{"role": "assistant", "content": ""}],
        ]
        solution = ["<answer></answer>", ""]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 空答案匹配
        self.assertEqual(rewards[1], 1.0)  # 空字符串匹配
    
    def test_accuracy_reward_case_sensitive(self):
        """测试大小写敏感性"""
        completions = [
            [{"role": "assistant", "content": "<answer>ABC</answer>"}],
            [{"role": "assistant", "content": "<answer>abc</answer>"}],
        ]
        solution = ["<answer>abc</answer>", "<answer>abc</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # 字符串匹配是大小写敏感的
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)  # 大小写不匹配
        self.assertEqual(rewards[1], 1.0)  # 大小写匹配
    
    def test_accuracy_reward_numeric_answers(self):
        """测试数值答案"""
        completions = [
            [{"role": "assistant", "content": "<answer>3.14</answer>"}],
            [{"role": "assistant", "content": "<answer>-5</answer>"}],
            [{"role": "assistant", "content": "<answer>0</answer>"}],
        ]
        solution = ["<answer>3.14</answer>", "<answer>-5</answer>", "<answer>1</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
        self.assertEqual(rewards[2], 0.0)
    
    def test_accuracy_reward_special_characters(self):
        """测试特殊字符"""
        completions = [
            [{"role": "assistant", "content": "<answer>x^2 + y^2 = r^2</answer>"}],
            [{"role": "assistant", "content": "<answer>α + β = γ</answer>"}],
        ]
        solution = ["<answer>x^2 + y^2 = r^2</answer>", "<answer>α + β = γ</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    @patch('open_r1.grpo.parse')
    @patch('open_r1.grpo.verify')
    def test_accuracy_reward_symbolic_verification_success(self, mock_verify, mock_parse):
        """测试符号验证成功的情况"""
        # Mock符号验证成功
        mock_parse.return_value = MagicMock()
        mock_verify.return_value = 1.0  # 验证成功
        
        completions = [
            [{"role": "assistant", "content": "<answer>2*x + 3</answer>"}],
        ]
        solution = ["<answer>3 + 2*x</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # 符号验证成功，应该返回1.0
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
        
        # 验证parse和verify被调用
        self.assertEqual(mock_parse.call_count, 2)  # 解析生成答案和正确答案
        mock_verify.assert_called_once()
    
    @patch('open_r1.grpo.parse')
    @patch('open_r1.grpo.verify')
    def test_accuracy_reward_symbolic_verification_failure_fallback_to_string(self, mock_verify, mock_parse):
        """测试符号验证失败后回退到字符串匹配"""
        # Mock符号验证失败
        mock_parse.side_effect = Exception("Parse error")
        
        completions = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
        ]
        solution = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        # 符号验证失败，但字符串匹配成功
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_accuracy_reward_batch_processing(self):
        """测试批量处理"""
        # 创建一个较大的批次
        batch_size = 10
        completions = [
            [{"role": "assistant", "content": f"<answer>{i}</answer>"}]
            for i in range(batch_size)
        ]
        solution = [f"<answer>{i}</answer>" for i in range(batch_size)]
        
        rewards = accuracy_reward(completions, solution)
        
        # 所有答案都应该正确
        self.assertEqual(len(rewards), batch_size)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_accuracy_reward_debug_mode_logging(self):
        """测试DEBUG_MODE下的日志记录"""
        # 设置DEBUG_MODE环境变量
        os.environ["DEBUG_MODE"] = "true"
        
        # 创建临时日志文件路径
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = f.name
        
        os.environ["LOG_PATH"] = log_path
        
        try:
            completions = [
                [{"role": "assistant", "content": "<answer>42</answer>"}],
            ]
            solution = ["<answer>42</answer>"]
            
            rewards = accuracy_reward(completions, solution)
            
            # 验证日志文件被创建并包含内容
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, 'r') as f:
                log_content = f.read()
                self.assertIn("Accuracy reward", log_content)
                self.assertIn("Content:", log_content)
                self.assertIn("Solution:", log_content)
        finally:
            # 清理
            if os.path.exists(log_path):
                os.remove(log_path)
            del os.environ["DEBUG_MODE"]
            del os.environ["LOG_PATH"]
    
    def test_accuracy_reward_edge_cases(self):
        """测试边缘情况"""
        # 测试样例1: 只有一个样本
        completions_single = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
        ]
        solution_single = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions_single, solution_single)
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
        
        # 测试样例2: 答案中包含answer标签但格式不完整
        completions_incomplete = [
            [{"role": "assistant", "content": "<answer>42"}],  # 缺少结束标签
        ]
        solution_incomplete = ["<answer>42</answer>"]
        
        rewards = accuracy_reward(completions_incomplete, solution_incomplete)
        self.assertEqual(len(rewards), 1)
        # 由于没有完整的标签，会使用整个字符串进行匹配
        self.assertEqual(rewards[0], 0.0)


class TestAccuracyRewardIntegration(unittest.TestCase):
    """集成测试：测试accuracy_reward与实际使用场景的集成"""
    
    def test_realistic_math_problem(self):
        """测试真实的数学问题场景"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>这是一个简单的加法问题。2 + 2 = 4</think>"
              "<long_answer>根据基本的算术运算，2加2等于4。</long_answer>"
              "<answer>4</answer>"}],
        ]
        solution = ["<answer>4</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_multiple_samples_mixed_results(self):
        """测试多个样本的混合结果"""
        completions = [
            [{"role": "assistant", "content": "<think>...</think><long_answer>...</long_answer><answer>正确答案</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>...</long_answer><answer>错误答案</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>...</long_answer><answer>另一个正确答案</answer>"}],
        ]
        solution = ["<answer>正确答案</answer>", "<answer>正确答案</answer>", "<answer>另一个正确答案</answer>"]
        
        rewards = accuracy_reward(completions, solution)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # 正确
        self.assertEqual(rewards[1], 0.0)  # 错误
        self.assertEqual(rewards[2], 1.0)  # 正确


if __name__ == "__main__":
    unittest.main()
