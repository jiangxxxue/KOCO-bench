import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import default_accuracy_reward


class TestDefaultAccuracyReward(unittest.TestCase):
    """测试default_accuracy_reward函数 - VLM-R1文档描述的默认准确性奖励函数"""
    
    def test_exact_string_match(self):
        """测试精确字符串匹配"""
        content = "<think>思考过程</think><answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_exact_string_match_without_tags(self):
        """测试没有标签的精确匹配"""
        content = "42"
        sol = "42"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_incorrect_answer(self):
        """测试错误答案"""
        content = "<answer>42</answer>"
        sol = "<answer>100</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 0.0)
    
    def test_numeric_answer_matching(self):
        """测试数值答案匹配"""
        content = "<answer>3.14</answer>"
        sol = "<answer>3.14</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_numeric_answer_mismatch(self):
        """测试数值答案不匹配"""
        content = "<answer>3.14</answer>"
        sol = "<answer>2.71</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # 数值不匹配，但会使用模糊匹配，相似度较低
        self.assertLess(reward, 0.5)
    
    def test_multiple_choice_correct(self):
        """测试多选题正确答案"""
        content = "<answer>A</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_multiple_choice_incorrect(self):
        """测试多选题错误答案"""
        content = "<answer>B</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 0.0)
    
    def test_multiple_choice_with_text(self):
        """测试带文本的多选题"""
        content = "<answer>The answer is A</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # 应该能提取出选项A并匹配
        self.assertEqual(reward, 1.0)
    
    def test_text_answer_fuzzy_match(self):
        """测试文本答案的模糊匹配"""
        content = "<answer>The capital of France is Paris</answer>"
        sol = "<answer>Paris</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # 模糊匹配应该给出一定的相似度分数
        self.assertGreater(reward, 0.0)
        self.assertLessEqual(reward, 1.0)
    
    def test_text_answer_exact_fuzzy_match(self):
        """测试完全相同的文本答案"""
        content = "<answer>Paris</answer>"
        sol = "<answer>Paris</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_whitespace_handling(self):
        """测试空白字符处理"""
        content = "<answer>  42  </answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # strip()应该移除前后空白
        self.assertEqual(reward, 1.0)
    
    def test_multiple_answer_tags(self):
        """测试多个answer标签（使用最后一个）"""
        content = "<answer>wrong</answer><think>more thinking</think><answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # 应该使用最后一个answer标签
        self.assertEqual(reward, 1.0)
    
    def test_case_sensitivity(self):
        """测试大小写敏感性"""
        content = "<answer>ABC</answer>"
        sol = "<answer>abc</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # clean_text会转换为小写，所以应该匹配
        self.assertEqual(reward, 1.0)
    
    @patch('open_r1.grpo_jsonl.parse')
    @patch('open_r1.grpo_jsonl.verify')
    def test_symbolic_verification_success(self, mock_verify, mock_parse):
        """测试符号验证成功"""
        mock_parse.return_value = MagicMock()
        mock_verify.return_value = 1.0
        
        content = "<answer>2*x + 3</answer>"
        sol = "<answer>3 + 2*x</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
        self.assertEqual(mock_parse.call_count, 2)
        mock_verify.assert_called_once()
    
    @patch('open_r1.grpo_jsonl.parse')
    def test_symbolic_verification_failure_fallback(self, mock_parse):
        """测试符号验证失败后回退到字符串匹配"""
        mock_parse.side_effect = Exception("Parse error")
        
        content = "<answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # 符号验证失败，但字符串匹配成功
        self.assertEqual(reward, 1.0)
    
    def test_empty_answer(self):
        """测试空答案"""
        content = "<answer></answer>"
        sol = "<answer></answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_special_characters(self):
        """测试特殊字符"""
        content = "<answer>x^2 + y^2 = r^2</answer>"
        sol = "<answer>x^2 + y^2 = r^2</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_numeric_with_units(self):
        """测试带单位的数值"""
        content = "<answer>42 meters</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # 数值匹配应该能提取出42
        self.assertGreater(reward, 0.0)
    
    def test_multiline_content(self):
        """测试多行内容"""
        content = "<think>思考\n过程</think><answer>42</answer>"
        sol = "<answer>42</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)


class TestDefaultAccuracyRewardIntegration(unittest.TestCase):
    """集成测试：测试default_accuracy_reward在实际场景中的表现"""
    
    def test_math_problem_scenario(self):
        """测试数学问题场景"""
        content = "<think>2 + 2 = 4</think><answer>4</answer>"
        sol = "<answer>4</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_multiple_choice_scenario(self):
        """测试多选题场景"""
        # 由于 extract_choice 函数未定义，会回退到模糊匹配
        content = "<think>分析选项...</think><answer>A</answer>"
        sol = "<answer>A</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        # 精确匹配应该返回1.0
        self.assertEqual(reward, 1.0)
    
    def test_text_answer_scenario(self):
        """测试文本答案场景"""
        content = "<think>思考...</think><answer>巴黎</answer>"
        sol = "<answer>巴黎</answer>"
        
        reward = default_accuracy_reward(content, sol)
        
        self.assertEqual(reward, 1.0)
    
    def test_mixed_verification_methods(self):
        """测试混合验证方法"""
        # 测试数值答案
        reward1 = default_accuracy_reward("<answer>3.14</answer>", "<answer>3.14</answer>")
        self.assertEqual(reward1, 1.0)
        
        # 测试多选题
        reward2 = default_accuracy_reward("<answer>B</answer>", "<answer>B</answer>")
        self.assertEqual(reward2, 1.0)
        
        # 测试文本答案
        reward3 = default_accuracy_reward("<answer>Paris</answer>", "<answer>Paris</answer>")
        self.assertEqual(reward3, 1.0)


if __name__ == "__main__":
    unittest.main()
