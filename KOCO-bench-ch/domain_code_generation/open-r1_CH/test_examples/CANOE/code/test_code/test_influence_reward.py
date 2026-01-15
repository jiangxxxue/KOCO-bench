import unittest
import sys
import os
from unittest.mock import patch

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train', 'src'))

# Import the actual function we want to test
from open_r1.grpo import influence_reward


class TestInfluenceReward(unittest.TestCase):
    """测试influence_reward函数 - CANOE文档第2节描述的长答案影响力评估"""
    
    def setUp(self):
        # 清除DEBUG_MODE环境变量
        if "DEBUG_MODE" in os.environ:
            del os.environ["DEBUG_MODE"]
    
    def test_influence_reward_basic_correct(self):
        """测试基本的正确影响力验证"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            [{"role": "assistant", "content": "<answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        # 调用真实的influence_reward函数
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # 验证输出
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 长答案生成的答案正确
        self.assertEqual(rewards[1], 1.0)  # 长答案生成的答案正确
    
    def test_influence_reward_incorrect_answer(self):
        """测试错误答案的情况"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            [{"role": "assistant", "content": "<answer>99</answer>"}],  # 错误答案
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 正确
        self.assertEqual(rewards[1], 0.0)  # 错误
    
    def test_influence_reward_none_completion(self):
        """测试completions_long_answer为None的情况"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            None,  # 没有长答案生成
        ]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 正确
        self.assertEqual(rewards[1], 0.0)  # None的情况返回0.0
    
    def test_influence_reward_all_none(self):
        """测试所有completions_long_answer都为None的情况"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [None, None]
        solution = ["<answer>42</answer>", "<answer>100</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)
        self.assertEqual(rewards[1], 0.0)
    
    def test_influence_reward_without_tags(self):
        """测试没有answer标签的情况"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "42"}],
        ]
        solution = ["42"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_influence_reward_with_whitespace(self):
        """测试带有空白字符的答案"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>  42  </answer>"}],
            [{"role": "assistant", "content": "<answer>100</answer>"}],
        ]
        solution = ["<answer>42</answer>", "<answer>  100  </answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # 由于strip()处理，空白字符应该被忽略
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
    
    def test_influence_reward_mixed_none_and_valid(self):
        """测试混合None和有效值的情况"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
            None,
            [{"role": "assistant", "content": "<answer>100</answer>"}],
            None,
        ]
        solution = ["<answer>42</answer>", "<answer>50</answer>", "<answer>100</answer>", "<answer>200</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)  # 正确
        self.assertEqual(rewards[1], 0.0)  # None
        self.assertEqual(rewards[2], 1.0)  # 正确
        self.assertEqual(rewards[3], 0.0)  # None
    
    def test_influence_reward_multiline_content(self):
        """测试多行内容"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<think>思考\n过程</think><long_answer>详细\n答案</long_answer><answer>42</answer>"}],
        ]
        solution = ["<answer>42</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)
    
    def test_influence_reward_empty_answer(self):
        """测试空答案的情况"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer></answer>"}],
            [{"role": "assistant", "content": ""}],
        ]
        solution = ["<answer></answer>", ""]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)  # 空答案匹配
        self.assertEqual(rewards[1], 1.0)  # 空字符串匹配
    
    def test_influence_reward_case_sensitive(self):
        """测试大小写敏感性"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>ABC</answer>"}],
            [{"role": "assistant", "content": "<answer>abc</answer>"}],
        ]
        solution = ["<answer>abc</answer>", "<answer>abc</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # 字符串匹配是大小写敏感的
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 0.0)  # 大小写不匹配
        self.assertEqual(rewards[1], 1.0)  # 大小写匹配
    
    def test_influence_reward_batch_processing(self):
        """测试批量处理"""
        batch_size = 10
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}]
            for _ in range(batch_size)
        ]
        completions_long_answer = [
            [{"role": "assistant", "content": f"<answer>{i}</answer>"}]
            for i in range(batch_size)
        ]
        solution = [f"<answer>{i}</answer>" for i in range(batch_size)]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        # 所有答案都应该正确
        self.assertEqual(len(rewards), batch_size)
        self.assertTrue(all(r == 1.0 for r in rewards))
    
    def test_influence_reward_debug_mode_logging(self):
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
                [{"role": "assistant", "content": "原始生成内容"}],
            ]
            completions_long_answer = [
                [{"role": "assistant", "content": "<answer>42</answer>"}],
            ]
            solution = ["<answer>42</answer>"]
            
            rewards = influence_reward(completions, solution, completions_long_answer)
            
            # 验证日志文件被创建并包含内容
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, 'r') as f:
                log_content = f.read()
                self.assertIn("influence_Accuracy reward", log_content)
                self.assertIn("Content:", log_content)
                self.assertIn("Solution:", log_content)
        finally:
            # 清理
            if os.path.exists(log_path):
                os.remove(log_path)
            del os.environ["DEBUG_MODE"]
            del os.environ["LOG_PATH"]
    
    def test_influence_reward_exception_handling(self):
        """测试异常处理"""
        completions = [
            [{"role": "assistant", "content": "原始生成内容"}],
        ]
        # 创建一个可能导致异常的情况
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],
        ]
        # 使用一个可能导致正则表达式异常的solution
        solution = ["<answer>42</answer>"]
        
        # 应该不会抛出异常
        try:
            rewards = influence_reward(completions, solution, completions_long_answer)
            self.assertEqual(len(rewards), 1)
        except Exception as e:
            self.fail(f"influence_reward raised an exception: {e}")


class TestInfluenceRewardIntegration(unittest.TestCase):
    """集成测试：测试influence_reward与实际使用场景的集成"""
    
    def test_realistic_dual_grpo_scenario(self):
        """测试真实的Dual-GRPO场景"""
        # 模拟第一阶段生成
        completions = [
            [{"role": "assistant", "content": 
              "<think>思考过程</think>"
              "<long_answer>这是一个详细的长答案，用于替换context。</long_answer>"
              "<answer>42</answer>"}],
        ]
        
        # 模拟第二阶段生成（使用长答案替换context后重新生成）
        completions_long_answer = [
            [{"role": "assistant", "content": 
              "<think>基于长答案的思考</think>"
              "<long_answer>基于长答案的详细答案</long_answer>"
              "<answer>42</answer>"}],  # 答案仍然正确
        ]
        
        solution = ["<answer>42</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)  # 长答案有正面影响
    
    def test_negative_influence_scenario(self):
        """测试负面影响场景（长答案导致错误答案）"""
        completions = [
            [{"role": "assistant", "content": 
              "<think>思考过程</think>"
              "<long_answer>这是一个可能误导的长答案。</long_answer>"
              "<answer>42</answer>"}],
        ]
        
        # 长答案导致错误的最终答案
        completions_long_answer = [
            [{"role": "assistant", "content": 
              "<think>基于误导性长答案的思考</think>"
              "<long_answer>错误的详细答案</long_answer>"
              "<answer>99</answer>"}],  # 答案变错了
        ]
        
        solution = ["<answer>42</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)  # 长答案有负面影响
    
    def test_mixed_influence_batch(self):
        """测试混合影响的批次"""
        completions = [
            [{"role": "assistant", "content": "<think>...</think><long_answer>好的长答案</long_answer><answer>42</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>误导性长答案</long_answer><answer>100</answer>"}],
            [{"role": "assistant", "content": "<think>...</think><long_answer>无效长答案</long_answer><answer>200</answer>"}],
        ]
        
        completions_long_answer = [
            [{"role": "assistant", "content": "<answer>42</answer>"}],   # 正确
            [{"role": "assistant", "content": "<answer>99</answer>"}],   # 错误
            None,  # 没有生成
        ]
        
        solution = ["<answer>42</answer>", "<answer>100</answer>", "<answer>200</answer>"]
        
        rewards = influence_reward(completions, solution, completions_long_answer)
        
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # 正面影响
        self.assertEqual(rewards[1], 0.0)  # 负面影响
        self.assertEqual(rewards[2], 0.0)  # 无影响


if __name__ == "__main__":
    unittest.main()
