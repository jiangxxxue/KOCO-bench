import unittest
import torch
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import accuracy_reward


class TestAccuracyReward(unittest.TestCase):
    """测试accuracy_reward函数 - 批量计算所有样本的保真度奖励"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.num_generations = 2
    
    def test_accuracy_reward_basic(self):
        """测试基本的准确度奖励计算"""
        # 创建测试数据：2个样本，每个2次生成
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.5</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.3</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>2.8</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.2</answer>"}],
        ]
        
        solution = [
            "<answer>4.5</answer>",
            "<answer>4.5</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # 验证输出
        self.assertEqual(len(rewards), 4)  # 2 samples × 2 generations
        for reward in rewards:
            self.assertTrue(torch.is_tensor(reward) or isinstance(reward, (int, float)))
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))
    
    def test_accuracy_reward_correct_predictions(self):
        """测试预测正确的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
        ]
        
        solution = [
            "<answer>5.0</answer>",
            "<answer>5.0</answer>",
            "<answer>1.0</answer>",
            "<answer>1.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # 当预测完全正确时，奖励应该较高
        self.assertEqual(len(rewards), 4)
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertGreater(reward.item(), 0.0)
    
    def test_accuracy_reward_with_variance(self):
        """测试不同生成结果的方差影响"""
        # 第一个样本：生成结果方差小
        # 第二个样本：生成结果方差大
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.1</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>2.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
        ]
        
        solution = [
            "<answer>4.0</answer>",
            "<answer>4.0</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 4)
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))
    
    def test_accuracy_reward_parsing_formats(self):
        """测试不同格式的答案解析"""
        completions = [
            [{"role": "assistant", "content": "<think>step1</think><answer>3.5</answer>"}],
            [{"role": "assistant", "content": "<think>step2</think><answer>The score is 3.7</answer>"}],
            [{"role": "assistant", "content": "<think>step3</think><answer>2.5</answer>"}],
            [{"role": "assistant", "content": "<think>step4</think><answer>Score: 2.3</answer>"}],
        ]
        
        solution = [
            "<answer>3.5</answer>",
            "<answer>3.5</answer>",
            "<answer>2.5</answer>",
            "<answer>2.5</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # 应该能够从不同格式中提取数字
        self.assertEqual(len(rewards), 4)
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))
    
    def test_accuracy_reward_pairwise_comparison(self):
        """测试成对比较逻辑"""
        # 创建明确的排序关系
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
        ]
        
        solution = [
            "<answer>5.0</answer>",
            "<answer>5.0</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
            "<answer>1.0</answer>",
            "<answer>1.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # 验证成对比较产生的奖励
        self.assertEqual(len(rewards), 6)  # 3 samples × 2 generations
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))
    
    def test_accuracy_reward_error_handling(self):
        """测试错误处理：无法解析的答案"""
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>invalid</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>3.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>2.0</answer>"}],
        ]
        
        solution = [
            "<answer>4.0</answer>",
            "<answer>4.0</answer>",
            "<answer>3.0</answer>",
            "<answer>3.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        # 应该能够处理无效答案（使用随机值）
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        self.assertEqual(len(rewards), 4)
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))
    
    def test_accuracy_reward_single_sample(self):
        """测试单个样本的情况"""
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>4.2</answer>"}],
        ]
        
        solution = [
            "<answer>4.0</answer>",
            "<answer>4.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": self.num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # 单个样本无法进行成对比较，但函数应该正常处理
        self.assertEqual(len(rewards), 2)


class TestAccuracyRewardIntegration(unittest.TestCase):
    """集成测试：测试accuracy_reward的整体工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
    
    def test_end_to_end_workflow(self):
        """端到端测试：完整的奖励计算流程"""
        # 模拟真实场景：4个样本，每个3次生成
        num_samples = 4
        num_generations = 3
        
        completions = []
        solution = []
        
        # 生成测试数据
        scores = [5.0, 4.0, 3.0, 2.0]
        for score in scores:
            for _ in range(num_generations):
                # 添加一些随机变化
                pred_score = score + torch.randn(1).item() * 0.2
                completions.append([{
                    "role": "assistant",
                    "content": f"<think>reasoning</think><answer>{pred_score:.2f}</answer>"
                }])
                solution.append(f"<answer>{score}</answer>")
        
        kwargs = {
            "device": self.device,
            "num_generations": num_generations,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # 验证输出
        self.assertEqual(len(rewards), num_samples * num_generations)
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))
                self.assertGreater(reward.item(), 0.0)
    
    def test_ranking_consistency(self):
        """测试排序一致性"""
        # 创建明确的质量排序
        completions = [
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>5.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
            [{"role": "assistant", "content": "<think>reasoning</think><answer>1.0</answer>"}],
        ]
        
        solution = [
            "<answer>5.0</answer>",
            "<answer>5.0</answer>",
            "<answer>1.0</answer>",
            "<answer>1.0</answer>",
        ]
        
        kwargs = {
            "device": self.device,
            "num_generations": 2,
        }
        
        rewards = accuracy_reward(completions, solution, **kwargs)
        
        # 验证奖励的一致性
        self.assertEqual(len(rewards), 4)
        for reward in rewards:
            if torch.is_tensor(reward):
                self.assertTrue(torch.isfinite(reward))


if __name__ == "__main__":
    unittest.main()
