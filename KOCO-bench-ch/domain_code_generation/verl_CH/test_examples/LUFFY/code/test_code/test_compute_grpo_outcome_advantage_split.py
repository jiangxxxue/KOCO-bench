import unittest
import torch
import sys
import os

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'luffy', 'verl'))

# Import the actual function we want to test
from verl.mix_src.mix_core_alg import compute_grpo_outcome_advantage_split


class TestComputeGrpoOutcomeAdvantageSplit(unittest.TestCase):
    """测试compute_grpo_outcome_advantage_split函数 - LUFFY文档第1节描述的分组GRPO优势估计"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_basic_functionality(self):
        """测试基本功能：仅使用on-policy样本计算组内基线"""
        batch_size, response_length = 4, 5
        
        # 创建token级奖励
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 3.0],  # 序列1: 总分3.0
            [0.0, 0.0, 0.0, 0.0, 5.0],  # 序列2: 总分5.0
            [0.0, 0.0, 0.0, 0.0, 2.0],  # 序列3: 总分2.0
            [0.0, 0.0, 0.0, 0.0, 4.0],  # 序列4: 总分4.0
        ])
        eos_mask = torch.ones(batch_size, response_length)
        
        # 前两个样本属于同一prompt (index=0)，后两个属于另一prompt (index=1)
        # 注意：index需要是整数列表，不是tensor，因为会作为字典的键
        index = [0, 0, 1, 1]
        
        # 所有样本都是on-policy
        on_policy_mask = torch.tensor([True, True, True, True])
        
        # 调用真实函数
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)
        # print(returns)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071]])
        expected_returns = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(returns, expected_returns, atol=1e-4, rtol=1e-4)
    
    def test_on_policy_mask_filtering(self):
        """测试on_policy_mask的过滤功能：仅用on-policy样本计算基线"""
        batch_size, response_length = 4, 5
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 2.0],  # on-policy
            [0.0, 0.0, 0.0, 0.0, 8.0],  # off-policy (不应影响基线)
            [0.0, 0.0, 0.0, 0.0, 4.0],  # on-policy
            [0.0, 0.0, 0.0, 0.0, 6.0],  # on-policy
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0]  # 所有样本同一prompt
        
        # 第二个样本是off-policy
        on_policy_mask = torch.tensor([True, False, True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)
        # print(returns)

        expected_advantages = torch.tensor([[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [ 2.0000,  2.0000,  2.0000,  2.0000,  2.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000]])
        expected_returns = torch.tensor([[-1.0000, -1.0000, -1.0000, -1.0000, -1.0000],
        [ 2.0000,  2.0000,  2.0000,  2.0000,  2.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(returns, expected_returns, atol=1e-4, rtol=1e-4)
        # 验证输出
    
    def test_single_on_policy_sample_per_group(self):
        """测试每组只有一个on-policy样本的情况：mean=0, std=1"""
        batch_size, response_length = 2, 4
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 3.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 1]  # 不同prompt
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # 每组只有一个样本时，mean=0, std=1
        # 所以归一化后: (score - 0) / 1 = score
        # 序列1: advantage = 5.0
        # 序列2: advantage = 3.0
        self.assertAlmostEqual(advantages[0, -1].item(), 5.0, places=4)
        self.assertAlmostEqual(advantages[1, -1].item(), 3.0, places=4)
    
    def test_use_std_parameter(self):
        """测试use_std参数：控制是否使用标准差归一化"""
        batch_size, response_length = 4, 3
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 6.0],
            [0.0, 0.0, 8.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0]  # 同一prompt
        on_policy_mask = torch.tensor([True, True, True, True])
        
        # 使用标准差归一化
        advantages_with_std, _ = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, use_std=True
        )
        
        # 不使用标准差归一化
        advantages_no_std, _ = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, use_std=False
        )
        
        # 两种方式应该产生不同结果
        self.assertFalse(torch.equal(advantages_with_std, advantages_no_std))
        
        # 验证use_std=False时只减去均值
        # mean([2, 4, 6, 8]) = 5.0
        # 不使用std时: score - mean，然后广播到所有token位置
        # 注意：优势值会被广播到整个序列
        expected_no_std = torch.tensor([
            [-3.0, -3.0, -3.0],  # 2 - 5 = -3，广播到所有位置
            [-1.0, -1.0, -1.0],  # 4 - 5 = -1，广播到所有位置
            [1.0, 1.0, 1.0],     # 6 - 5 = 1，广播到所有位置
            [3.0, 3.0, 3.0],     # 8 - 5 = 3，广播到所有位置
        ])
        torch.testing.assert_close(advantages_no_std, expected_no_std, atol=1e-4, rtol=1e-4)
    
    def test_epsilon_parameter(self):
        """测试epsilon参数：数值稳定性"""
        batch_size, response_length = 2, 3
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],  # 相同分数，std=0
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        # std=0的情况会被替换为1.0，所以不会除零
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, epsilon=1e-6
        )

        # print(advantages)
        # print(returns)
        expected_advantages = torch.tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
        expected_returns = torch.tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(returns, expected_returns, atol=1e-4, rtol=1e-4)
        # 验证没有NaN或Inf
        # self.assertTrue(torch.isfinite(advantages).all())
        # self.assertTrue(torch.isfinite(returns).all())
    
    def test_eos_mask_application(self):
        """测试eos_mask的应用：只在有效token位置计算优势"""
        batch_size, response_length = 2, 5
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 6.0],
        ])
        
        # 部分mask
        eos_mask = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 0.0],  # 最后一个位置无效
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.0000],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # 验证mask位置为0
        self.assertEqual(advantages[0, -1].item(), 0.0)
        
        # 验证有效位置有值
        self.assertNotEqual(advantages[1, -1].item(), 0.0)
    
    def test_multiple_groups(self):
        """测试多个prompt组的处理"""
        batch_size, response_length = 6, 4
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 2.0],  # group 0
            [0.0, 0.0, 0.0, 4.0],  # group 0
            [0.0, 0.0, 0.0, 3.0],  # group 1
            [0.0, 0.0, 0.0, 5.0],  # group 1
            [0.0, 0.0, 0.0, 1.0],  # group 2
            [0.0, 0.0, 0.0, 3.0],  # group 2
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 1, 1, 2, 2]
        on_policy_mask = torch.tensor([True, True, True, True, True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.7071,  0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # 验证输出形状
        self.assertEqual(advantages.shape, (batch_size, response_length))
        self.assertTrue(torch.isfinite(advantages).all())
        
        # 每组内的样本应该基于各自组的统计量归一化
        # Group 0: mean([2, 4]) = 3.0
        # Group 1: mean([3, 5]) = 4.0
        # Group 2: mean([1, 3]) = 2.0
    
    def test_token_level_broadcast(self):
        """测试序列级优势广播到token级"""
        batch_size, response_length = 2, 4
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 3.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[ 0.7071,  0.7071,  0.7071,  0.7071],
        [-0.7071, -0.7071, -0.7071, -0.7071]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # 验证每个序列的所有有效token位置有相同的优势值
        # （因为是outcome reward，广播到所有位置）
        for i in range(batch_size):
            valid_positions = eos_mask[i].bool()
            unique_values = torch.unique(advantages[i, valid_positions])
            # 所有有效位置应该有相同的值
            self.assertEqual(len(unique_values), 1)
    
    def test_edge_case_zero_rewards(self):
        """测试边缘情况：全零奖励"""
        batch_size, response_length = 2, 3
        
        token_level_rewards = torch.zeros(batch_size, response_length)
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0]
        on_policy_mask = torch.tensor([True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # 全零奖励应该产生全零优势
        self.assertTrue((advantages == 0).all())
        self.assertTrue((returns == 0).all())
    
    def test_edge_case_single_sample(self):
        """测试边缘情况：单个样本"""
        batch_size, response_length = 1, 4
        
        token_level_rewards = torch.tensor([[0.0, 0.0, 0.0, 7.0]])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0]
        on_policy_mask = torch.tensor([True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )

        # print(advantages)

        expected_advantages = torch.tensor([[7.0000, 7.0000, 7.0000, 7.0000]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        
        # 单样本组：mean=0, std=1，所以advantage=score
        self.assertEqual(advantages.shape, (batch_size, response_length))
        self.assertAlmostEqual(advantages[0, -1].item(), 7.0, places=4)


class TestComputeGrpoOutcomeAdvantageSplitIntegration(unittest.TestCase):
    """集成测试：测试compute_grpo_outcome_advantage_split在实际场景中的应用"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_mixed_on_off_policy_scenario(self):
        """测试混合on-policy和off-policy样本的真实场景"""
        batch_size, response_length = 8, 6
        
        # 模拟真实场景：每个prompt有4个响应，其中2个on-policy，2个off-policy
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],  # prompt 0, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 7.0],  # prompt 0, off-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],  # prompt 0, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 9.0],  # prompt 0, off-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],  # prompt 1, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 8.0],  # prompt 1, off-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],  # prompt 1, on-policy
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.0],  # prompt 1, off-policy
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0, 1, 1, 1, 1]
        on_policy_mask = torch.tensor([True, False, True, False, True, False, True, False])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask
        )
        
        # print(advantages)

        expected_advantages = torch.tensor([[-0.7071, -0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 2.1213,  2.1213,  2.1213,  2.1213,  2.1213,  2.1213],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [ 3.5355,  3.5355,  3.5355,  3.5355,  3.5355,  3.5355],
        [-0.7071, -0.7071, -0.7071, -0.7071, -0.7071, -0.7071],
        [ 3.5355,  3.5355,  3.5355,  3.5355,  3.5355,  3.5355],
        [ 0.7071,  0.7071,  0.7071,  0.7071,  0.7071,  0.7071],
        [ 2.1213,  2.1213,  2.1213,  2.1213,  2.1213,  2.1213]])
        torch.testing.assert_close(advantages, expected_advantages, atol=1e-4, rtol=1e-4)
        # 验证输出
        self.assertEqual(advantages.shape, (batch_size, response_length))
        # self.assertTrue(torch.isfinite(advantages).all())
        
        # Prompt 0的基线应该只基于on-policy样本: mean([3, 5]) = 4.0
        # Prompt 1的基线应该只基于on-policy样本: mean([2, 4]) = 3.0
        # 但所有样本（包括off-policy）都会被归一化
    
    def test_normalization_consistency(self):
        """测试归一化的一致性"""
        batch_size, response_length = 4, 5
        
        token_level_rewards = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 4.0],
        ])
        eos_mask = torch.ones(batch_size, response_length)
        index = [0, 0, 0, 0]
        on_policy_mask = torch.tensor([True, True, True, True])
        
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards, eos_mask, index, on_policy_mask, use_std=True
        )
        
        # 验证归一化后的优势均值接近0（对于同一组）
        valid_advantages = advantages[:, -1]  # 取最后一个位置
        mean_advantage = valid_advantages.mean()
        
        # 标准化后均值应该接近0
        self.assertAlmostEqual(mean_advantage.item(), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
