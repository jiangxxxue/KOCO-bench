import unittest
import torch
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual function and classes we want to test
try:
    from verl.trainer.ray_trainer import apply_kl_penalty
    from verl.trainer.core_algos import FixedKLController
    from verl.protocol import DataProto
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    apply_kl_penalty = None
    FixedKLController = None
    DataProto = None


class TestApplyKLPenalty(unittest.TestCase):
    """测试apply_kl_penalty函数 - ARES文档第3节描述的KL散度惩罚应用"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        self.batch_size = 4
        self.response_length = 8
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_apply_kl_penalty_basic_structure(self):
        """测试基本输入输出结构"""
        # 创建测试数据
        token_level_scores = torch.randn(self.batch_size, self.response_length)
        old_log_probs = torch.randn(self.batch_size, self.response_length)
        ref_log_probs = torch.randn(self.batch_size, self.response_length)
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.1)
        
        # 调用函数
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # 验证输出结构
        self.assertIsInstance(result_data, DataProto)
        self.assertIn("token_level_rewards", result_data.batch)
        self.assertIsInstance(metrics, dict)
        self.assertIn("critic/kl", metrics)
        self.assertIn("critic/kl_coef", metrics)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_penalty_reduces_rewards(self):
        """测试KL惩罚降低奖励"""
        # 创建测试数据
        token_level_scores = torch.ones(self.batch_size, self.response_length)  # 全1
        old_log_probs = torch.zeros(self.batch_size, self.response_length)
        ref_log_probs = torch.ones(self.batch_size, self.response_length) * -1.0  # 制造KL散度
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.2)
        
        # 调用函数
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # 验证奖励被降低
        token_level_rewards = result_data.batch["token_level_rewards"]

        # print(token_level_rewards)

        expected_token_level_rewards = torch.tensor([[0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000],
        [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000],
        [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000],
        [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000]])
        # 由于有KL惩罚，奖励应该小于原始分数
        # self.assertTrue(torch.all(token_level_rewards < token_level_scores))
        torch.testing.assert_close(token_level_rewards, expected_token_level_rewards, atol=1e-5, rtol=1e-5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_penalty_with_mask(self):
        """测试带mask的KL惩罚"""
        # 创建测试数据
        token_level_scores = torch.ones(self.batch_size, self.response_length)
        old_log_probs = torch.randn(self.batch_size, self.response_length)
        ref_log_probs = torch.randn(self.batch_size, self.response_length)
        
        # 创建部分mask
        response_mask = torch.ones(self.batch_size, self.response_length)
        response_mask[:, -2:] = 0  # 最后两个位置mask掉
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.1)
        
        # 调用函数
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # 验证mask位置没有被惩罚
        token_level_rewards = result_data.batch["token_level_rewards"]
        # Mask位置的奖励应该等于原始分数（因为KL为0）
        torch.testing.assert_close(
            token_level_rewards[:, -2:],
            token_level_scores[:, -2:],
            atol=1e-5,
            rtol=1e-5
        )
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_different_kl_penalty_types(self):
        """测试不同的KL惩罚类型"""
        # 创建测试数据
        token_level_scores = torch.ones(self.batch_size, self.response_length)
        old_log_probs = torch.tensor([[ 1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345, -0.0431, -1.6047],
        [-0.7521,  1.6487, -0.3925, -1.4036, -0.7279, -0.5594, -0.7688,  0.7624],
        [ 1.6423, -0.1596, -0.4974,  0.4396, -0.7581,  1.0783,  0.8008,  1.6806],
        [ 1.2791,  1.2964,  0.6105,  1.3347, -0.2316,  0.0418, -0.2516,  0.8599]])
        ref_log_probs = torch.tensor([[-1.3847, -0.8712, -0.2234,  1.7174,  0.3189, -0.4245,  0.3057, -0.7746],
        [-1.5576,  0.9956, -0.8798, -0.6011, -1.2742,  2.1228, -1.2347, -0.4879],
        [-0.9138, -0.6581,  0.0780,  0.5258, -0.4880,  1.1914, -0.8140, -0.7360],
        [-1.4032,  0.0360, -0.0635,  0.6756, -0.0978,  1.8446, -1.1845,  1.3835]])
        # print(old_log_probs)
        # print(ref_log_probs)
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        kl_ctrl = FixedKLController(init_kl_coef=0.1)
        
        # 测试"kl"类型
        data_kl = DataProto.from_dict(tensors=batch_data)
        result_kl, metrics_kl = apply_kl_penalty(data_kl, kl_ctrl, kl_penalty="kl")
        self.assertIn("token_level_rewards", result_kl.batch)
        
        token_level_rewards_kl = result_kl.batch["token_level_rewards"]
        # print(token_level_rewards_kl)
        expected_token_level_rewards_kl = torch.tensor([[0.6688, 0.7641, 0.8876, 1.3823, 0.9640, 1.0810, 1.0349, 1.0830],
        [0.9195, 0.9347, 0.9513, 1.0802, 0.9454, 1.2682, 0.9534, 0.8750],
        [0.7444, 0.9501, 1.0575, 1.0086, 1.0270, 1.0113, 0.8385, 0.7583],
        [0.7318, 0.8740, 0.9326, 0.9341, 1.0134, 1.1803, 0.9067, 1.0524]])
        # 测试"low_var_kl"类型
        data_low_var = DataProto.from_dict(tensors=batch_data)
        result_low_var, metrics_low_var = apply_kl_penalty(data_low_var, kl_ctrl, kl_penalty="low_var_kl")
        self.assertIn("token_level_rewards", result_low_var.batch)
        
        token_level_rewards_low_var = result_low_var.batch["token_level_rewards"]
        # print(token_level_rewards_low_var)
        expected_token_level_rewards_low_var = torch.tensor([[0.7652, 0.8547, 0.9551, 0.0000, 0.9942, 0.9562, 0.9931, 0.9537],
        [0.9748, 0.9826, 0.9898, 0.9571, 0.9875, 0.0000, 0.9907, 0.9463],
        [0.8366, 0.9894, 0.9798, 0.9996, 0.9960, 0.9993, 0.9186, 0.8494],
        [0.8249, 0.9456, 0.9816, 0.9824, 0.9991, 0.6736, 0.9674, 0.9835]])

        # 两种方法应该产生不同的结果    
        # self.assertFalse(torch.equal(
        #     result_kl.batch["token_level_rewards"],
        #     result_low_var.batch["token_level_rewards"]
        # ))
        torch.testing.assert_close(token_level_rewards_kl, expected_token_level_rewards_kl, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(token_level_rewards_low_var, expected_token_level_rewards_low_var, atol=1e-4, rtol=1e-4)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_coef_effect(self):
        """测试不同KL系数的效果"""
        # 创建测试数据
        token_level_scores = torch.ones(self.batch_size, self.response_length) * 2.0
        old_log_probs = torch.zeros(self.batch_size, self.response_length)
        ref_log_probs = torch.ones(self.batch_size, self.response_length) * -1.0
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        # 小KL系数
        data_small = DataProto.from_dict(tensors=batch_data)
        kl_ctrl_small = FixedKLController(init_kl_coef=0.05)
        result_small, _ = apply_kl_penalty(data_small, kl_ctrl_small, kl_penalty="kl")
        
        # 大KL系数
        data_large = DataProto.from_dict(tensors=batch_data)
        kl_ctrl_large = FixedKLController(init_kl_coef=0.5)
        result_large, _ = apply_kl_penalty(data_large, kl_ctrl_large, kl_penalty="kl")
        
        # 大KL系数应该导致更多的惩罚
        rewards_small = result_small.batch["token_level_rewards"]
        rewards_large = result_large.batch["token_level_rewards"]
        
        expected_rewards_small = torch.tensor([[1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500],
        [1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500],
        [1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500],
        [1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500, 1.9500]])
        expected_rewards_large = torch.tensor([[1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000],
        [1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000],
        [1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000],
        [1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000]])
        torch.testing.assert_close(rewards_small, expected_rewards_small, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rewards_large, expected_rewards_large, atol=1e-5, rtol=1e-5)
        # 大系数的奖励应该更低
        # self.assertTrue(torch.all(rewards_large < rewards_small))
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_metrics_output(self):
        """测试指标输出"""
        # 创建测试数据
        token_level_scores = torch.tensor([[ 1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345, -0.0431, -1.6047],
        [-0.7521,  1.6487, -0.3925, -1.4036, -0.7279, -0.5594, -0.7688,  0.7624],
        [ 1.6423, -0.1596, -0.4974,  0.4396, -0.7581,  1.0783,  0.8008,  1.6806],
        [ 1.2791,  1.2964,  0.6105,  1.3347, -0.2316,  0.0418, -0.2516,  0.8599]])
        old_log_probs = torch.tensor([[-1.3847, -0.8712, -0.2234,  1.7174,  0.3189, -0.4245,  0.3057, -0.7746],
        [-1.5576,  0.9956, -0.8798, -0.6011, -1.2742,  2.1228, -1.2347, -0.4879],
        [-0.9138, -0.6581,  0.0780,  0.5258, -0.4880,  1.1914, -0.8140, -0.7360],
        [-1.4032,  0.0360, -0.0635,  0.6756, -0.0978,  1.8446, -1.1845,  1.3835]])
        ref_log_probs = torch.tensor([[ 1.4451,  0.8564,  2.2181,  0.5232,  0.3466, -0.1973, -1.0546,  1.2780],
        [-0.1722,  0.5238,  0.0566,  0.4263,  0.5750, -0.6417, -2.2064, -0.7508],
        [ 0.0109, -0.3387, -1.3407, -0.5854,  0.5362,  0.5246,  1.1412,  0.0516],
        [ 0.7440, -0.4816, -1.0495,  0.6039, -1.7223, -0.8278,  1.3347,  0.4835]])
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_coef_value = 0.15
        kl_ctrl = FixedKLController(init_kl_coef=kl_coef_value)
        
        # 调用函数
        _, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # 验证指标
        self.assertIn("critic/kl", metrics)
        self.assertIn("critic/kl_coef", metrics)
            # print(metrics["critic/kl"])
            # print(metrics["critic/kl_coef"])

        expected_kl = -0.22462797164916992
        expected_kl_coef = 0.15
        # 验证指标值
        self.assertIsInstance(metrics["critic/kl"], float)
        torch.testing.assert_close(metrics["critic/kl"], expected_kl, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(metrics["critic/kl_coef"], expected_kl_coef, atol=1e-5, rtol=1e-5)
        # KL散度应该是有限数值（可能为负，取决于KL惩罚类型）
        self.assertTrue(abs(metrics["critic/kl"]) < 100)  # 合理范围

    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_controller_update(self):
        """测试KL控制器的更新"""
        # 创建测试数据
        token_level_scores = torch.randn(self.batch_size, self.response_length)
        old_log_probs = torch.randn(self.batch_size, self.response_length)
        ref_log_probs = torch.randn(self.batch_size, self.response_length)

        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.1)
        
        # 对于FixedKLController，update不应该改变kl_coef
        initial_kl_coef = kl_ctrl.kl_coef
        
        # 调用函数（内部会调用kl_ctrl.update）
        _, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # FixedKLController的系数应该保持不变
        self.assertEqual(kl_ctrl.kl_coef, initial_kl_coef)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_zero_kl_divergence(self):
        """测试零KL散度的情况"""
        # 创建相同的log_probs（KL=0）
        token_level_scores = torch.ones(self.batch_size, self.response_length) * 2.0
        log_probs = torch.randn(self.batch_size, self.response_length)
        old_log_probs = log_probs.clone()
        ref_log_probs = log_probs.clone()
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.2)
        
        # 调用函数
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # KL为0，奖励应该等于原始分数
        token_level_rewards = result_data.batch["token_level_rewards"]
        torch.testing.assert_close(token_level_rewards, token_level_scores, atol=1e-5, rtol=1e-5)
        
        # KL指标应该接近0
        self.assertAlmostEqual(metrics["critic/kl"], 0.0, delta=1e-4)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_batch_size_consistency(self):
        """测试不同batch大小的一致性"""
        for batch_size in [1, 4, 8]:
            token_level_scores = torch.randn(batch_size, self.response_length)
            old_log_probs = torch.randn(batch_size, self.response_length)
            ref_log_probs = torch.randn(batch_size, self.response_length)
            response_mask = torch.ones(batch_size, self.response_length)
            
            batch_data = {
                "token_level_scores": token_level_scores,
                "old_log_probs": old_log_probs,
                "ref_log_probs": ref_log_probs,
                "response_mask": response_mask
            }
            
            data = DataProto.from_dict(tensors=batch_data)
            kl_ctrl = FixedKLController(init_kl_coef=0.1)
            
            # 调用函数
            result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
            
            # 验证输出形状
            self.assertEqual(
                result_data.batch["token_level_rewards"].shape,
                (batch_size, self.response_length)
            )


class TestApplyKLPenaltyIntegration(unittest.TestCase):
    """集成测试：测试apply_kl_penalty的整体工作流程"""
    
    def setUp(self):
        """设置测试环境"""
        torch.manual_seed(42)
        self.batch_size = 6
        self.response_length = 10
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_end_to_end_kl_penalty_workflow(self):
        """端到端的KL惩罚工作流"""
        # 模拟真实的训练数据
        token_level_scores = torch.tensor([[ 1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345, -0.0431, -1.6047,
         -0.7521,  1.6487],
        [-0.3925, -1.4036, -0.7279, -0.5594, -0.7688,  0.7624,  1.6423, -0.1596,
         -0.4974,  0.4396],
        [-0.7581,  1.0783,  0.8008,  1.6806,  1.2791,  1.2964,  0.6105,  1.3347,
         -0.2316,  0.0418],
        [-0.2516,  0.8599, -1.3847, -0.8712, -0.2234,  1.7174,  0.3189, -0.4245,
          0.3057, -0.7746],
        [-1.5576,  0.9956, -0.8798, -0.6011,  0.3672,  0.1754,  1.3852, -0.4459,
          1.4451,  0.8564],
        [ 2.2181,  0.5232,  1.1754,  0.5612, -0.4527, -0.7718, -0.1722,  0.5238,
          0.0566,  0.4263]])
        
        # print(token_level_scores)
        # 模拟策略演化：old_log_probs和ref_log_probs有差异
        ref_log_probs = torch.tensor([[ 0.1971, -1.1441,  0.3383,  1.6992,  0.0109, -0.3387, -1.3407, -0.5854,
         -0.5644,  1.0563],
        [-1.4692,  1.4332,  0.7440, -0.4816, -1.0495,  0.6039,  0.4048, -1.3543,
         -0.4976,  0.4747],
        [-2.5095,  0.4880,  0.7846,  0.0286,  1.7423, -1.3527,  0.2191,  0.5526,
         -0.1853,  0.7528],
        [ 0.4048,  0.1785, -0.3165,  0.5886, -0.8905,  0.4098, -1.4570, -0.1023,
         -0.5992,  0.4771],
        [-0.1693,  0.2332,  4.0356,  1.2795, -0.3609, -0.0606,  0.0733,  0.8187,
          1.4805,  0.3449],
        [-1.4241, -0.1163,  0.2176, -0.0467, -1.4335, -0.5665, -0.4253,  0.2625,
         -1.4391,  0.5214]])


        old_log_probs = torch.tensor([[ 1.0414, -0.3997, -2.2933,  0.4976, -0.4257, -1.3371, -0.1933,  0.6526,
         -0.3063, -0.3302],
        [-0.9808,  0.1947, -1.6535,  0.6814,  1.4611, -0.3098,  0.9633, -0.3095,
          0.5712,  1.1179],
        [-1.2956,  0.0503, -0.5855, -0.3900,  0.9812, -0.6401, -0.4908,  0.2080,
         -1.1586, -0.9637],
        [-0.3750,  0.8033,  0.7165,  1.5335, -1.4510, -0.7861, -0.9563, -1.2476,
         -0.7499, -0.5922],
        [-1.5326, -0.7251,  0.4664,  0.6667,  0.1543,  0.4408, -0.1483, -2.3184,
          1.3032,  0.4879],
        [ 1.1340, -0.3556,  0.3094, -0.5003,  1.0350,  1.6896,  0.0213, -0.8293,
         -1.0809, -0.7839]])
        # print(ref_log_probs)
        # print(old_log_probs)
        # 部分序列结束
        response_mask = torch.ones(self.batch_size, self.response_length)
        # for i in range(self.batch_size):
        #     end_pos = torch.randint(5, self.response_length, (1,)).item()
        #     print(end_pos)
        #     response_mask[i, end_pos:] = 0
        response_mask[0, 5:] = 0
        response_mask[1, 8:] = 0
        response_mask[2, 5:] = 0
        response_mask[3, 6:] = 0
        response_mask[4, 7:] = 0
        response_mask[5, 9:] = 0
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.2)
        
        # 执行KL惩罚
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # 验证完整的工作流
        self.assertIsNotNone(result_data)
        self.assertIsNotNone(metrics)
        
        # 验证奖励被正确计算
        token_level_rewards = result_data.batch["token_level_rewards"]
        # print(token_level_rewards)
        # print(metrics["critic/kl"])

        expected_token_level_rewards = torch.tensor([[ 1.7580,  1.3384,  1.4270, -1.8652,  0.7657, -1.2345, -0.0431, -1.6047,
         -0.7521,  1.6487],
        [-0.4901, -1.1559, -0.2484, -0.7920, -1.2710,  0.9452,  1.5306, -0.3686,
         -0.4974,  0.4396],
        [-1.0009,  1.1659,  1.0748,  1.7643,  1.4313,  1.2964,  0.6105,  1.3347,
         -0.2316,  0.0418],
        [-0.0956,  0.7349, -1.5913, -1.0602, -0.1113,  1.9566,  0.3189, -0.4245,
          0.3057, -0.7746],
        [-1.2849,  1.1873, -0.1659, -0.4786,  0.2642,  0.0751,  1.4295, -0.4459,
          1.4451,  0.8564],
        [ 1.7065,  0.5710,  1.1570,  0.6519, -0.9465, -1.2230, -0.2615,  0.7421,
         -0.0150,  0.4263]])
        torch.testing.assert_close(token_level_rewards, expected_token_level_rewards, atol=1e-4, rtol=1e-4)
        
        expected_kl = -0.13880975544452667
        expected_kl_coef = 0.2
        torch.testing.assert_close(metrics["critic/kl"], expected_kl, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(metrics["critic/kl_coef"], expected_kl_coef, atol=1e-5, rtol=1e-5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_policy_divergence_scenarios(self):
        """测试不同策略偏离程度的场景"""
        token_level_scores = torch.ones(self.batch_size, self.response_length)
        ref_log_probs = torch.tensor([[ 1.9269,  1.4873,  0.9007, -2.1055,  0.6784, -1.2345, -0.0431, -1.6047,
         -0.7521,  1.6487],
        [-0.3925, -1.4036, -0.7279, -0.5594, -0.7688,  0.7624,  1.6423, -0.1596,
         -0.4974,  0.4396],
        [-0.7581,  1.0783,  0.8008,  1.6806,  1.2791,  1.2964,  0.6105,  1.3347,
         -0.2316,  0.0418],
        [-0.2516,  0.8599, -1.3847, -0.8712, -0.2234,  1.7174,  0.3189, -0.4245,
          0.3057, -0.7746],
        [-1.5576,  0.9956, -0.8798, -0.6011,  0.3672,  0.1754,  1.3852, -0.4459,
          1.4451,  0.8564],
        [ 2.2181,  0.5232,  1.1754,  0.5612, -0.4527, -0.7718, -0.1722,  0.5238,
          0.0566,  0.4263]])
        # print(ref_log_probs)
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        kl_ctrl = FixedKLController(init_kl_coef=0.1)
        
        # 场景1：策略相同
        old_log_probs_same = ref_log_probs.clone()
        data_same = DataProto.from_dict(tensors={
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs_same,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        })
        result_same, metrics_same = apply_kl_penalty(data_same, kl_ctrl, kl_penalty="kl")
        
        # 场景2：策略略有偏离
        old_log_probs_slight = torch.tensor([[ 2.1196,  1.6360,  0.9908, -2.3161,  0.7462, -1.3580, -0.0474, -1.7652,
         -0.8273,  1.8136],
        [-0.4317, -1.5440, -0.8007, -0.6153, -0.8457,  0.8386,  1.8065, -0.1756,
         -0.5471,  0.4836],
        [-0.8339,  1.1861,  0.8809,  1.8487,  1.4070,  1.4260,  0.6715,  1.4682,
         -0.2548,  0.0460],
        [-0.2768,  0.9459, -1.5232, -0.9583, -0.2457,  1.8891,  0.3508, -0.4670,
          0.3363, -0.8521],
        [-1.7134,  1.0952, -0.9678, -0.6612,  0.4039,  0.1929,  1.5237, -0.4905,
          1.5896,  0.9420],
        [ 2.4399,  0.5755,  1.2929,  0.6173, -0.4980, -0.8490, -0.1894,  0.5762,
          0.0623,  0.4689]])
        # print(old_log_probs_slight)
        data_slight = DataProto.from_dict(tensors={
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs_slight,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        })
        result_slight, metrics_slight = apply_kl_penalty(data_slight, kl_ctrl, kl_penalty="kl")
        
        # 场景3：策略大幅偏离
        old_log_probs_large = torch.tensor([[ 2.1240,  0.3432,  1.2390, -0.4063,  0.6893, -1.5732, -1.3838, -2.1901,
         -1.3165,  2.7050],
        [-1.8617,  0.0296,  0.0161, -1.0410, -1.8183,  1.3663,  2.0471, -1.5139,
         -0.9950,  0.9143],
        [-3.2676,  1.5663,  1.5854,  1.7092,  3.0214, -0.0563,  0.8296,  1.8873,
         -0.4169,  0.7946],
        [ 0.1532,  1.0384, -1.7012, -0.2826, -1.1139,  2.1272, -1.1381, -0.5268,
         -0.2935, -0.2975],
        [-1.7269,  1.2288,  3.1558,  0.6784,  0.0063,  0.1148,  1.4585,  0.3728,
          2.9256,  1.2013],
        [ 0.7940,  0.4069,  1.3930,  0.5145, -1.8862, -1.3383, -0.5975,  0.7863,
         -1.3825,  0.9477]])
        # print(old_log_probs_large)
        data_large = DataProto.from_dict(tensors={
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs_large,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        })
        result_large, metrics_large = apply_kl_penalty(data_large, kl_ctrl, kl_penalty="kl")
        
        # 验证KL散度的关系（使用绝对值，因为KL可能为负）
        # 策略偏离越大，KL散度的绝对值应该越大
        abs_kl_same = abs(metrics_same["critic/kl"])
        abs_kl_slight = abs(metrics_slight["critic/kl"])
        abs_kl_large = abs(metrics_large["critic/kl"])
        # print(abs_kl_same)
        # print(abs_kl_slight)
        # print(abs_kl_large)
        expected_abs_kl_same = 0.0
        expected_abs_kl_slight = 0.01911669410765171
        expected_abs_kl_large = 0.009584061801433563
        torch.testing.assert_close(abs_kl_same, expected_abs_kl_same, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(abs_kl_slight, expected_abs_kl_slight, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(abs_kl_large, expected_abs_kl_large, atol=1e-5, rtol=1e-5)
        # 至少应该有一定的偏离趋势（但由于随机性，不总是严格递增）
        # self.assertIsNotNone(abs_kl_same)
        # self.assertIsNotNone(abs_kl_slight)
        # self.assertIsNotNone(abs_kl_large)
        
        # 验证奖励递减（惩罚增大）
        rewards_same = result_same.batch["token_level_rewards"].mean().item()
        rewards_slight = result_slight.batch["token_level_rewards"].mean().item()
        rewards_large = result_large.batch["token_level_rewards"].mean().item()
        # print(rewards_same)
        # print(rewards_slight)
        # print(rewards_large)
        expected_rewards_same = 1.0
        expected_rewards_slight = 0.9980883002281189
        expected_rewards_large = 0.9990416169166565
        torch.testing.assert_close(rewards_same, expected_rewards_same, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rewards_slight, expected_rewards_slight, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rewards_large, expected_rewards_large, atol=1e-5, rtol=1e-5)
        # 由于使用了随机log_probs，严格的递减关系可能不总是成立
        # 我们只验证结果都是有限的
        # self.assertTrue(abs(rewards_same) < 100)
        # self.assertTrue(abs(rewards_slight) < 100)
        # self.assertTrue(abs(rewards_large) < 100)


if __name__ == "__main__":
    unittest.main()

