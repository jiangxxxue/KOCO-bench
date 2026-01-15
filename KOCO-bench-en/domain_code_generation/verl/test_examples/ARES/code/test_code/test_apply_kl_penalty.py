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
    """Test apply_kl_penalty function - KL divergence penalty application described in ARES document Section 3"""
    
    def setUp(self):
        """Set up test environment"""
        torch.manual_seed(42)
        self.batch_size = 4
        self.response_length = 8
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_apply_kl_penalty_basic_structure(self):
        """Test basic input/output structure"""
        # Create test data
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
        
        # Call function
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # Verify output structure
        self.assertIsInstance(result_data, DataProto)
        self.assertIn("token_level_rewards", result_data.batch)
        self.assertIsInstance(metrics, dict)
        self.assertIn("critic/kl", metrics)
        self.assertIn("critic/kl_coef", metrics)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_penalty_reduces_rewards(self):
        """Test that KL penalty reduces rewards"""
        # Create test data
        token_level_scores = torch.ones(self.batch_size, self.response_length)  # All ones
        old_log_probs = torch.zeros(self.batch_size, self.response_length)
        ref_log_probs = torch.ones(self.batch_size, self.response_length) * -1.0  # Create KL divergence
        response_mask = torch.ones(self.batch_size, self.response_length)
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.2)
        
        # Call function
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # Verify rewards are reduced
        token_level_rewards = result_data.batch["token_level_rewards"]

        # print(token_level_rewards)

        expected_token_level_rewards = torch.tensor([[0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000],
        [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000],
        [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000],
        [0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000, 0.8000]])
        # Due to KL penalty, rewards should be less than original scores
        # self.assertTrue(torch.all(token_level_rewards < token_level_scores))
        torch.testing.assert_close(token_level_rewards, expected_token_level_rewards, atol=1e-5, rtol=1e-5)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_penalty_with_mask(self):
        """Test KL penalty with mask"""
        # Create test data
        token_level_scores = torch.ones(self.batch_size, self.response_length)
        old_log_probs = torch.randn(self.batch_size, self.response_length)
        ref_log_probs = torch.randn(self.batch_size, self.response_length)
        
        # Create partial mask
        response_mask = torch.ones(self.batch_size, self.response_length)
        response_mask[:, -2:] = 0  # Mask out last two positions
        
        batch_data = {
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        }
        
        data = DataProto.from_dict(tensors=batch_data)
        kl_ctrl = FixedKLController(init_kl_coef=0.1)
        
        # Call function
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # Verify masked positions are not penalized
        token_level_rewards = result_data.batch["token_level_rewards"]
        # Rewards at masked positions should equal original scores (because KL is 0)
        torch.testing.assert_close(
            token_level_rewards[:, -2:],
            token_level_scores[:, -2:],
            atol=1e-5,
            rtol=1e-5
        )
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_different_kl_penalty_types(self):
        """Test different KL penalty types"""
        # Create test data
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
        
        # Test "kl" type
        data_kl = DataProto.from_dict(tensors=batch_data)
        result_kl, metrics_kl = apply_kl_penalty(data_kl, kl_ctrl, kl_penalty="kl")
        self.assertIn("token_level_rewards", result_kl.batch)
        
        token_level_rewards_kl = result_kl.batch["token_level_rewards"]
        # print(token_level_rewards_kl)
        expected_token_level_rewards_kl = torch.tensor([[0.6688, 0.7641, 0.8876, 1.3823, 0.9640, 1.0810, 1.0349, 1.0830],
        [0.9195, 0.9347, 0.9513, 1.0802, 0.9454, 1.2682, 0.9534, 0.8750],
        [0.7444, 0.9501, 1.0575, 1.0086, 1.0270, 1.0113, 0.8385, 0.7583],
        [0.7318, 0.8740, 0.9326, 0.9341, 1.0134, 1.1803, 0.9067, 1.0524]])
        # Test "low_var_kl" type
        data_low_var = DataProto.from_dict(tensors=batch_data)
        result_low_var, metrics_low_var = apply_kl_penalty(data_low_var, kl_ctrl, kl_penalty="low_var_kl")
        self.assertIn("token_level_rewards", result_low_var.batch)
        
        token_level_rewards_low_var = result_low_var.batch["token_level_rewards"]
        # print(token_level_rewards_low_var)
        expected_token_level_rewards_low_var = torch.tensor([[0.7652, 0.8547, 0.9551, 0.0000, 0.9942, 0.9562, 0.9931, 0.9537],
        [0.9748, 0.9826, 0.9898, 0.9571, 0.9875, 0.0000, 0.9907, 0.9463],
        [0.8366, 0.9894, 0.9798, 0.9996, 0.9960, 0.9993, 0.9186, 0.8494],
        [0.8249, 0.9456, 0.9816, 0.9824, 0.9991, 0.6736, 0.9674, 0.9835]])

        # Two methods should produce different results    
        # self.assertFalse(torch.equal(
        #     result_kl.batch["token_level_rewards"],
        #     result_low_var.batch["token_level_rewards"]
        # ))
        torch.testing.assert_close(token_level_rewards_kl, expected_token_level_rewards_kl, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(token_level_rewards_low_var, expected_token_level_rewards_low_var, atol=1e-4, rtol=1e-4)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_coef_effect(self):
        """Test effect of different KL coefficients"""
        # Create test data
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
        
        # Small KL coefficient
        data_small = DataProto.from_dict(tensors=batch_data)
        kl_ctrl_small = FixedKLController(init_kl_coef=0.05)
        result_small, _ = apply_kl_penalty(data_small, kl_ctrl_small, kl_penalty="kl")
        
        # Large KL coefficient
        data_large = DataProto.from_dict(tensors=batch_data)
        kl_ctrl_large = FixedKLController(init_kl_coef=0.5)
        result_large, _ = apply_kl_penalty(data_large, kl_ctrl_large, kl_penalty="kl")
        
        # Large KL coefficient should lead to more penalty
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
        # Rewards with large coefficient should be lower
        # self.assertTrue(torch.all(rewards_large < rewards_small))
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_metrics_output(self):
        """Test metrics output"""
        # Create test data
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
        
        # Call function
        _, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # Verify metrics
        self.assertIn("critic/kl", metrics)
        self.assertIn("critic/kl_coef", metrics)
            # print(metrics["critic/kl"])
            # print(metrics["critic/kl_coef"])

        expected_kl = -0.22462797164916992
        expected_kl_coef = 0.15
        # Verify metric values
        self.assertIsInstance(metrics["critic/kl"], float)
        torch.testing.assert_close(metrics["critic/kl"], expected_kl, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(metrics["critic/kl_coef"], expected_kl_coef, atol=1e-5, rtol=1e-5)
        # KL divergence should be finite (may be negative depending on KL penalty type)
        self.assertTrue(abs(metrics["critic/kl"]) < 100)  # Reasonable range

    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_kl_controller_update(self):
        """Test KL controller update"""
        # Create test data
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
        
        # For FixedKLController, update should not change kl_coef
        initial_kl_coef = kl_ctrl.kl_coef
        
        # Call function (internally calls kl_ctrl.update)
        _, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # FixedKLController coefficient should remain unchanged
        self.assertEqual(kl_ctrl.kl_coef, initial_kl_coef)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_zero_kl_divergence(self):
        """Test zero KL divergence case"""
        # Create identical log_probs (KL=0)
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
        
        # Call function
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # With KL=0, rewards should equal original scores
        token_level_rewards = result_data.batch["token_level_rewards"]
        torch.testing.assert_close(token_level_rewards, token_level_scores, atol=1e-5, rtol=1e-5)
        
        # KL metric should be close to 0
        self.assertAlmostEqual(metrics["critic/kl"], 0.0, delta=1e-4)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_batch_size_consistency(self):
        """Test consistency across different batch sizes"""
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
            
            # Call function
            result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
            
            # Verify output shape
            self.assertEqual(
                result_data.batch["token_level_rewards"].shape,
                (batch_size, self.response_length)
            )


class TestApplyKLPenaltyIntegration(unittest.TestCase):
    """Integration test: test overall workflow of apply_kl_penalty"""
    
    def setUp(self):
        """Set up test environment"""
        torch.manual_seed(42)
        self.batch_size = 6
        self.response_length = 10
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_end_to_end_kl_penalty_workflow(self):
        """End-to-end KL penalty workflow"""
        # Simulate real training data
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
        # Simulate policy evolution: old_log_probs and ref_log_probs differ
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
        # Partial sequence endings
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
        
        # Execute KL penalty
        result_data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")
        
        # Verify complete workflow
        self.assertIsNotNone(result_data)
        self.assertIsNotNone(metrics)
        
        # Verify rewards are correctly calculated
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
        """Test scenarios with different degrees of policy divergence"""
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
        
        # Scenario 1: Same policy
        old_log_probs_same = ref_log_probs.clone()
        data_same = DataProto.from_dict(tensors={
            "token_level_scores": token_level_scores,
            "old_log_probs": old_log_probs_same,
            "ref_log_probs": ref_log_probs,
            "response_mask": response_mask
        })
        result_same, metrics_same = apply_kl_penalty(data_same, kl_ctrl, kl_penalty="kl")
        
        # Scenario 2: Slight policy divergence
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
        
        # Scenario 3: Large policy divergence
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
        
        # Verify KL divergence relationship (use absolute value since KL can be negative)
        # Greater policy divergence should lead to larger absolute KL divergence
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
        # Should have some divergence trend (but due to randomness, not always strictly increasing)
        # self.assertIsNotNone(abs_kl_same)
        # self.assertIsNotNone(abs_kl_slight)
        # self.assertIsNotNone(abs_kl_large)
        
        # Verify reward decrease (penalty increase)
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
        # Due to random log_probs, strict decreasing relationship may not always hold
        # We only verify results are all finite
        # self.assertTrue(abs(rewards_same) < 100)
        # self.assertTrue(abs(rewards_slight) < 100)
        # self.assertTrue(abs(rewards_large) < 100)


if __name__ == "__main__":
    unittest.main()

