import unittest
import torch
import sys
import os

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import fidelity_reward


class TestFidelityReward(unittest.TestCase):
    """Test fidelity_reward function - compute ranking fidelity reward between a single sample pair"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
    
    def test_fidelity_reward_basic(self):
        """Test basic fidelity reward computation"""
        # Test case 1: pred1 > pred2, gt = 1.0 (correct prediction)
        pred1 = torch.tensor(4.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.3, dtype=torch.float32, device=self.device)
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # print(reward)

        expected_reward = torch.tensor(0.9946)
        # Verify output
        torch.testing.assert_close(reward, expected_reward, atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_correct_ranking(self):
        """Test correct prediction ranking"""
        # pred1 > pred2, gt = 1.0
        pred1 = torch.tensor(5.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.1, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.1, dtype=torch.float32, device=self.device)
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # print(reward)

        expected_reward = torch.tensor(1.0010)
        # Verify output
        torch.testing.assert_close(reward, expected_reward, atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_incorrect_ranking(self):
        """Test incorrect prediction ranking"""
        # pred1 < pred2, but gt = 1.0 (incorrect prediction)
        pred1 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(5.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.1, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.1, dtype=torch.float32, device=self.device)
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # print(reward)

        expected_reward = torch.tensor(0.0020)
        # Verify output
        torch.testing.assert_close(reward, expected_reward, atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_equal_quality(self):
        """Test equal quality case (gt = 0.5)"""
        pred1 = torch.tensor(3.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(3.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.2, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.2, dtype=torch.float32, device=self.device)
        gt = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # When predictions are equal and ground truth is also equal, reward should be close to maximum
        # print(reward)

        expected_reward = torch.tensor(1.0000)
        # Verify output
        torch.testing.assert_close(reward, expected_reward, atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_with_variance(self):
        """Test impact of variance on reward"""
        pred1 = torch.tensor(4.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        # Small variance
        reward_small_var = fidelity_reward(pred1, pred2, torch.tensor(0.1, dtype=torch.float32, device=self.device), torch.tensor(0.1, dtype=torch.float32, device=self.device), gt, self.device)
        
        # Large variance
        reward_large_var = fidelity_reward(pred1, pred2, torch.tensor(2.0, dtype=torch.float32, device=self.device), torch.tensor(2.0, dtype=torch.float32, device=self.device), gt, self.device)
        
        # print(reward_small_var)
        # print(reward_large_var)
        # Smaller variance means higher confidence in the difference, reward difference should be more pronounced
        torch.testing.assert_close(reward_small_var, torch.tensor(1.0010), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(reward_large_var, torch.tensor(0.9182), atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_tensor_inputs(self):
        """Test tensor inputs"""
        pred1 = torch.tensor(4.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.3, dtype=torch.float32, device=self.device)
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        
        # print(reward)
        torch.testing.assert_close(reward, torch.tensor(0.9946), atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_edge_cases(self):
        """Test edge cases"""
        # Test case 1: very small variance
        pred1 = torch.tensor(3.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(1e-8, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(1e-8, dtype=torch.float32, device=self.device)
        gt = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        
        reward = fidelity_reward(pred1, pred2, var1, var2, gt, self.device)
        # print(reward)
        
        # Test case 2: zero variance (relies on epsilon protection)
        reward_zero_var = fidelity_reward(pred1, pred2, torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0.0, dtype=torch.float32, device=self.device), gt, self.device)
        # print(reward_zero_var)

        expected_reward = torch.tensor(1.0010) 
        torch.testing.assert_close(reward, expected_reward, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(reward_zero_var, expected_reward, atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_different_gt_values(self):
        """Test different ground truth ranking relationships"""
        pred1 = torch.tensor(4.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        
        # gt = 1.0: pred1 should be better than pred2
        gt_1 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        reward_1 = fidelity_reward(pred1, pred2, var1, var2, gt_1, self.device)
        
        # gt = 0.0: pred2 should be better than pred1
        gt_0 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        reward_0 = fidelity_reward(pred1, pred2, var1, var2, gt_0, self.device)
        
        # gt = 0.5: both are equal
        gt_05 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        reward_05 = fidelity_reward(pred1, pred2, var1, var2, gt_05, self.device)

        # print(reward_1)
        # print(reward_0)
        # print(reward_05)

        expected_reward_1 = torch.tensor(0.9896)
        expected_reward_0 = torch.tensor(0.1518)
        expected_reward_05 = torch.tensor(0.8057)
        torch.testing.assert_close(reward_1, expected_reward_1, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(reward_0, expected_reward_0, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(reward_05, expected_reward_05, atol=1e-4, rtol=1e-4)
    
    def test_fidelity_reward_symmetry(self):
        """Test symmetry: swapping pred1 and pred2 with corresponding gt change should yield same reward"""
        pred1 = torch.tensor(4.0, dtype=torch.float32, device=self.device)
        pred2 = torch.tensor(2.0, dtype=torch.float32, device=self.device)
        var1 = torch.tensor(0.5, dtype=torch.float32, device=self.device)
        var2 = torch.tensor(0.3, dtype=torch.float32, device=self.device)
        
        # Original case
        gt_1 = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        reward_1 = fidelity_reward(pred1, pred2, var1, var2, gt_1, self.device)
        
        # Swapped case
        gt_0 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        reward_2 = fidelity_reward(pred2, pred1, var2, var1, gt_0, self.device)
        
        # Due to symmetry, rewards should be the same
        torch.testing.assert_close(reward_1, reward_2, atol=1e-5, rtol=1e-5)




if __name__ == "__main__":
    import math
    unittest.main()
