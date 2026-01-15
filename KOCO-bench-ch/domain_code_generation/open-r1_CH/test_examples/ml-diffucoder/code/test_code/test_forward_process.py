import unittest
import torch
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual class we want to test
from open_r1.coupled_grpo import DiffuGRPOTrainer


class TestForwardProcess(unittest.TestCase):
    """测试DiffuGRPOTrainer.forward_process函数 - Coupled Sampling方案的掩码生成"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 20
        self.mask_id = 50256  # 假设的mask token ID
        
    def test_forward_process_basic_output_structure(self):
        """测试forward_process的基本输出结构"""
        # 创建测试数据
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        # 创建一个简化的trainer实例用于测试
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # 调用forward_process
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        # 验证输出结构
        self.assertEqual(len(noisy_batch), 3, "应该返回3个版本的掩码序列")
        self.assertEqual(len(weights), 3, "应该返回3个权重值")
        
        # 验证每个版本的形状
        for i, noisy_seq in enumerate(noisy_batch):
            self.assertEqual(noisy_seq.shape, batch.shape, 
                           f"版本{i+1}的形状应该与输入batch相同")
        
        # 验证completion_mask的形状
        self.assertEqual(completion_mask.shape, batch.shape,
                        "completion_mask的形状应该与输入batch相同")
    
    def test_forward_process_version1_masks_all_completion(self):
        """测试Version 1掩码所有完成token"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version1 = noisy_batch[0]
        
        # 验证Version 1：所有完成token应该被掩码
        for i in range(self.batch_size):
            # 提示部分应该保持不变
            torch.testing.assert_close(version1[i, :prompt_length], batch[i, :prompt_length])
            
            # 完成部分应该全部被掩码
            self.assertTrue((version1[i, prompt_length:] == self.mask_id).all(),
                          "Version 1应该掩码所有完成token")
    
    def test_forward_process_version2_partial_masking(self):
        """测试Version 2按mask_ratio随机掩码"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version2 = noisy_batch[1]
        
        # 验证Version 2：部分完成token被掩码
        for i in range(self.batch_size):
            # 提示部分应该保持不变
            torch.testing.assert_close(version2[i, :prompt_length], batch[i, :prompt_length])
            
            # 完成部分应该有部分被掩码，部分保持原样
            completion_tokens = version2[i, prompt_length:]
            masked_count = (completion_tokens == self.mask_id).sum().item()
            unmasked_count = (completion_tokens != self.mask_id).sum().item()
            
            # 应该既有掩码的也有未掩码的token（在大多数情况下）
            self.assertGreater(masked_count, 0, "Version 2应该有掩码的token")
            self.assertGreater(unmasked_count, 0, "Version 2应该有未掩码的token")
    
    def test_forward_process_version3_reverse_masking(self):
        """测试Version 3反向掩码（与Version 2互补）"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version2 = noisy_batch[1]
        version3 = noisy_batch[2]
        
        # 验证Version 2和Version 3的互补性
        for i in range(self.batch_size):
            for j in range(prompt_length, self.seq_len):
                # 如果Version 2中某个位置被掩码，Version 3中应该不被掩码
                # 如果Version 2中某个位置不被掩码，Version 3中应该被掩码
                v2_masked = (version2[i, j] == self.mask_id)
                v3_masked = (version3[i, j] == self.mask_id)
                
                # 两者应该互补（不能同时为True或同时为False）
                self.assertNotEqual(v2_masked.item(), v3_masked.item(),
                                  f"位置{j}在Version 2和3中应该互补")
    
    def test_forward_process_weights_calculation(self):
        """测试权重计算的正确性"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        # 验证权重结构
        self.assertEqual(weights[0], 1, "Version 1的权重应该为1")
        
        # Version 2和3的权重应该是mask_ratio的倒数
        # weights[1] = 1/mask_ratio, weights[2] = 1/(1-mask_ratio)
        # 因此 1/weights[1] + 1/weights[2] 应该等于 1
        mask_ratio = 1.0 / weights[1]
        expected_weight3 = 1.0 / (1.0 - mask_ratio)
        
        self.assertAlmostEqual(weights[2], expected_weight3, places=5,
                              msg="Version 3的权重应该为1/(1-mask_ratio)")
        
        # 验证mask_ratio在[0.2, 0.8]范围内
        self.assertGreaterEqual(mask_ratio, 0.2, "mask_ratio应该>=0.2")
        self.assertLessEqual(mask_ratio, 0.8, "mask_ratio应该<=0.8")
    
    def test_forward_process_completion_mask_consistency(self):
        """测试completion_mask与Version 2的一致性"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        version2 = noisy_batch[1]
        
        # 验证completion_mask标记了Version 2中被掩码的位置
        for i in range(self.batch_size):
            for j in range(self.seq_len):
                is_masked_in_v2 = (version2[i, j] == self.mask_id)
                is_marked_in_mask = completion_mask[i, j].item()
                
                if j < prompt_length:
                    # 提示部分不应该被标记
                    self.assertFalse(is_marked_in_mask,
                                   f"提示位置{j}不应该在completion_mask中被标记")
                else:
                    # 完成部分：completion_mask应该与Version 2的掩码状态一致
                    self.assertEqual(is_masked_in_v2, is_marked_in_mask,
                                   f"位置{j}的completion_mask应该与Version 2一致")
    
    def test_forward_process_seed_reproducibility(self):
        """测试随机种子的可重复性"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # 使用相同种子生成两次
        noisy_batch1, weights1, mask1 = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=123
        )
        noisy_batch2, weights2, mask2 = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=123
        )
        
        # 验证结果完全相同
        for i in range(3):
            torch.testing.assert_close(noisy_batch1[i], noisy_batch2[i],
                                      msg=f"使用相同种子，Version {i+1}应该相同")
        
        self.assertEqual(weights1, weights2, "使用相同种子，权重应该相同")
        torch.testing.assert_close(mask1, mask2, msg="使用相同种子，completion_mask应该相同")
    
    def test_forward_process_prompt_preservation(self):
        """测试所有版本都保留提示部分"""
        batch = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        prompt_length = 10
        prompt_index = torch.zeros(self.seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, self.mask_id, seed=42
        )
        
        # 验证所有版本的提示部分都保持不变
        for version_idx, noisy_seq in enumerate(noisy_batch):
            for i in range(self.batch_size):
                torch.testing.assert_close(
                    noisy_seq[i, :prompt_length],
                    batch[i, :prompt_length],
                    msg=f"Version {version_idx+1}应该保留提示部分"
                )
    
    def test_forward_process_edge_cases(self):
        """测试边缘情况"""
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # 测试1: 只有提示，没有完成
        batch_prompt_only = torch.randint(0, 1000, (2, 10))
        prompt_index_all = torch.ones(10, dtype=torch.bool)
        
        noisy_batch, weights, mask = trainer.forward_process(
            batch_prompt_only, prompt_index_all, self.mask_id, seed=42
        )
        
        # 所有版本应该与原始batch相同（因为没有完成token需要掩码）
        for version in noisy_batch:
            torch.testing.assert_close(version, batch_prompt_only)
        
        # 测试2: 只有完成，没有提示
        batch_completion_only = torch.randint(0, 1000, (2, 10))
        prompt_index_none = torch.zeros(10, dtype=torch.bool)
        
        noisy_batch, weights, mask = trainer.forward_process(
            batch_completion_only, prompt_index_none, self.mask_id, seed=42
        )
        
        # Version 1应该全部被掩码
        self.assertTrue((noisy_batch[0] == self.mask_id).all(),
                       "没有提示时，Version 1应该全部掩码")
        
        # 测试3: 单个token
        batch_single = torch.randint(0, 1000, (1, 1))
        prompt_index_single = torch.zeros(1, dtype=torch.bool)
        
        noisy_batch, weights, mask = trainer.forward_process(
            batch_single, prompt_index_single, self.mask_id, seed=42
        )
        
        self.assertEqual(len(noisy_batch), 3)
        self.assertEqual(noisy_batch[0].shape, (1, 1))


class TestForwardProcessIntegration(unittest.TestCase):
    """集成测试：测试forward_process在实际训练流程中的表现"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_coverage_property(self):
        """测试Coupled Sampling的覆盖性：每个token至少在一个版本中被计算"""
        batch_size, seq_len = 3, 15
        prompt_length = 8
        mask_id = 50256
        
        batch = torch.randint(0, 1000, (batch_size, seq_len))
        prompt_index = torch.zeros(seq_len, dtype=torch.bool)
        prompt_index[:prompt_length] = True
        
        trainer = object.__new__(DiffuGRPOTrainer)
        noisy_batch, weights, completion_mask = trainer.forward_process(
            batch, prompt_index, mask_id, seed=42
        )
        
        # 对于每个完成token，检查它在至少一个版本中未被掩码
        for i in range(batch_size):
            for j in range(prompt_length, seq_len):
                # 检查这个token在三个版本中的状态
                v1_unmasked = (noisy_batch[0][i, j] != mask_id)
                v2_unmasked = (noisy_batch[1][i, j] != mask_id)
                v3_unmasked = (noisy_batch[2][i, j] != mask_id)
                
                # 至少有一个版本应该未掩码这个token
                # 注意：Version 1总是掩码所有完成token，所以主要看Version 2和3
                self.assertTrue(v2_unmasked or v3_unmasked,
                              f"Token at position {j} should be unmasked in at least one version")


if __name__ == "__main__":
    unittest.main()
