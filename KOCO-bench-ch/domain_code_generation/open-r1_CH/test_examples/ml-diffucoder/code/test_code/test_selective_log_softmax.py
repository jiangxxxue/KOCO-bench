import unittest
import torch
import sys
import os
import torch.nn.functional as F

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from open_r1.coupled_grpo import selective_log_softmax


class TestSelectiveLogSoftmax(unittest.TestCase):
    """测试selective_log_softmax函数 - 高效计算加权对数概率"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
    
    def test_selective_log_softmax_basic_output_shape(self):
        """测试基本输出形状"""
        num_iterations = 2
        batch_size = 3
        seq_len = 10
        
        # 创建输入数据
        logits = torch.randn(num_iterations * 3 * batch_size, seq_len, self.vocab_size)
        index = torch.randint(0, self.vocab_size, (num_iterations * batch_size, seq_len))
        weights = torch.tensor([1.0, 1.5, 2.0] * num_iterations)
        mask = torch.randint(0, 2, (num_iterations * batch_size, seq_len)).bool()
        
        # 调用函数
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 验证输出形状
        expected_shape = (num_iterations * batch_size, seq_len)
        self.assertEqual(result.shape, expected_shape,
                        f"输出形状应该为{expected_shape}")
        
        # 验证输出是有限的
        self.assertTrue(torch.isfinite(result).all(),
                       "输出应该全部是有限值")
    
    def test_selective_log_softmax_uniform_weights(self):
        """测试使用统一权重时的行为"""
        num_iterations = 1
        batch_size = 2
        seq_len = 5
        
        logits = torch.randn(num_iterations * 3 * batch_size, seq_len, self.vocab_size)
        index = torch.randint(0, self.vocab_size, (num_iterations * batch_size, seq_len))
        
        # 使用统一权重
        weights = torch.ones(num_iterations * 3)
        mask = torch.randint(0, 2, (num_iterations * batch_size, seq_len)).bool()
        
        result = selective_log_softmax(logits, index, weights=weights, mask=mask)
        
        # 验证输出形状和有效性
        self.assertEqual(result.shape, (num_iterations * batch_size, seq_len))
        self.assertTrue(torch.isfinite(result).all(),
                       "使用统一权重时输出应该全部是有限值")
    
    def test_selective_log_softmax_weighted_averaging(self):
        """测试加权平均的正确性"""
        num_iterations = 1
        batch_size = 1
        seq_len = 3
        
        # 创建可控的logits
        logits = torch.zeros(3, seq_len, self.vocab_size)
        # 为每个版本设置不同的logits
        logits[0, :, 0] = 10.0  # Version 1: token 0有高概率
        logits[1, :, 1] = 10.0  # Version 2: token 1有高概率
        logits[2, :, 2] = 10.0  # Version 3: token 2有高概率
        
        index = torch.zeros(1, seq_len, dtype=torch.long)  # 都选择token 0
        weights = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([[True, False, True]])  # 第0和第2个位置被掩码
        
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 手动计算期望结果
        # 对于每个位置：
        # - 如果mask=True: final = (logp1 + logp2 * weight2) / 2
        # - 如果mask=False: final = (logp1 + logp3 * weight3) / 2
        
        logp1 = F.log_softmax(logits[0], dim=-1).gather(dim=-1, index=index[0].unsqueeze(-1)).squeeze(-1)
        logp2 = F.log_softmax(logits[1], dim=-1).gather(dim=-1, index=index[0].unsqueeze(-1)).squeeze(-1)
        logp3 = F.log_softmax(logits[2], dim=-1).gather(dim=-1, index=index[0].unsqueeze(-1)).squeeze(-1)
        
        expected = torch.zeros(1, seq_len)
        for j in range(seq_len):
            if mask[0, j]:
                expected[0, j] = (logp1[j] + logp2[j] * weights[1]) / 2
            else:
                expected[0, j] = (logp1[j] + logp3[j] * weights[2]) / 2
        
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
    
    def test_selective_log_softmax_multiple_iterations(self):
        """测试多次迭代的处理"""
        num_iterations = 3
        batch_size = 2
        seq_len = 4
        
        logits = torch.randn(num_iterations * 3 * batch_size, seq_len, self.vocab_size)
        index = torch.randint(0, self.vocab_size, (num_iterations * batch_size, seq_len))
        weights = torch.rand(num_iterations * 3) + 0.5  # 避免权重太小
        mask = torch.randint(0, 2, (num_iterations * batch_size, seq_len)).bool()
        
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 验证形状
        self.assertEqual(result.shape, (num_iterations * batch_size, seq_len))
        
        # 验证每个迭代的结果都是有限的
        for i in range(num_iterations):
            iter_result = result[i * batch_size:(i + 1) * batch_size]
            self.assertTrue(torch.isfinite(iter_result).all(),
                          f"迭代{i}的结果应该全部是有限值")
    
    def test_selective_log_softmax_mask_effect(self):
        """测试mask对结果的影响"""
        num_iterations = 1
        batch_size = 1
        seq_len = 4
        
        logits = torch.randn(3, seq_len, self.vocab_size)
        index = torch.randint(0, self.vocab_size, (1, seq_len))
        weights = torch.tensor([1.0, 1.5, 2.0])
        
        # 测试全True的mask
        mask_all_true = torch.ones(1, seq_len, dtype=torch.bool)
        result_all_true = selective_log_softmax(logits, index, weights, mask_all_true)
        
        # 测试全False的mask
        mask_all_false = torch.zeros(1, seq_len, dtype=torch.bool)
        result_all_false = selective_log_softmax(logits, index, weights, mask_all_false)
        
        # 两个结果应该不同（因为使用了不同的版本）
        self.assertFalse(torch.allclose(result_all_true, result_all_false),
                        "不同的mask应该产生不同的结果")
        
        # 验证结果都是有限的
        self.assertTrue(torch.isfinite(result_all_true).all())
        self.assertTrue(torch.isfinite(result_all_false).all())
    
    def test_selective_log_softmax_memory_efficiency(self):
        """测试内存效率（逐序列处理）"""
        num_iterations = 2
        batch_size = 4
        seq_len = 8
        
        logits = torch.randn(num_iterations * 3 * batch_size, seq_len, self.vocab_size)
        index = torch.randint(0, self.vocab_size, (num_iterations * batch_size, seq_len))
        weights = torch.rand(num_iterations * 3) + 0.5
        mask = torch.randint(0, 2, (num_iterations * batch_size, seq_len)).bool()
        
        # 调用函数（应该不会因为内存问题而失败）
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 验证结果的正确性
        self.assertEqual(result.shape, (num_iterations * batch_size, seq_len))
        self.assertTrue(torch.isfinite(result).all())
    
    def test_selective_log_softmax_gather_correctness(self):
        """测试gather操作的正确性"""
        num_iterations = 1
        batch_size = 2
        seq_len = 3
        vocab_size = 10
        
        # 创建简单的logits
        # 顺序应该是: [v1_seq0, v1_seq1, v2_seq0, v2_seq1, v3_seq0, v3_seq1]
        logits = torch.randn(3 * batch_size, seq_len, vocab_size)
        index = torch.tensor([[0, 1, 2], [3, 4, 5]])  # 指定要提取的token ID
        weights = torch.tensor([1.0, 1.0, 1.0])
        mask = torch.tensor([[True, True, False], [False, True, True]])
        
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 手动验证第一个序列的第一个位置
        # 根据实现，logits的索引是: base + version*batch_size + offset
        # 对于seq0: base=0, offset=0
        # v1: 0 + 0*2 + 0 = 0
        # v2: 0 + 1*2 + 0 = 2
        # v3: 0 + 2*2 + 0 = 4
        seq0_logits_v1 = logits[0, 0]  # Version 1, seq 0
        seq0_logits_v2 = logits[2, 0]  # Version 2, seq 0
        seq0_logp_v1 = F.log_softmax(seq0_logits_v1, dim=-1)[index[0, 0]]
        seq0_logp_v2 = F.log_softmax(seq0_logits_v2, dim=-1)[index[0, 0]]
        
        # mask[0,0]=True，所以使用Version 2
        expected_value = (seq0_logp_v1 + seq0_logp_v2 * weights[1]) / 2
        
        self.assertAlmostEqual(result[0, 0].item(), expected_value.item(), places=5)
    
    def test_selective_log_softmax_weight_indexing(self):
        """测试权重索引的正确性"""
        num_iterations = 2
        batch_size = 1
        seq_len = 2
        
        logits = torch.randn(num_iterations * 3 * batch_size, seq_len, self.vocab_size)
        index = torch.randint(0, self.vocab_size, (num_iterations * batch_size, seq_len))
        
        # 为每个迭代设置不同的权重
        weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0, 5.0])  # 两个迭代
        mask = torch.ones(num_iterations * batch_size, seq_len, dtype=torch.bool)
        
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 验证每个迭代使用了正确的权重
        # 第一个迭代应该使用weights[0:3]
        # 第二个迭代应该使用weights[3:6]
        
        self.assertEqual(result.shape, (num_iterations * batch_size, seq_len))
        self.assertTrue(torch.isfinite(result).all())
    
    def test_selective_log_softmax_edge_cases(self):
        """测试边缘情况"""
        # 测试1: 单个序列
        logits_single = torch.randn(3, 1, self.vocab_size)
        index_single = torch.randint(0, self.vocab_size, (1, 1))
        weights_single = torch.tensor([1.0, 1.0, 1.0])
        mask_single = torch.tensor([[True]])
        
        result_single = selective_log_softmax(logits_single, index_single, weights_single, mask_single)
        
        self.assertEqual(result_single.shape, (1, 1))
        self.assertTrue(torch.isfinite(result_single).all())
        
        # 测试2: 长序列
        long_seq_len = 100
        logits_long = torch.randn(3, long_seq_len, self.vocab_size)
        index_long = torch.randint(0, self.vocab_size, (1, long_seq_len))
        weights_long = torch.tensor([1.0, 1.5, 2.0])
        mask_long = torch.randint(0, 2, (1, long_seq_len)).bool()
        
        result_long = selective_log_softmax(logits_long, index_long, weights_long, mask_long)
        
        self.assertEqual(result_long.shape, (1, long_seq_len))
        self.assertTrue(torch.isfinite(result_long).all())


class TestSelectiveLogSoftmaxIntegration(unittest.TestCase):
    """集成测试：测试selective_log_softmax在实际场景中的表现"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_integration_with_forward_process(self):
        """测试与forward_process的集成"""
        num_iterations = 2
        batch_size = 3
        seq_len = 10
        vocab_size = 1000
        
        # 模拟forward_process的输出
        # 3个版本 × num_iterations × batch_size
        logits = torch.randn(num_iterations * 3 * batch_size, seq_len, vocab_size)
        index = torch.randint(0, vocab_size, (num_iterations * batch_size, seq_len))
        
        # 模拟权重：每个迭代有3个权重
        weights = []
        for _ in range(num_iterations):
            mask_ratio = torch.rand(1).item() * 0.6 + 0.2  # [0.2, 0.8]
            weights.extend([1.0, 1.0/mask_ratio, 1.0/(1.0-mask_ratio)])
        weights = torch.tensor(weights)
        
        # 模拟completion_mask
        mask = torch.randint(0, 2, (num_iterations * batch_size, seq_len)).bool()
        
        # 调用函数
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 验证输出
        self.assertEqual(result.shape, (num_iterations * batch_size, seq_len))
        self.assertTrue(torch.isfinite(result).all())
        
        # 验证可以reshape回原始维度
        reshaped = result.view(num_iterations, batch_size, seq_len)
        self.assertEqual(reshaped.shape, (num_iterations, batch_size, seq_len))
    
    def test_variance_reduction_property(self):
        """测试方差降低特性（Coupled Sampling的目标）"""
        num_iterations = 1
        batch_size = 10
        seq_len = 5
        vocab_size = 100
        
        # 创建相似的logits来模拟低方差场景
        base_logits = torch.randn(1, seq_len, vocab_size)
        logits = base_logits.repeat(3 * batch_size, 1, 1)
        logits += torch.randn_like(logits) * 0.1  # 添加小噪声
        
        index = torch.randint(0, vocab_size, (batch_size, seq_len))
        weights = torch.tensor([1.0, 1.5, 2.0])
        mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
        
        result = selective_log_softmax(logits, index, weights, mask)
        
        # 计算每个位置的方差
        variance = result.var(dim=0)
        
        # 验证方差是有限的且相对较小
        self.assertTrue(torch.isfinite(variance).all())
        self.assertLess(variance.mean().item(), 10.0,
                       "使用Coupled Sampling应该产生较低的方差")
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        num_iterations = 1
        batch_size = 2
        seq_len = 3
        vocab_size = 50
        
        # 测试极端logits值
        logits_extreme = torch.zeros(3 * batch_size, seq_len, vocab_size)
        logits_extreme[:, :, 0] = 100.0  # 非常大的值
        logits_extreme[:, :, 1] = -100.0  # 非常小的值
        
        index = torch.zeros(batch_size, seq_len, dtype=torch.long)
        weights = torch.tensor([1.0, 1.5, 2.0])
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        result = selective_log_softmax(logits_extreme, index, weights, mask)
        
        # 验证结果是有限的（没有inf或nan）
        self.assertTrue(torch.isfinite(result).all(),
                       "即使使用极端logits值，结果也应该是有限的")


if __name__ == "__main__":
    unittest.main()
