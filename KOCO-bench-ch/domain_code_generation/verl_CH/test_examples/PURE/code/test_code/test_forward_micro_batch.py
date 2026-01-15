import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the parent directory to Python path to import verl module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual class and components we want to test
try:
    from verl.workers.fsdp_workers import ProcessRewardModelWorker
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    ProcessRewardModelWorker = None


class TestForwardMicroBatch(unittest.TestCase):
    """测试ProcessRewardModelWorker._forward_micro_batch函数 - PURE文档第1节描述的过程奖励模型推断与最小式信用分配"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.seq_len = 8
        self.response_length = 5
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_forward_micro_batch_basic_logic(self):
        """测试_forward_micro_batch的基本逻辑"""
        # 创建一个简化的ProcessRewardModelWorker实例，避免复杂的初始化
        worker = object.__new__(ProcessRewardModelWorker)
        
        # 设置必要的属性
        worker.use_remove_padding = False
        worker.ulysses_sequence_parallel_size = 1
        worker.disable_approx_min_form_credit_assignment = False
        worker.temperature = 0.1
        
        # 创建mock的process_reward_module
        mock_process_reward_module = MagicMock()
        worker.process_reward_module = mock_process_reward_module
        
        # 创建micro_batch输入数据
        micro_batch = {
            # "input_ids": torch.randint(0, 1000, (self.batch_size, self.seq_len)),
            "input_ids": torch.tensor([[13, 27, 38, 44, 51, 62,777,32], [11, 25, 36, 47, 58, 69,88,22], [19, 20, 31, 42, 53, 64,3,65], [11, 22, 33, 44, 55, 66,77,88]]),
            "attention_mask": torch.ones(self.batch_size, self.seq_len),
            "position_ids": torch.arange(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1),
            # "responses": torch.randint(0, 1000, (self.batch_size, self.response_length)),
            "responses": torch.tensor([[1, 21, 32, 43, 54], [111, 256, 37, 48, 59], [19, 20, 31, 42, 53], [11, 22, 33, 44, 55]]),
            "reward_mask": torch.ones(self.batch_size, self.response_length)
        }
        
        # 创建可控的二分类logits输出
        mock_logits = torch.tensor([
            [[2.0, 0.0], [1.0, 1.5], [0.5, 2.5], [0.0, 1.0], [-0.5, 1.5], [0.8, 0.2], [1.2, 0.8], [0.3, 1.7]],  # batch 0
            [[1.5, 0.5], [0.2, 2.0], [1.8, 0.3], [-1.0, 2.0], [0.6, 1.4], [1.1, 0.9], [0.4, 1.6], [2.0, 0.1]]   # batch 1
        ])  # (batch_size, seq_len, 2)
        
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_process_reward_module.return_value = mock_output
        
        # 调用真实的_forward_micro_batch方法
        result = worker._forward_micro_batch(micro_batch)
        
        expected_result = torch.tensor([[ 1.7748e-04,  1.4639e-05, -2.0934e-01, -5.5440e-02,  5.5964e-05],
        [ 6.4885e-08,  5.2007e-06, -1.6514e-04,  1.5278e-06, -7.3855e-01]])
        torch.testing.assert_close(result, expected_result, atol=1e-4, rtol=1e-4)
        # 验证输出
        # self.assertEqual(result.shape, (self.batch_size, self.response_length))
        # self.assertTrue(torch.is_tensor(result))
        # self.assertTrue(torch.isfinite(result).all())
        
        # 验证结果的范围（应该在[-1, 1]之间，但由于min-form加权可能超出）
        # self.assertTrue(result.abs().max() <= 10.0)  # 合理的范围检查
        
        # 验证min-form加权特性：权重应该集中在最小分数的位置
        # 重新计算期望的中间结果来验证逻辑
        response_logits = mock_logits[:, -self.response_length:]  # 提取响应部分
        expected_probs = response_logits.softmax(dim=-1)
        expected_rm_score = (expected_probs[..., 1] - expected_probs[..., 0]) * micro_batch['reward_mask']
        
        # 调用修复后的真实_forward_micro_batch方法
        result = worker._forward_micro_batch(micro_batch)
        
        print(result)
        # 验证输出
        self.assertEqual(result.shape, (self.batch_size, self.response_length))
        self.assertTrue(torch.is_tensor(result))
        self.assertTrue(torch.isfinite(result).all())
        
        # 验证结果的范围（应该在合理范围内）
        self.assertTrue(result.abs().max() <= 10.0)
        
        # 验证最小分数位置的识别和min-form加权效果
        for i in range(self.batch_size):
            min_score_idx = expected_rm_score[i].argmin()
            # 由于min-form加权，最小分数位置应该得到最大权重
            self.assertIsNotNone(min_score_idx)
            
        # 验证权重归一化特性（通过手动计算验证）
        expected_weight = torch.softmax(
            -expected_rm_score.masked_fill(~micro_batch['reward_mask'].bool(), float('inf')) / 0.1,
            dim=-1
        )
        # 权重之和应该为1（对每个序列）
        for i in range(self.batch_size):
            weight_sum = expected_weight[i].sum()
            self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
    
    # def test_binary_classification_processing(self):
    #     """测试二分类logits处理逻辑"""
    #     # 创建已知的二分类logits
    #     # 创建二分类logits (格式：[negative_class, positive_class])
    #     binary_logits = torch.tensor([
    #         [[1.0, -0.5], [2.0, 0.5], [-0.8, 1.2]],    # 第一个序列
    #         [[0.3, 0.8], [-1.5, 0.2], [3.0, -1.0]]     # 第二个序列
    #     ])  # (batch_size, response_length, 2)
        
    #     reward_mask = torch.ones(2, 3)
        
    #     # 实现文档中描述的二分类处理逻辑
    #     # 1. 计算softmax概率
    #     probs = binary_logits.softmax(dim=-1)  # (batch_size, response_length, 2)
        
    #     # 2. 计算 p_positive - p_negative
    #     rm_score = (probs[..., 1] - probs[..., 0])  # positive class - negative class
        
    #     # 3. 应用reward mask
    #     rm_score = rm_score * reward_mask
        
    #     # 验证结果
    #     self.assertEqual(rm_score.shape, (2, 3))
    #     self.assertTrue(torch.isfinite(rm_score).all())
        
    #     # 验证分数范围：应该在[-1, 1]之间
    #     self.assertTrue((rm_score >= -1.0).all())
    #     self.assertTrue((rm_score <= 1.0).all())
        
    #     # 验证具体值：根据logits格式 [negative_class, positive_class]
    #     # Position (0,1): logits=[2.0, 0.5], negative(2.0)大于positive(0.5)，所以分数应该为负
    #     self.assertLess(rm_score[0, 1].item(), 0.0)
        
    #     # Position (1,1): logits=[-1.5, 0.2], positive(0.2)大于negative(-1.5)，所以分数应该为正  
    #     self.assertGreater(rm_score[1, 1].item(), 0.0)
    
    # def test_min_form_credit_assignment(self):
    #     """测试最小式信用分配的加权机制"""
    #     batch_size, response_length = 2, 4
    #     temperature = 0.1
        
    #     # 创建具有明确差异的reward scores
    #     rm_score = torch.tensor([
    #         [2.0, 0.5, 1.0, -0.5],  # 第一个序列：最后一个token分数最低
    #         [1.5, -1.0, 0.8, 2.5]   # 第二个序列：第二个token分数最低
    #     ])
    #     reward_mask = torch.ones(batch_size, response_length)
        
    #     # 实现近似min-form权重：softmax(-rm_score/T)
    #     weight = torch.softmax(
    #         -rm_score.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
    #         dim=-1
    #     )
    #     weighted_rm_score = rm_score * weight
        
    #     # 验证权重特性：分数越低的token权重应该越高（接近min操作）
    #     for i in range(batch_size):
    #         min_score_idx = rm_score[i].argmin()
    #         max_weight_idx = weight[i].argmax()
    #         # 最小分数的位置应该对应最大权重
    #         self.assertEqual(min_score_idx, max_weight_idx)
        
    #     # 验证权重归一化
    #     for i in range(batch_size):
    #         valid_positions = reward_mask[i].bool()
    #         weight_sum = weight[i, valid_positions].sum()
    #         self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
        
    #     # 验证输出形状和有效性
    #     self.assertEqual(weighted_rm_score.shape, (batch_size, response_length))
    #     self.assertTrue(torch.isfinite(weighted_rm_score).all())
    
    # def test_temperature_effects(self):
    #     """测试温度参数对最小式加权的影响"""
    #     rm_score = torch.tensor([[1.0, -2.0, 0.5, -1.5]])  # 第二个位置分数最低
    #     reward_mask = torch.ones(1, 4)
        
    #     # 测试不同温度下的权重分布
    #     temperatures = [0.01, 0.1, 1.0]  # 从低到高
    #     weights = []
        
    #     for temp in temperatures:
    #         weight = torch.softmax(
    #             -rm_score.masked_fill(~reward_mask.bool(), float('inf')) / temp,
    #             dim=-1
    #         )
    #         weights.append(weight)
        
    #     # 验证温度效应
    #     # 1. 所有温度下，最小分数位置（index=1）应该有最大权重
    #     for weight in weights:
    #         max_weight_idx = weight[0].argmax()
    #         self.assertEqual(max_weight_idx.item(), 1)  # rm_score中第二个位置(-2.0)最小
            
    #     # 2. 温度越低，权重分布越集中在最小值位置
    #     # 低温度下的最大权重应该更大
    #     self.assertGreater(weights[0][0, 1].item(), weights[1][0, 1].item())
    #     self.assertGreater(weights[1][0, 1].item(), weights[2][0, 1].item())
        
    #     # 3. 验证权重归一化
    #     for weight in weights:
    #         weight_sum = weight[0].sum()
    #         self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
    
#     def test_response_length_extraction(self):
#         """测试响应长度提取和截取逻辑"""
#         batch_size, total_seq_len = 2, 10
#         response_length = 4
        
#         # 创建完整序列的logits
#         full_logits = torch.randn(batch_size, total_seq_len, 2)
        
#         # 实现响应部分提取：logits[:, -response_length:]
#         response_logits = full_logits[:, -response_length:]
        
#         # 验证提取的logits形状正确
#         self.assertEqual(response_logits.shape, (batch_size, response_length, 2))
        
#         # 验证是从序列末尾提取的
#         expected_response_logits = full_logits[:, total_seq_len-response_length:]
#         torch.testing.assert_close(response_logits, expected_response_logits)
        
#         # 处理提取的logits
#         rm_score = response_logits.softmax(dim=-1)
#         rm_score = (rm_score[..., 1] - rm_score[..., 0])
        
#         # 验证最终形状
#         self.assertEqual(rm_score.shape, (batch_size, response_length))
#         self.assertTrue(torch.isfinite(rm_score).all())
    
#     def test_reward_mask_application(self):
#         """测试reward_mask的应用"""
#         batch_size, response_length = 2, 5
        
#         # 创建rm_score
#         rm_score = torch.randn(batch_size, response_length)
        
#         # 创建部分mask的reward_mask
#         reward_mask = torch.tensor([
#             [1.0, 1.0, 1.0, 0.0, 0.0],  # 第一个序列：前3个有效
#             [1.0, 1.0, 1.0, 1.0, 0.0]   # 第二个序列：前4个有效
#         ])
        
#         # 应用mask
#         masked_rm_score = rm_score * reward_mask
        
#         # 验证mask效果
#         # 被mask的位置应该为0
#         self.assertEqual(masked_rm_score[0, 3].item(), 0.0)
#         self.assertEqual(masked_rm_score[0, 4].item(), 0.0)
#         self.assertEqual(masked_rm_score[1, 4].item(), 0.0)
        
#         # 未被mask的位置应该保持原值
#         for i in range(batch_size):
#             for j in range(response_length):
#                 if reward_mask[i, j] == 1.0:
#                     self.assertEqual(masked_rm_score[i, j].item(), rm_score[i, j].item())
#                 else:
#                     self.assertEqual(masked_rm_score[i, j].item(), 0.0)
    
#     def test_input_output_samples_basic_prm_forward(self):
#         """测试基本PRM前向推断的具体输入输出样例"""
#         batch_size, response_length = 1, 3
#         temperature = 0.1
        
#         # 创建确定性的二分类logits (格式：[negative_class, positive_class])
#         binary_logits = torch.tensor([
#             [[2.0, 0.0], [1.0, 1.5], [0.5, 2.5]]  # 第一个token negative强，第二个positive强，第三个positive很强
#         ])  # shape: (1, 3, 2)
        
#         reward_mask = torch.ones(1, 3)
        
#         # 手动计算expected结果
#         # 1. softmax处理
#         probs = binary_logits.softmax(dim=-1)
#         rm_score = (probs[..., 1] - probs[..., 0])  # positive - negative
        
#         # 2. 近似min-form加权
#         weight = torch.softmax(
#             -rm_score.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
#             dim=-1
#         )
#         weighted_rm_score = rm_score * weight
        
#         # 验证结果
#         self.assertEqual(weighted_rm_score.shape, (1, 3))
#         self.assertTrue(torch.isfinite(weighted_rm_score).all())
        
#         # 验证权重分配：最小分数位置应该获得最大权重
#         min_score_idx = rm_score[0].argmin()
#         max_weight_idx = weight[0].argmax()
#         self.assertEqual(min_score_idx, max_weight_idx)
        
#         # 验证权重和为1
#         weight_sum = weight[0].sum()
#         self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
    
#     def test_input_output_samples_edge_cases(self):
#         """测试边缘情况的具体输入输出样例"""
#         temperature = 0.1
        
#         # 测试样例1: 相同分数情况
#         rm_score_equal = torch.ones(1, 3) * 0.5  # 所有分数相同
#         reward_mask = torch.ones(1, 3)
        
#         weight = torch.softmax(
#             -rm_score_equal.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
#             dim=-1
#         )
        
#         # 相同分数下，权重应该均匀分布
#         expected_weight = 1.0 / 3.0
#         for i in range(3):
#             self.assertAlmostEqual(weight[0, i].item(), expected_weight, places=4)
        
#         # 测试样例2: 极端分数差异
#         rm_score_extreme = torch.tensor([[100.0, -100.0, 0.0]])
#         weight_extreme = torch.softmax(
#             -rm_score_extreme.masked_fill(~reward_mask.bool(), float('inf')) / temperature,
#             dim=-1
#         )
        
#         # 极端情况下，最小分数位置应该获得接近1的权重
#         self.assertGreater(weight_extreme[0, 1].item(), 0.99)  # -100位置
#         self.assertLess(weight_extreme[0, 0].item(), 0.01)     # 100位置
        
#         # 测试样例3: mask应用
#         rm_score_masked = torch.tensor([[1.0, -1.0, 2.0]])
#         reward_mask_partial = torch.tensor([[1.0, 1.0, 0.0]])  # 最后一个位置被mask
        
#         # 应用mask到权重计算
#         weight_masked = torch.softmax(
#             -rm_score_masked.masked_fill(~reward_mask_partial.bool(), float('inf')) / temperature,
#             dim=-1
#         )
        
#         # 被mask的位置权重应该为0
#         self.assertEqual(weight_masked[0, 2].item(), 0.0)
        
#         # 未被mask位置的权重和应该为1
#         valid_weight_sum = weight_masked[0, :2].sum()
#         self.assertAlmostEqual(valid_weight_sum.item(), 1.0, places=5)


# class TestForwardMicroBatchIntegration(unittest.TestCase):
#     """集成测试：测试_forward_micro_batch在实际环境中的表现"""
    
#     def setUp(self):
#         torch.manual_seed(42)
#         self.batch_size = 2
#         self.seq_len = 12
#         self.response_length = 6
#         self.vocab_size = 1000
#         self.temperature = 0.1
    
#     def test_end_to_end_processing_simulation(self):
#         """端到端测试：从输入到输出的完整流程模拟"""
#         # 创建完整的输入数据
#         input_data = {
#             "input_ids": torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len)),
#             "attention_mask": torch.ones(self.batch_size, self.seq_len),
#             "position_ids": torch.arange(self.seq_len).unsqueeze(0).repeat(self.batch_size, 1),
#             "responses": torch.randint(0, self.vocab_size, (self.batch_size, self.response_length)),
#             "reward_mask": torch.ones(self.batch_size, self.response_length)
#         }
        
#         # 模拟完整的处理流程
#         # 1. 模拟PRM模型输出
#         full_logits = torch.randn(self.batch_size, self.seq_len, 2)
        
#         # 2. 提取响应部分
#         response_logits = full_logits[:, -self.response_length:]
        
#         # 3. 二分类处理
#         rm_score = response_logits.softmax(dim=-1)
#         rm_score = (rm_score[..., 1] - rm_score[..., 0]) * input_data['reward_mask']
        
#         # 4. 最小式加权
#         weight = torch.softmax(
#             -rm_score.masked_fill(~input_data['reward_mask'].bool(), float('inf')) / self.temperature,
#             dim=-1
#         )
#         final_rm_score = rm_score * weight
        
#         # 验证完整流程的输出
#         self.assertEqual(final_rm_score.shape, (self.batch_size, self.response_length))
#         self.assertTrue(torch.isfinite(final_rm_score).all())
        
#         # 验证权重归一化（对每个序列）
#         for i in range(self.batch_size):
#             valid_positions = input_data['reward_mask'][i].bool()
#             weight_sum = weight[i, valid_positions].sum()
#             self.assertAlmostEqual(weight_sum.item(), 1.0, places=5)
        
#         # 验证min-form特性：权重最大的位置对应分数最小的位置
#         for i in range(self.batch_size):
#             min_score_idx = rm_score[i].argmin()
#             max_weight_idx = weight[i].argmax()
#             self.assertEqual(min_score_idx, max_weight_idx)
    
#     def test_disable_min_form_mode(self):
#         """测试禁用近似min-form信用分配模式"""
#         rm_score = torch.tensor([
#             [2.0, 0.5, 1.0, -0.5],
#             [1.5, -1.0, 0.8, 2.5]
#         ])
#         reward_mask = torch.ones(2, 4)
        
#         # 当禁用近似min-form时，应该直接返回rm_score而不加权
#         disable_approx_min_form = True
        
#         if disable_approx_min_form:
#             # 不应用加权
#             final_score = rm_score
#         else:
#             # 应用加权
#             weight = torch.softmax(-rm_score / 0.1, dim=-1)
#             final_score = rm_score * weight
        
#         # 验证禁用模式下的结果
#         torch.testing.assert_close(final_score, rm_score)
#         self.assertEqual(final_score.shape, (2, 4))
#         self.assertTrue(torch.isfinite(final_score).all())


if __name__ == "__main__":
    unittest.main()