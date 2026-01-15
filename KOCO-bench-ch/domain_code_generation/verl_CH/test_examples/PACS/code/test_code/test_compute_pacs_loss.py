import unittest
import torch
import sys
import os
from unittest.mock import MagicMock

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual function we want to test
from pacs.pacs_core_algos import compute_pacs_loss


class TestComputePacsLoss(unittest.TestCase):
    """测试compute_pacs_loss函数 - PACS文档第3节描述的核心损失函数"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_compute_pacs_loss_basic_rloo(self):
        """测试使用RLOO优势估计的基本PACS损失计算"""
        batch_size = 4
        rollout_n = 2
        prompt_length = 10
        response_length = 5
        
        # 创建输入数据
        # 确保同一prompt的n个响应在同一批次中连续排列
        prompts = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]])
        # 让前2个样本有相同的prompt，后2个样本有相同的prompt
        # prompts[2:] = torch.randint(0, 1000, (prompt_length,))

        # print(prompts)
        
        old_log_prob = torch.tensor([[-0.7581,  1.0783,  0.8008,  1.6806,  0.0349],
        [ 0.3211,  1.5736, -0.8455,  1.3123,  0.6872],
        [-1.0892, -0.3553, -1.4181,  0.8963,  0.0499],
        [ 2.2667,  1.1790, -0.4345, -1.3864, -1.2862]])
        log_prob = torch.tensor([[-0.8371, -0.9224,  1.8113,  0.1606,  0.1971],
        [-1.1441,  0.3383,  1.6992,  0.0109, -0.3387],
        [-1.3407, -0.5854, -0.5644,  1.0563, -1.4692],
        [ 1.4332,  0.7440, -0.4816, -1.0495,  0.6039]])
        token_level_scores = torch.tensor([[1., 1., 1., 1., 1.],
        [1., 1., 0., 0., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 0., 1., 0.]])
        response_mask = torch.ones(batch_size, response_length)
        
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        
        # 创建算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "rloo"
        algorithm_config.use_weight = False
        
        # 调用真实的compute_pacs_loss函数
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )

        # print(loss)

        expected_loss = torch.tensor(1.8122)
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
        # 验证输出
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)  # 标量
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)  # BCE损失应该非负
    
    def test_compute_pacs_loss_with_grpo(self):
        """测试使用GRPO优势估计的PACS损失计算"""
        batch_size = 6
        rollout_n = 3
        prompt_length = 8
        response_length = 4
        
        # 创建输入数据
        prompts = torch.tensor([[950, 113, 378,  14, 210, 954, 231, 572],
        [950, 113, 378,  14, 210, 954, 231, 572],
        [950, 113, 378,  14, 210, 954, 231, 572],
        [315, 295, 567, 706, 749, 876,  73, 111],
        [315, 295, 567, 706, 749, 876,  73, 111],
        [315, 295, 567, 706, 749, 876,  73, 111]])

        old_log_prob = torch.tensor([[ 0.3559, -0.6866, -0.4934,  0.2415],
        [-1.1109,  0.0915, -2.3169, -0.2168],
        [-0.9138, -0.6581,  0.0780,  0.5258],
        [-0.4880,  1.1914, -0.8140, -0.7360],
        [-1.4032,  0.0360, -0.0635,  0.6756],
        [-0.0978,  1.8446, -1.1845,  1.3835]])
        log_prob = torch.tensor([[ 1.4451,  0.8564,  2.2181,  0.5232],
        [ 0.3466, -0.1973, -1.0546,  1.2780],
        [ 0.7281, -0.7106, -0.6021,  0.9604],
        [ 0.4048, -1.3543, -0.4976,  0.4747],
        [-0.1976,  1.2683,  1.2243,  0.0981],
        [ 1.7423, -1.3527,  0.2191,  0.5526]])
        token_level_scores = torch.tensor([[1., 1., 1., 1.],
        [1., 0., 1., 0.],
        [1., 1., 0., 1.],
        [0., 1., 1., 0.],
        [1., 0., 1., 0.],
        [0., 1., 1., 0.]])
        response_mask = torch.ones(batch_size, response_length)
        
        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)




        # 创建算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "grpo"
        algorithm_config.norm_adv_by_std_in_grpo = True
        algorithm_config.use_weight = False
        
        # 调用真实的compute_pacs_loss函数
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )

        # print(loss)
        expected_loss = torch.tensor(0.6418)
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
    
    def test_compute_pacs_loss_with_naive(self):
        """测试使用naive优势估计的PACS损失计算"""
        batch_size = 4
        rollout_n = 2
        prompt_length = 10
        response_length = 5
        
        prompts = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]])
        
        old_log_prob = torch.tensor([[-0.7581,  1.0783,  0.8008,  1.6806,  0.0349],
        [ 0.3211,  1.5736, -0.8455,  1.3123,  0.6872],
        [-1.0892, -0.3553, -1.4181,  0.8963,  0.0499],
        [ 2.2667,  1.1790, -0.4345, -1.3864, -1.2862]])
        log_prob = torch.tensor([[-0.8371, -0.9224,  1.8113,  0.1606,  0.1971],
        [-1.1441,  0.3383,  1.6992,  0.0109, -0.3387],
        [-1.3407, -0.5854, -0.5644,  1.0563, -1.4692],
        [ 1.4332,  0.7440, -0.4816, -1.0495,  0.6039]])
        token_level_scores = torch.tensor([[1., 1., 1., 1., 1.],
        [1., 1., 0., 0., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 0., 1., 0.]])
        response_mask = torch.ones(batch_size, response_length)


        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)

        # 创建算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "naive"
        algorithm_config.use_weight = False
        
        # 调用真实的compute_pacs_loss函数
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )

        # print(loss)
        expected_loss = torch.tensor(5.8792)
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
        # 验证输出
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_compute_pacs_loss_with_weight(self):
        """测试使用样本权重的PACS损失计算"""
        batch_size = 4
        rollout_n = 4
        prompt_length = 10
        response_length = 5
        
        # 所有样本使用相同的prompt
        prompts = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113]])
        
        old_log_prob = torch.tensor([[-0.2429, -0.9342, -0.2483, -1.2082, -2.3169],
        [-0.2168, -1.3847, -0.8712, -0.2234,  1.7174],
        [ 0.3189, -0.4245, -0.8286,  0.3309, -1.5576],
        [ 0.9956, -0.8798, -0.6011, -1.2742,  2.1228]])
        log_prob = torch.tensor([[-1.0892, -0.3553, -0.9138, -0.6581,  2.2181],
        [ 0.5232,  0.3466, -0.1973, -1.0546,  1.2780],
        [ 0.1453,  0.2311,  0.0566,  0.4263,  0.5750],
        [-0.6417, -2.2064, -0.7508,  2.8140,  0.3598]])
        # 创建有多样性的标签
        token_level_scores = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0]
        ])

        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        response_mask = torch.ones(batch_size, response_length)
        
        # 创建算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "rloo"
        algorithm_config.use_weight = True
        algorithm_config.weight_mode = "question"
        
        # 调用真实的compute_pacs_loss函数
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )

        # print(loss)
        expected_loss = torch.tensor(0.1830)
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
        # 验证输出
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_compute_pacs_loss_reward_method_2(self):
        """测试使用reward_method=2的PACS损失计算"""
        batch_size = 4
        rollout_n = 2
        prompt_length = 10
        response_length = 5
        
        prompts = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]])
        
        old_log_prob = torch.tensor([[-0.7581,  1.0783,  0.8008,  1.6806,  0.0349],
        [ 0.3211,  1.5736, -0.8455,  1.3123,  0.6872],
        [-1.0892, -0.3553, -1.4181,  0.8963,  0.0499],
        [ 2.2667,  1.1790, -0.4345, -1.3864, -1.2862]])
        log_prob = torch.tensor([[-0.8371, -0.9224,  1.8113,  0.1606,  0.1971],
        [-1.1441,  0.3383,  1.6992,  0.0109, -0.3387],
        [-1.3407, -0.5854, -0.5644,  1.0563, -1.4692],
        [ 1.4332,  0.7440, -0.4816, -1.0495,  0.6039]])
        token_level_scores = torch.tensor([[1., 1., 1., 1., 1.],
        [1., 1., 0., 0., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 0., 1., 0.]])

        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        response_mask = torch.ones(batch_size, response_length)
        
        # 创建算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "2"
        algorithm_config.adv_estimator = "rloo"
        algorithm_config.use_weight = False
        
        # 调用真实的compute_pacs_loss函数
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )
        # print(loss)
        expected_loss = torch.tensor(1.1661)
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
        # 验证输出
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_compute_pacs_loss_with_mask(self):
        """测试带有response_mask的PACS损失计算"""
        batch_size = 4
        rollout_n = 2
        prompt_length = 10
        response_length = 6
        
        prompts = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]])
        
        old_log_prob = torch.tensor([[-0.7581,  1.0783,  0.8008,  1.6806,  0.3559, -0.6866],
        [-0.4934,  0.2415,  1.3123,  0.6872, -1.0892, -0.3553],
        [-0.9138, -0.6581,  0.0780,  0.5258,  1.1790, -0.4345],
        [-1.3864, -1.2862, -1.4032,  0.0360, -0.0635,  0.6756]])
        log_prob = torch.tensor([[ 0.3672,  0.1754,  1.3852, -0.4459,  1.4451,  0.8564],
        [ 2.2181,  0.5232,  0.5362,  0.5246,  1.1412,  0.0516],
        [ 0.7281, -0.7106, -0.6021,  0.9604, -1.7223, -0.8278],
        [ 1.3347,  0.4835, -0.1976,  1.2683,  1.2243,  0.0981]])
        token_level_scores = torch.tensor([[0., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 0.],
        [1., 1., 0., 1., 0., 1.],
        [1., 0., 1., 0., 1., 0.]])
        
        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        # 创建部分mask
        response_mask = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        ])
        
        # 创建算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "rloo"
        algorithm_config.use_weight = False
        
        # 调用真实的compute_pacs_loss函数
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )
        # print(loss)
        expected_loss = torch.tensor(6.5012)
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
        # 验证输出
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_compute_pacs_loss_prompt_validation(self):
        """测试prompt验证：同一组的prompt必须相同"""
        batch_size = 4
        rollout_n = 2
        prompt_length = 10
        response_length = 5
        
        # 创建不符合要求的prompts（同一组内prompt不同）
        prompts = torch.randint(0, 1000, (batch_size, prompt_length))
        
        old_log_prob = torch.randn(batch_size, response_length)
        log_prob = torch.randn(batch_size, response_length)
        token_level_scores = torch.randint(0, 2, (batch_size, response_length)).float()
        response_mask = torch.ones(batch_size, response_length)
        
        # 创建算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "rloo"
        algorithm_config.use_weight = False
        
        # 应该抛出AssertionError
        with self.assertRaises(AssertionError) as context:
            compute_pacs_loss(
                prompts, old_log_prob, log_prob, token_level_scores,
                response_mask, rollout_n, algorithm_config
            )
        
        self.assertIn("same prompts are not in the same batch", str(context.exception))
    
    def test_compute_pacs_loss_different_beta_values(self):
        """测试不同beta值对损失的影响"""
        batch_size = 4
        rollout_n = 2
        prompt_length = 10
        response_length = 5
        
        prompts = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706]])
        
        old_log_prob = torch.tensor([[-0.7581,  1.0783,  0.8008,  1.6806,  0.0349],
        [ 0.3211,  1.5736, -0.8455,  1.3123,  0.6872],
        [-1.0892, -0.3553, -1.4181,  0.8963,  0.0499],
        [ 2.2667,  1.1790, -0.4345, -1.3864, -1.2862]])
        log_prob = torch.tensor([[-0.8371, -0.9224,  1.8113,  0.1606,  0.1971],
        [-1.1441,  0.3383,  1.6992,  0.0109, -0.3387],
        [-1.3407, -0.5854, -0.5644,  1.0563, -1.4692],
        [ 1.4332,  0.7440, -0.4816, -1.0495,  0.6039]])
        token_level_scores = torch.tensor([[1., 1., 1., 1., 1.],
        [1., 1., 0., 0., 1.],
        [1., 1., 1., 1., 1.],
        [1., 1., 0., 1., 0.]])

        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        response_mask = torch.ones(batch_size, response_length)
        
        # 测试不同的beta值
        betas = [0.5, 1.0, 2.0]
        losses = []
        
        for beta in betas:
            algorithm_config = MagicMock()
            algorithm_config.beta = beta
            algorithm_config.reward_method = "1"
            algorithm_config.adv_estimator = "rloo"
            algorithm_config.use_weight = False
            
            loss = compute_pacs_loss(
                prompts, old_log_prob, log_prob, token_level_scores,
                response_mask, rollout_n, algorithm_config
            )
            losses.append(loss)
        # print(losses)

        expected_losses = [torch.tensor(1.2081), torch.tensor(1.8122), torch.tensor(3.1498)]
        # 验证所有损失都有效
        for i, loss in enumerate(losses):
            torch.testing.assert_close(loss, expected_losses[i], atol=1e-4, rtol=1e-4) 
    
    def test_compute_pacs_loss_edge_cases(self):
        """测试边缘情况"""
        # 测试样例1: 最小批次（rollout_n=1）
        batch_size = 1
        rollout_n = 1
        prompt_length = 5
        response_length = 3
        
        prompts = torch.tensor([[542,  67, 876, 414,  26]])
        old_log_prob = torch.tensor([[ 0.3258, -0.8676,  1.5231]])
        log_prob = torch.tensor([[ 0.6647, -1.0324, -0.2770]])
        token_level_scores = torch.tensor([[1., 1., 0.]])
        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        response_mask = torch.ones(batch_size, response_length)
        
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "naive"
        algorithm_config.use_weight = False
        
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )
        # print(loss)
        expected_loss = torch.tensor(3.4317)
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)


class TestComputePacsLossIntegration(unittest.TestCase):
    """集成测试：测试compute_pacs_loss的整体工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_advantage_estimator_comparison(self):
        """比较不同优势估计方法的差异"""
        batch_size = 6
        rollout_n = 3
        prompt_length = 10
        response_length = 5
        
        # 创建相同的输入数据
        prompts = torch.tensor([[378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [378,  14, 210, 954, 231, 572, 315, 295, 567, 706],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219],
        [749, 876,  73, 111, 899, 213, 541, 769, 287, 219]])


        
        old_log_prob = torch.tensor([[-2.3169, -0.2168, -1.3847, -0.8712, -0.2234],
        [ 1.7174,  0.3189, -0.4245, -0.8286,  0.3309],
        [-1.5576,  0.9956, -0.8798, -0.6011,  0.3672],
        [ 0.1754,  1.3852, -0.4459,  1.4451,  0.8564],
        [ 2.2181,  0.5232,  1.1754,  0.5612, -0.4527],
        [-0.7718, -0.1722,  0.5238,  0.0566,  0.4263]])
        log_prob = torch.tensor([[ 0.1971, -1.1441,  0.3383,  1.6992,  0.0109],
        [-0.3387, -1.3407, -0.5854, -0.5644,  1.0563],
        [-1.4692,  1.4332,  0.7440, -0.4816,  0.1877],
        [-0.3576, -0.3165,  0.5886, -0.8905,  0.4098],
        [-1.4570, -0.1023,  0.3499,  0.6173, -0.1693],
        [ 0.2332,  4.0356,  1.2795, -0.0127,  0.2408]])
        token_level_scores = torch.tensor([[1., 0., 0., 1., 1.],
        [0., 1., 1., 1., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0.],
        [1., 1., 1., 0., 0.],
        [0., 0., 1., 0., 0.]])

        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        response_mask = torch.ones(batch_size, response_length)
        
        # 测试三种优势估计方法
        estimators = ["rloo", "grpo", "naive"]
        losses = {}
        
        for estimator in estimators:
            algorithm_config = MagicMock()
            algorithm_config.beta = 1.0
            algorithm_config.reward_method = "1"
            algorithm_config.adv_estimator = estimator
            algorithm_config.use_weight = False
            if estimator == "grpo":
                algorithm_config.norm_adv_by_std_in_grpo = True
            
            loss = compute_pacs_loss(
                prompts, old_log_prob, log_prob, token_level_scores,
                response_mask, rollout_n, algorithm_config
            )
            losses[estimator] = loss
        
        # print(losses)

        expected_losses = {'rloo': torch.tensor(5.0535), 'grpo': torch.tensor(1.0234), 'naive': torch.tensor(2.8446)}
        # 验证所有方法都产生有效损失
        for estimator, loss in losses.items():
            torch.testing.assert_close(loss, expected_losses[estimator], atol=1e-4, rtol=1e-4) 
    
    def test_weight_mode_impact(self):  
        """测试权重模式对损失的影响"""
        batch_size = 4
        rollout_n = 4
        prompt_length = 10
        response_length = 5
        
        prompts = torch.tensor([[542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113],
        [542,  67, 876, 414,  26, 335, 620, 924, 950, 113]])
        old_log_prob = torch.tensor([[-0.2429, -0.9342, -0.2483, -1.2082, -2.3169],
        [-0.2168, -1.3847, -0.8712, -0.2234,  1.7174],
        [ 0.3189, -0.4245, -0.8286,  0.3309, -1.5576],
        [ 0.9956, -0.8798, -0.6011, -1.2742,  2.1228]])
        log_prob = torch.tensor([[-1.0892, -0.3553, -0.9138, -0.6581,  2.2181],
        [ 0.5232,  0.3466, -0.1973, -1.0546,  1.2780],
        [ 0.1453,  0.2311,  0.0566,  0.4263,  0.5750],
        [-0.6417, -2.2064, -0.7508,  2.8140,  0.3598]])

        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        token_level_scores = torch.tensor([
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0]
        ])
        response_mask = torch.ones(batch_size, response_length)
        
        # 测试不同的权重模式
        weight_modes = ["question", "only_positive", "only_negative"]
        losses = {}
        
        for mode in weight_modes:
            algorithm_config = MagicMock()
            algorithm_config.beta = 1.0
            algorithm_config.reward_method = "1"
            algorithm_config.adv_estimator = "rloo"
            algorithm_config.use_weight = True
            algorithm_config.weight_mode = mode
            
            loss = compute_pacs_loss(
                prompts, old_log_prob, log_prob, token_level_scores,
                response_mask, rollout_n, algorithm_config
            )
            losses[mode] = loss
        
        # print(losses)
        expected_losses = {'question': torch.tensor(0.1830), 'only_positive': torch.tensor(0.), 'only_negative': torch.tensor(0.)}
        # 验证所有模式都产生有效损失
        for mode, loss in losses.items():
            torch.testing.assert_close(loss, expected_losses[mode], atol=1e-4, rtol=1e-4) 
    
    def test_end_to_end_workflow(self):
        """端到端测试：完整的PACS损失计算流程"""
        batch_size = 8
        rollout_n = 4
        prompt_length = 12
        response_length = 6
        
        # 创建完整的输入数据
        prompts = torch.tensor([[210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111],
        [210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111],
        [210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111],
        [210, 954, 231, 572, 315, 295, 567, 706, 749, 876,  73, 111],
        [899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564],
        [899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564],
        [899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564],
        [899, 213, 541, 769, 287, 219, 372, 880, 475, 329, 733, 564]])
        
        old_log_prob = torch.tensor([[ 0.3189, -0.4245,  0.3057, -0.7746,  0.0349,  0.3211],
        [ 1.5736, -0.8455, -1.2742,  2.1228, -1.2347, -0.4879],
        [-1.4181,  0.8963,  0.0499,  2.2667, -0.4880,  1.1914],
        [-0.8140, -0.7360, -0.8371, -0.9224,  1.8113,  0.1606],
        [-0.0978,  1.8446, -1.1845,  1.3835, -1.2024,  0.7078],
        [-1.0759,  0.5357,  0.3466, -0.1973, -1.0546,  1.2780],
        [ 0.1453,  0.2311,  0.0087, -0.1423,  0.5750, -0.6417],
        [-2.2064, -0.7508,  2.8140,  0.3598, -0.0898,  0.4584]])
        log_prob = torch.tensor([[ 5.3619e-01,  5.2462e-01,  1.1412e+00,  5.1644e-02,  7.2811e-01,
         -7.1064e-01],
        [-6.0207e-01,  9.6045e-01, -1.7223e+00, -8.2777e-01,  1.3347e+00,
          4.8354e-01],
        [-1.9756e-01,  1.2683e+00,  1.2243e+00,  9.8117e-02,  6.4076e-01,
          5.8325e-01],
        [ 1.0669e+00, -4.5015e-01, -6.7875e-01,  5.7432e-01,  1.8775e-01,
         -3.5762e-01],
        [ 2.6491e-01,  1.2732e+00, -1.3109e-03, -3.0360e-01, -9.8644e-01,
          1.2330e-01],
        [ 3.4987e-01,  6.1728e-01,  7.2618e-01,  9.1152e-02, -3.8907e-01,
          5.2792e-01],
        [ 1.0311e+00, -7.0477e-01,  1.0131e+00, -3.3082e-01,  1.0950e+00,
          3.3989e-01],
        [ 7.1997e-01,  4.1141e-01, -5.7332e-01,  5.0686e-01, -4.7521e-01,
         -4.9203e-01]])
        token_level_scores = torch.tensor([[0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 1., 1.],
        [1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 1., 0., 0.],
        [1., 1., 1., 0., 1., 0.],
        [0., 1., 1., 0., 0., 1.],
        [1., 1., 0., 0., 0., 0.]])

        # print(prompts)
        # print(old_log_prob)
        # print(log_prob)
        # print(token_level_scores)
        response_mask = torch.ones(batch_size, response_length)
        
        # 创建完整的算法配置
        algorithm_config = MagicMock()
        algorithm_config.beta = 1.0
        algorithm_config.reward_method = "1"
        algorithm_config.adv_estimator = "grpo"
        algorithm_config.norm_adv_by_std_in_grpo = True
        algorithm_config.use_weight = True
        algorithm_config.weight_mode = "question"
        
        # 计算损失
        loss = compute_pacs_loss(
            prompts, old_log_prob, log_prob, token_level_scores,
            response_mask, rollout_n, algorithm_config
        )
        # print(loss)
        expected_loss = torch.tensor(0.6249)
        # 验证最终输出
        torch.testing.assert_close(loss, expected_loss, atol=1e-4, rtol=1e-4) 
        
        # 验证损失可以用于反向传播
        self.assertTrue(loss.requires_grad or not old_log_prob.requires_grad) 


if __name__ == "__main__":
    unittest.main()
