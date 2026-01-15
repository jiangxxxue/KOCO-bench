import unittest
import sys
import os
import torch
# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual function we want to test
from examples.reward_function.aepo import compute_score


class TestComputeScore(unittest.TestCase):
    """测试compute_score函数 - ARES文档第3节描述的分层熵奖励计算"""
    
    def setUp(self):
        """设置测试环境"""
        pass
    
    def test_compute_score_basic_structure(self):
        """测试基本输入输出结构"""
        reward_inputs = [
            {
                "response": "<think>思考过程</think>\\boxed{42}",
                "ground_truth": "42",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)
        
        # print(scores[0]["overall"])
        # print(scores[0]["accuracy"])
        # print(scores[0]["format"])
        # print(scores[0]["high_entropy_token_num_score"])

        expected_overall = 1.0
        expected_accuracy = 1.0
        expected_format = 1.0
        expected_high_entropy_token_num_score = -0.0
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["format"], expected_format, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # 验证输出结构
        # self.assertEqual(len(scores), 1)
        # self.assertIn("overall", scores[0])
        # self.assertIn("accuracy", scores[0])
        # self.assertIn("format", scores[0])
        # self.assertIn("high_entropy_token_num_score", scores[0])
        
        # # 验证所有分数都是有限数值
        # self.assertTrue(isinstance(scores[0]["overall"], float))
        # self.assertTrue(isinstance(scores[0]["accuracy"], float))
        # self.assertIn(scores[0]["accuracy"], [0.0, 1.0])
    
    def test_compute_score_invalid_input(self):
        """测试无效输入"""
        # 测试非列表输入应该抛出ValueError
        with self.assertRaises(ValueError):
            compute_score({"response": "test"})
    
    def test_compute_score_easy_correct_in_tolerance(self):
        """测试Easy任务，答案正确，HWE token数在容忍带内"""
        reward_inputs = [
            {
                "response": "<think>简单推理</think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)

        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])

        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        expected_overall = 1.0
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # Easy任务，正确答案，delta=0在容忍带内，熵奖励应该接近0或很小
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], -0.1)
        # self.assertLessEqual(scores[0]["high_entropy_token_num_score"], 0.1)
        # # overall = accuracy + entropy_score ≈ 1.0
        # self.assertAlmostEqual(scores[0]["overall"], 1.0, delta=0.2)
    
    def test_compute_score_easy_correct_over_thinking(self):
        """测试Easy任务，答案正确但过度思考（HWE token数过多）"""
        reward_inputs = [
            {
                "response": "<think>非常长的推理过程</think>\\boxed{50}",
                "ground_truth": "50",
                "difficulty": "easy",
                "high_entropy_token_num": 30,  # 远超目标
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)
        
        #     print(scores[0]["accuracy"])
        #       print(scores[0]["high_entropy_token_num_score"])
        #     print(scores[0]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.5
        expected_overall = 0.5
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # Easy任务，正确但过度思考，应施加惩罚
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertLess(scores[0]["high_entropy_token_num_score"], 0.0)  # 负奖励（惩罚）
        # # overall = 1.0 + 负奖励 < 1.0
        # self.assertLess(scores[0]["overall"], 1.0)
    
    def test_compute_score_easy_incorrect_some_exploration(self):
        """测试Easy任务，答案错误但有一定探索"""
        reward_inputs = [
            {
                "response": "<think>错误的推理</think>\\boxed{999}",
                "ground_truth": "50",
                "difficulty": "easy",
                "high_entropy_token_num": 20,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)

        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])

        expected_accuracy = 0.0
        expected_high_entropy_token_num_score = 0.2957809118870263
        expected_overall = 0.2957809118870263

        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # Easy任务，答案错误，但有探索，给予小幅鼓励
        # self.assertEqual(scores[0]["accuracy"], 0.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)  # 给予鼓励
        # # overall = 0.0 + 正奖励
        # self.assertGreater(scores[0]["overall"], 0.0)
        # self.assertLess(scores[0]["overall"], 0.5)  # 鼓励幅度较小
    
    def test_compute_score_medium_correct_in_tolerance(self):
        """测试Medium任务，答案正确，HWE token数在容忍带内"""
        reward_inputs = [
            {
                "response": "<think>中等难度推理</think>\\boxed{200}",
                "ground_truth": "200",
                "difficulty": "medium",
                "high_entropy_token_num": 20,
                "target_high_entropy_token_num": 20,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)
        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        expected_overall = 1.0
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # Medium任务，正确答案，delta=0在容忍带内，熵奖励应该接近0
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], -0.1)
        # self.assertLessEqual(scores[0]["high_entropy_token_num_score"], 0.1)
        # self.assertAlmostEqual(scores[0]["overall"], 1.0, delta=0.2)
    
    def test_compute_score_medium_correct_deviation(self):
        """测试Medium任务，答案正确但偏差较大（过多或过少）"""
        # 测试过多的情况
        reward_inputs_over = [
            {
                "response": "<think>过多推理</think>\\boxed{300}",
                "ground_truth": "300",
                "difficulty": "medium",
                "high_entropy_token_num": 40,
                "target_high_entropy_token_num": 20,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_over = compute_score(reward_inputs_over)
        # print(scores_over[0]["accuracy"])
        # print(scores_over[0]["high_entropy_token_num_score"])
        # print(scores_over[0]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.5
        expected_overall = 0.5
        torch.testing.assert_close(scores_over[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_over[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_over[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # self.assertEqual(scores_over[0]["accuracy"], 1.0)
        # self.assertLess(scores_over[0]["high_entropy_token_num_score"], 0.0)  # 惩罚
        
        # 测试过少的情况
        reward_inputs_under = [
            {
                "response": "<think>简短</think>\\boxed{300}",
                "ground_truth": "300",
                "difficulty": "medium",
                "high_entropy_token_num": 5,
                "target_high_entropy_token_num": 20,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_under = compute_score(reward_inputs_under)
        self.assertEqual(scores_under[0]["accuracy"], 1.0)
        self.assertLess(scores_under[0]["high_entropy_token_num_score"], 0.0)  # 惩罚
    
    def test_compute_score_medium_incorrect_exploration(self):
        """测试Medium任务，答案错误但有探索"""
        reward_inputs = [
            {
                "response": "<think>错误推理过程</think>\\boxed{999}",
                "ground_truth": "300",
                "difficulty": "medium",
                "high_entropy_token_num": 30,
                "target_high_entropy_token_num": 20,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)
        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])
        expected_accuracy = 0.0
        expected_high_entropy_token_num_score = 0.3523188311911529
        expected_overall =0.3523188311911529
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # Medium任务，答案错误但有探索，给予中等鼓励
        self.assertEqual(scores[0]["accuracy"], 0.0)
        self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)
        # 鼓励幅度应该比Easy大
        self.assertGreater(scores[0]["overall"], 0.0)
    
    def test_compute_score_hard_correct_sufficient_exploration(self):
        """测试Hard任务，答案正确且探索充分"""
        reward_inputs = [
            {
                "response": "<think>深入详细的推理过程</think>\\boxed{500}",
                "ground_truth": "500",
                "difficulty": "hard",
                "high_entropy_token_num": 50,
                "target_high_entropy_token_num": 40,
                "alpha_entropy": 0.6
            }
        ]
        
        scores = compute_score(reward_inputs)

        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = 0.7197585479060642
        expected_overall = 1.7197585479060642
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        
        # # Hard任务，正确且探索充分，应给予正奖励
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)  # 正奖励
        # # overall = 1.0 + 正奖励 > 1.0
        # self.assertGreater(scores[0]["overall"], 1.0)
    
    def test_compute_score_hard_correct_insufficient_exploration(self):
        """测试Hard任务，答案正确但探索不足"""
        reward_inputs = [
            {
                "response": "<think>简短</think>\\boxed{500}",
                "ground_truth": "500",
                "difficulty": "hard",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 40,
                "alpha_entropy": 0.6
            }
        ]
        
        scores = compute_score(reward_inputs)
        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.27999999999999997
        expected_overall = 0.72
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # Hard任务，正确但探索不足，应施加轻度惩罚
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertLess(scores[0]["high_entropy_token_num_score"], 0.0)  # 轻度惩罚
        # # overall = 1.0 + 负奖励
        # self.assertLess(scores[0]["overall"], 1.0)
        # self.assertGreater(scores[0]["overall"], 0.5)  # 但仍然较高
    
    def test_compute_score_hard_incorrect_with_exploration(self):
        """测试Hard任务，答案错误但探索充分"""
        reward_inputs = [
            {
                "response": "<think>大量错误推理</think>\\boxed{999}",
                "ground_truth": "500",
                "difficulty": "hard",
                "high_entropy_token_num": 50,
                "target_high_entropy_token_num": 40,
                "alpha_entropy": 0.6
            }
        ]
        
        scores = compute_score(reward_inputs)
        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])
        expected_accuracy = 0.0
        expected_high_entropy_token_num_score = 0.36
        expected_overall = 0.36
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # Hard任务，答案错误但探索充分，给予鼓励
        self.assertEqual(scores[0]["accuracy"], 0.0)
        self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)
        # 鼓励幅度应该较大
        self.assertGreater(scores[0]["overall"], 0.0)
    
    def test_compute_score_batch_processing(self):
        """测试批量处理多个样本"""
        reward_inputs = [
            {
                "response": "<think>推理1</think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            },
            {
                "response": "<think>推理2</think>\\boxed{200}",
                "ground_truth": "200",
                "difficulty": "medium",
                "high_entropy_token_num": 20,
                "target_high_entropy_token_num": 20,
                "alpha_entropy": 0.5
            },
            {
                "response": "<think>推理3</think>\\boxed{300}",
                "ground_truth": "300",
                "difficulty": "hard",
                "high_entropy_token_num": 40,
                "target_high_entropy_token_num": 40,
                "alpha_entropy": 0.6
            }
        ]
        
        scores = compute_score(reward_inputs)
        
        # 验证批量处理
        self.assertEqual(len(scores), 3)
        
        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        # print(scores[0]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        expected_overall = 1.0
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)

        # print(scores[1]["accuracy"])
        # print(scores[1]["high_entropy_token_num_score"])
        # print(scores[1]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        expected_overall = 1.0
        torch.testing.assert_close(scores[1]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[1]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[1]["overall"], expected_overall, atol=1e-5, rtol=1e-5)

        # print(scores[2]["accuracy"])
        # print(scores[2]["high_entropy_token_num_score"])
        # print(scores[2]["overall"])
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = 0.713292509271552
        expected_overall = 1.713292509271552
        torch.testing.assert_close(scores[2]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[2]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[2]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        # 验证每个样本的结果
        # for score in scores:
        #     self.assertIn("overall", score)
        #     self.assertIn("accuracy", score)
        #     self.assertIn("high_entropy_token_num_score", score)
        #     self.assertEqual(score["accuracy"], 1.0)
    
    def test_compute_score_different_alpha_entropy(self):
        """测试不同alpha_entropy值的影响"""
        base_input = {
            "response": "<think>推理过程</think>\\boxed{100}",
            "ground_truth": "999",  # 错误答案
            "difficulty": "easy",
            "high_entropy_token_num": 20,
            "target_high_entropy_token_num": 10
        }
        
        # 小alpha
        input_small_alpha = {**base_input, "alpha_entropy": 0.1}
        scores_small = compute_score([input_small_alpha])
        
        # 大alpha
        input_large_alpha = {**base_input, "alpha_entropy": 1.0}
        scores_large = compute_score([input_large_alpha])

        
        # print(scores_small[0]["high_entropy_token_num_score"])
        # print(scores_large[0]["high_entropy_token_num_score"])
        expected_high_entropy_token_num_score_small = 0.05915618237740526
        expected_high_entropy_token_num_score_large = 0.5915618237740526
        torch.testing.assert_close(scores_small[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score_small, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_large[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score_large, atol=1e-5, rtol=1e-5)
        
        # alpha越大，熵奖励/惩罚的影响应该越大
        entropy_score_small = abs(scores_small[0]["high_entropy_token_num_score"])
        entropy_score_large = abs(scores_large[0]["high_entropy_token_num_score"])
        # self.assertLess(entropy_score_small, entropy_score_large)
    
    def test_compute_score_format_detection(self):
        """测试格式检测功能"""
        # 正确格式
        reward_inputs_correct = [
            {
                "response": "<think>推理</think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_correct = compute_score(reward_inputs_correct)
        self.assertEqual(scores_correct[0]["format"], 1.0)
        
        # 错误格式（缺少think标签）
        reward_inputs_wrong = [
            {
                "response": "直接回答\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_wrong = compute_score(reward_inputs_wrong)
        # 格式不正确，format应该为0
        self.assertEqual(scores_wrong[0]["format"], 0.0)
    
    def test_compute_score_accuracy_extraction(self):
        """测试准确率自动提取功能"""
        # 不提供accuracy，让函数自动提取
        reward_inputs = [
            {
                "response": "<think>推理</think>\\boxed{42}",
                "ground_truth": "42",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)
        
        # 应该自动判断为正确
        self.assertEqual(scores[0]["accuracy"], 1.0)
        
        # 测试错误答案
        reward_inputs_wrong = [
            {
                "response": "<think>推理</think>\\boxed{999}",
                "ground_truth": "42",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_wrong = compute_score(reward_inputs_wrong)
        self.assertEqual(scores_wrong[0]["accuracy"], 0.0)
    
    def test_compute_score_edge_cases(self):
        """测试边缘情况"""
        # 测试HWE token数为0
        reward_inputs_zero = [
            {
                "response": "<think></think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 0,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_zero = compute_score(reward_inputs_zero)
        # print(scores_zero[0]["overall"])
        # print(scores_zero[0]["accuracy"])
        # print(scores_zero[0]["high_entropy_token_num_score"])
        expected_overall = 1.0
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        torch.testing.assert_close(scores_zero[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_zero[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_zero[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # self.assertIsInstance(scores_zero[0]["overall"], float)
        # self.assertTrue(abs(scores_zero[0]["overall"]) < 100)  # 合理范围
        
        # 测试目标为0
        reward_inputs_target_zero = [
            {
                "response": "<think>推理</think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 0,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_target_zero = compute_score(reward_inputs_target_zero)
        # print(scores_target_zero[0]["overall"])
        # print(scores_target_zero[0]["accuracy"])
        # print(scores_target_zero[0]["high_entropy_token_num_score"])
        expected_overall = 0.5
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.5
        torch.testing.assert_close(scores_target_zero[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_target_zero[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_target_zero[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
    
    def test_compute_score_default_parameters(self):
        """测试默认参数"""
        # 不提供难度信息
        reward_inputs = [
            {
                "response": "<think>推理</think>\\boxed{100}",
                "ground_truth": "100",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10
            }
        ]
        
        # 使用默认alpha_entropy
        scores = compute_score(reward_inputs, alpha_entropy=0.5)
        # print(scores[0]["overall"])
        # print(scores[0]["accuracy"])
        # print(scores[0]["high_entropy_token_num_score"])
        expected_overall = 1.0
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        torch.testing.assert_close(scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)

        self.assertIsNotNone(scores)
        self.assertEqual(len(scores), 1)
        self.assertIn("overall", scores[0])


class TestComputeScoreIntegration(unittest.TestCase):
    """集成测试：测试compute_score的复杂场景"""
    
    def test_difficulty_progression(self):
        """测试不同难度下的奖励变化"""
        # 相同的条件，不同难度
        base_correct = {
            "response": "<think>充分推理</think>\\boxed{100}",
            "ground_truth": "100",
            "high_entropy_token_num": 30,
            "target_high_entropy_token_num": 20,
            "alpha_entropy": 0.5
        }
        
        # Easy: 过度思考会被惩罚
        easy_input = {**base_correct, "difficulty": "easy"}
        easy_scores = compute_score([easy_input])
        # print(easy_scores[0]["overall"])
        # print(easy_scores[0]["accuracy"])
        # print(easy_scores[0]["high_entropy_token_num_score"])
        expected_overall = 0.5
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.5
        torch.testing.assert_close(easy_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(easy_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(easy_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # Medium: 偏差会被惩罚
        medium_input = {**base_correct, "difficulty": "medium"}
        medium_scores = compute_score([medium_input])
            # print(medium_scores[0]["overall"])
            # print(medium_scores[0]["accuracy"])
            # print(medium_scores[0]["high_entropy_token_num_score"])
        expected_overall = 0.78125
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.21875
        torch.testing.assert_close(medium_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(medium_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(medium_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # Hard: 充分探索会被奖励
        hard_input = {**base_correct, "difficulty": "hard"}
        hard_scores = compute_score([hard_input])
        # print(hard_scores[0]["overall"])
        # print(hard_scores[0]["accuracy"])
        # print(hard_scores[0]["high_entropy_token_num_score"])
        expected_overall = 1.597931529219062
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = 0.5979315292190618
        torch.testing.assert_close(hard_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(hard_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(hard_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # # 验证趋势：Hard应该最高，Easy应该被惩罚
        # self.assertGreater(hard_scores[0]["overall"], medium_scores[0]["overall"])
        # self.assertLess(easy_scores[0]["overall"], 1.0)  # Easy被惩罚
        # self.assertGreater(hard_scores[0]["overall"], 1.0)  # Hard被奖励
    
    def test_accuracy_dominant_factor(self):
        """测试准确率是主导因素"""
        # 正确答案，即使熵奖励为负，overall仍然较高
        correct_input = [
            {
                "response": "<think>过度推理</think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 100,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        correct_scores = compute_score(correct_input)

        # print(correct_scores[0]["overall"])
        # print(correct_scores[0]["accuracy"])
        # print(correct_scores[0]["high_entropy_token_num_score"])
        expected_overall = 0.5
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.5
        torch.testing.assert_close(correct_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(correct_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(correct_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        
        # 错误答案，即使熵奖励为正，overall仍然较低
        incorrect_input = [
            {
                "response": "<think>充分推理</think>\\boxed{999}",
                "ground_truth": "100",
                "difficulty": "hard",
                "high_entropy_token_num": 50,
                "target_high_entropy_token_num": 40,
                "alpha_entropy": 0.5
            }
        ]
        
        incorrect_scores = compute_score(incorrect_input)
        # print(incorrect_scores[0]["overall"])
        # print(incorrect_scores[0]["accuracy"])
        # print(incorrect_scores[0]["high_entropy_token_num_score"])
        expected_overall = 0.3
        expected_accuracy = 0.0
        expected_high_entropy_token_num_score = 0.3
        torch.testing.assert_close(incorrect_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(incorrect_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(incorrect_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # 正确答案的overall应该大于错误答案
        self.assertGreater(correct_scores[0]["overall"], incorrect_scores[0]["overall"])
    
    def test_margin_tolerance(self):
        """测试容忍带机制"""
        # 在容忍带边缘的测试
        base_input = {
            "response": "<think>推理</think>\\boxed{100}",
            "ground_truth": "100",
            "difficulty": "easy",
            "target_high_entropy_token_num": 100,
            "alpha_entropy": 0.5
        }
        
        # 刚好在目标值
        exact_input = {**base_input, "high_entropy_token_num": 100}
        exact_scores = compute_score([exact_input])
        
        # print(exact_scores[0]["overall"])
        # print(exact_scores[0]["accuracy"])
        # print(exact_scores[0]["high_entropy_token_num_score"])
        expected_overall = 1.0
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        torch.testing.assert_close(exact_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(exact_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(exact_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # 稍微超出（在容忍带内）
        slightly_over_input = {**base_input, "high_entropy_token_num": 105}
        slightly_over_scores = compute_score([slightly_over_input])
        # print(slightly_over_scores[0]["overall"])
        # print(slightly_over_scores[0]["accuracy"])
        # print(slightly_over_scores[0]["high_entropy_token_num_score"])
        expected_overall = 1.0
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.0
        torch.testing.assert_close(slightly_over_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(slightly_over_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(slightly_over_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # 大幅超出（超出容忍带）
        far_over_input = {**base_input, "high_entropy_token_num": 150}
        far_over_scores = compute_score([far_over_input])
        # print(far_over_scores[0]["overall"])
        # print(far_over_scores[0]["accuracy"])
        # print(far_over_scores[0]["high_entropy_token_num_score"])
        expected_overall = 0.5
        expected_accuracy = 1.0
        expected_high_entropy_token_num_score = -0.5
        torch.testing.assert_close(far_over_scores[0]["overall"], expected_overall, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(far_over_scores[0]["accuracy"], expected_accuracy, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(far_over_scores[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score, atol=1e-5, rtol=1e-5)
        # 验证：在容忍带内的惩罚应该很小或为0
        self.assertAlmostEqual(exact_scores[0]["high_entropy_token_num_score"], 0.0, delta=0.1)
        self.assertGreater(slightly_over_scores[0]["high_entropy_token_num_score"], 
                          far_over_scores[0]["high_entropy_token_num_score"])


if __name__ == "__main__":
    unittest.main()

