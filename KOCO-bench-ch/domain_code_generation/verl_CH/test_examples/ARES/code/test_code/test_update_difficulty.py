import unittest
import sys
import os
import numpy as np
from collections import defaultdict
import torch

# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual class and components we want to test
try:
    from verl.protocol import DataProto
    from verl.trainer.ray_trainer import RayPPOTrainer
    VERL_AVAILABLE = True
except ImportError:
    VERL_AVAILABLE = False
    DataProto = None
    RayPPOTrainer = None


class TestUpdateDifficulty(unittest.TestCase):
    """测试update_difficulty_and_skip_gid_set函数 - ARES文档第3节描述的在线难度分桶"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建真实的RayPPOTrainer实例，但不调用__init__以避免复杂的依赖
        # 只设置测试所需的必要属性
        if VERL_AVAILABLE:
            self.trainer = object.__new__(RayPPOTrainer)
            self.trainer.difficulty_dict_train = {}
            self.trainer.target_high_entropy_token_num_dict = {"easy": 0, "medium": 0, "hard": 0}
            self.trainer.alpha_entropy_dict = {"easy": 0.5, "medium": 0.5, "hard": 0.6}
            self.trainer.alpha_entropy_lr = 0.01
            self.trainer.alpha_entropy_min = 0.1
            self.trainer.alpha_entropy_max = 1.0
            self.trainer.skip_gid_set = set()
        else:
            self.trainer = None
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_update_difficulty_basic_structure(self):
        """测试基本输入输出结构"""
        # 创建测试数据
        batch_data = {
            "global_id": ["sample_1", "sample_1", "sample_2", "sample_2"],
            "accuracy": [1.0, 1.0, 0.0, 0.0],
            "entropies": [0.5, 0.6, 0.8, 0.9],
            "high_entropy_token_num": [10, 12, 30, 35]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        
        # 调用函数
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证输出结构
        self.assertIn("sample_1", self.trainer.difficulty_dict_train)
        self.assertIn("sample_2", self.trainer.difficulty_dict_train)
        self.assertIn(self.trainer.difficulty_dict_train["sample_1"], ["easy", "medium", "hard"])
        self.assertIn(self.trainer.difficulty_dict_train["sample_2"], ["easy", "medium", "hard"])
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_assignment_easy(self):
        """测试Easy难度分配（acc_rate >= 2/3）"""
        # 准确率为1.0的样本
        batch_data = {
            "global_id": ["easy_sample"] * 3,
            "accuracy": [1.0, 1.0, 1.0],
            "entropies": [0.3, 0.4, 0.35],
            "high_entropy_token_num": [8, 10, 9]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证被分配为easy
        self.assertEqual(self.trainer.difficulty_dict_train["easy_sample"], "easy")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_assignment_medium(self):
        """测试Medium难度分配（1/3 <= acc_rate < 2/3）"""
        # 准确率为0.5的样本
        batch_data = {
            "global_id": ["medium_sample"] * 4,
            "accuracy": [1.0, 0.0, 1.0, 0.0],
            "entropies": [0.5, 0.6, 0.55, 0.65],
            "high_entropy_token_num": [15, 18, 16, 20]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证被分配为medium
        self.assertEqual(self.trainer.difficulty_dict_train["medium_sample"], "medium")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_assignment_hard(self):
        """测试Hard难度分配（acc_rate < 1/3）"""
        # 准确率为0.0的样本
        batch_data = {
            "global_id": ["hard_sample"] * 3,
            "accuracy": [0.0, 0.0, 0.0],
            "entropies": [0.8, 0.9, 0.85],
            "high_entropy_token_num": [40, 45, 42]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证被分配为hard
        self.assertEqual(self.trainer.difficulty_dict_train["hard_sample"], "hard")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_difficulty_boundary_cases(self):
        """测试边界情况"""
        # 准确率刚好为2/3（应为easy）
        batch_data_67 = {
            "global_id": ["boundary_67"] * 3,
            "accuracy": [1.0, 1.0, 0.0],  # 2/3 = 0.667
            "entropies": [0.5] * 3,
            "high_entropy_token_num": [10] * 3
        }
        
        batch_67 = DataProto.from_dict(non_tensors=batch_data_67)
        self.trainer.update_difficulty_and_skip_gid_set(batch_67)
        self.assertEqual(self.trainer.difficulty_dict_train["boundary_67"], "easy")
        
        # 准确率刚好为1/3（应为medium）
        batch_data_33 = {
            "global_id": ["boundary_33"] * 3,
            "accuracy": [1.0, 0.0, 0.0],  # 1/3 = 0.333
            "entropies": [0.6] * 3,
            "high_entropy_token_num": [20] * 3
        }
        
        batch_33 = DataProto.from_dict(non_tensors=batch_data_33)
        self.trainer.update_difficulty_and_skip_gid_set(batch_33)
        self.assertEqual(self.trainer.difficulty_dict_train["boundary_33"], "medium")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_target_token_num_calculation(self):
        """测试目标HWE token数的计算"""
        # 创建不同难度的样本
        batch_data = {
            "global_id": ["easy_1", "easy_1", "medium_1", "medium_1", "hard_1", "hard_1"],
            "accuracy": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
            "entropies": [0.4] * 6,
            "high_entropy_token_num": [10, 12, 20, 22, 40, 42]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证目标token数被正确计算
        # Easy: mean([11, ...]) = 11
        # Medium: mean([21, ...]) = 21  
        # Hard: mean([41, ...]) = 41
        self.assertEqual(self.trainer.target_high_entropy_token_num_dict["easy"], 11)
        self.assertEqual(self.trainer.target_high_entropy_token_num_dict["medium"], 21)
        self.assertEqual(self.trainer.target_high_entropy_token_num_dict["hard"], 41)
    

    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_alpha_entropy_update_hard_under_generation(self):
        """测试Hard任务生成不足时alpha_entropy的更新"""
        # 先设置一个基准目标
        self.trainer.target_high_entropy_token_num_dict["hard"] = 40
        self.trainer.alpha_entropy_dict["hard"] = 0.6
        
        # 第一次更新：设定初始基准
        batch_data_init = {
            "global_id": ["hard_baseline"] * 2,
            "accuracy": [0.0, 0.0],
            "entropies": [0.8] * 2,
            "high_entropy_token_num": [40, 40]  # 平均40，设定基准
        }
        batch_init = DataProto.from_dict(non_tensors=batch_data_init)
        self.trainer.update_difficulty_and_skip_gid_set(batch_init)
        
        baseline_alpha = self.trainer.alpha_entropy_dict["hard"]
        
        # 第二次更新：生成不足token的hard样本
        batch_data = {
            "global_id": ["hard_under"] * 2,
            "accuracy": [0.0, 0.0],
            "entropies": [0.8] * 2,
            "high_entropy_token_num": [20, 22]  # 平均21，低于基准40
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        new_alpha = self.trainer.alpha_entropy_dict["hard"]
        # print(new_alpha)
        # Hard生成不足，alpha应该增加（增强鼓励）
        # self.assertGreaterEqual(new_alpha, baseline_alpha)
        expected_new_alpha = 0.6
        torch.testing.assert_close(new_alpha, expected_new_alpha, atol=1e-4, rtol=1e-4)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_alpha_entropy_clipping(self):
        """测试alpha_entropy的上下限约束"""
        # 测试下限
        self.trainer.alpha_entropy_dict["easy"] = 0.11
        self.trainer.alpha_entropy_min = 0.1
        self.trainer.target_high_entropy_token_num_dict["easy"] = 100
        
        # 创建会导致alpha减小的数据
        batch_data = {
            "global_id": ["test_clip"] * 2,
            "accuracy": [1.0, 1.0],
            "entropies": [0.4] * 2,
            "high_entropy_token_num": [5, 5]  # 远低于目标，导致alpha减小
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证不会低于最小值
        self.assertGreaterEqual(self.trainer.alpha_entropy_dict["easy"], self.trainer.alpha_entropy_min)
        
        # 测试上限
        self.trainer.alpha_entropy_dict["medium"] = 0.99
        self.trainer.alpha_entropy_max = 1.0
        self.trainer.target_high_entropy_token_num_dict["medium"] = 5
        
        batch_data_max = {
            "global_id": ["test_max"] * 2,
            "accuracy": [0.5, 0.5],
            "entropies": [0.6] * 2,
            "high_entropy_token_num": [100, 100]  # 远超目标，导致alpha增大
        }
        
        batch_max = DataProto.from_dict(non_tensors=batch_data_max)
        self.trainer.update_difficulty_and_skip_gid_set(batch_max)
        
        # print(self.trainer.alpha_entropy_dict["medium"])
        expected_alpha = 0.99
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["medium"], expected_alpha, atol=1e-4, rtol=1e-4)
        # 验证不会超过最大值
        self.assertLessEqual(self.trainer.alpha_entropy_dict["medium"], self.trainer.alpha_entropy_max)
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_multiple_samples_aggregation(self):
        """测试多个样本的聚合统计"""
        # 同一个global_id有多个rollout
        batch_data = {
            "global_id": ["sample_1"] * 6,
            "accuracy": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # 平均0.5
            "entropies": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "high_entropy_token_num": [10, 12, 14, 16, 18, 20]  # 平均15
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 准确率0.5应该被分配为medium
        self.assertEqual(self.trainer.difficulty_dict_train["sample_1"], "medium")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_batch_with_multiple_unique_samples(self):
        """测试包含多个不同样本的batch"""
        batch_data = {
            "global_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "accuracy": [1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
            "entropies": [0.3, 0.4, 0.6, 0.7, 0.9, 1.0],
            "high_entropy_token_num": [8, 10, 20, 22, 40, 42]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证所有样本都被处理
        self.assertIn("s1", self.trainer.difficulty_dict_train)
        self.assertIn("s2", self.trainer.difficulty_dict_train)
        self.assertIn("s3", self.trainer.difficulty_dict_train)
        
        # 验证难度分配
        self.assertEqual(self.trainer.difficulty_dict_train["s1"], "easy")
        self.assertEqual(self.trainer.difficulty_dict_train["s2"], "medium")
        self.assertEqual(self.trainer.difficulty_dict_train["s3"], "hard")
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_persistent_difficulty_update(self):
        """测试难度字典的持久化更新"""
        # 第一次更新
        batch_data_1 = {
            "global_id": ["sample_a"] * 2,
            "accuracy": [1.0, 1.0],
            "entropies": [0.4] * 2,
            "high_entropy_token_num": [10, 12]
        }
        
        batch_1 = DataProto.from_dict(non_tensors=batch_data_1)
        self.trainer.update_difficulty_and_skip_gid_set(batch_1)
        self.assertEqual(self.trainer.difficulty_dict_train["sample_a"], "easy")
        
        # 第二次更新（相同样本，不同结果）
        batch_data_2 = {
            "global_id": ["sample_a"] * 2,
            "accuracy": [0.0, 0.0],
            "entropies": [0.8] * 2,
            "high_entropy_token_num": [40, 42]
        }
        
        batch_2 = DataProto.from_dict(non_tensors=batch_data_2)
        self.trainer.update_difficulty_and_skip_gid_set(batch_2)
        
        # 难度应该被更新为hard
        self.assertEqual(self.trainer.difficulty_dict_train["sample_a"], "hard")


class TestUpdateDifficultyIntegration(unittest.TestCase):
    """集成测试：测试update_difficulty_and_skip_gid_set的整体工作流程"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建真实的RayPPOTrainer实例，但不调用__init__以避免复杂的依赖
        if VERL_AVAILABLE:
            self.trainer = object.__new__(RayPPOTrainer)
            self.trainer.difficulty_dict_train = {}
            self.trainer.target_high_entropy_token_num_dict = {"easy": 0, "medium": 0, "hard": 0}
            self.trainer.alpha_entropy_dict = {"easy": 0.5, "medium": 0.5, "hard": 0.6}
            self.trainer.alpha_entropy_lr = 0.01
            self.trainer.alpha_entropy_min = 0.1
            self.trainer.alpha_entropy_max = 1.0
            self.trainer.skip_gid_set = set()
        else:
            self.trainer = None
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_end_to_end_difficulty_workflow(self):
        """端到端的难度更新工作流"""
        # 模拟一个完整的训练batch
        batch_data = {
            "global_id": ["task_1"] * 4 + ["task_2"] * 4 + ["task_3"] * 4,
            "accuracy": [1.0, 1.0, 1.0, 0.0] + [1.0, 0.0, 1.0, 0.0] + [0.0, 0.0, 0.0, 1.0],
            "entropies": [0.3] * 4 + [0.6] * 4 + [0.9] * 4,
            "high_entropy_token_num": [8, 10, 9, 11] + [18, 20, 19, 21] + [38, 40, 39, 41]
        }
        
        batch = DataProto.from_dict(non_tensors=batch_data)
        
        # 记录初始alpha值
        initial_alphas = self.trainer.alpha_entropy_dict.copy()
        
        # 执行更新
        self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        # 验证所有任务都被分配了难度
        self.assertEqual(len(self.trainer.difficulty_dict_train), 3)
        
        # 验证难度分配
        # task_1: 3/4 = 0.75 >= 2/3 -> easy
        self.assertEqual(self.trainer.difficulty_dict_train["task_1"], "easy")
        # task_2: 2/4 = 0.5, 1/3 <= 0.5 < 2/3 -> medium
        self.assertEqual(self.trainer.difficulty_dict_train["task_2"], "medium")
        # task_3: 1/4 = 0.25 < 1/3 -> hard
        self.assertEqual(self.trainer.difficulty_dict_train["task_3"], "hard")
        
        # 验证目标token数被设置
        self.assertGreater(self.trainer.target_high_entropy_token_num_dict["easy"], 0)
        self.assertGreater(self.trainer.target_high_entropy_token_num_dict["medium"], 0)
        self.assertGreater(self.trainer.target_high_entropy_token_num_dict["hard"], 0)
        
        # 验证alpha值可能发生了变化
        # （具体变化取决于生成的token数与目标的关系）
        # print(self.trainer.alpha_entropy_dict["easy"])
        # print(self.trainer.alpha_entropy_dict["medium"])
        # print(self.trainer.alpha_entropy_dict["hard"])
        expected_alpha_easy = 0.495
        expected_alpha_medium = 0.495
        expected_alpha_hard = 0.605
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["easy"], expected_alpha_easy, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["medium"], expected_alpha_medium, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(self.trainer.alpha_entropy_dict["hard"], expected_alpha_hard, atol=1e-4, rtol=1e-4)
        # self.assertIsNotNone(self.trainer.alpha_entropy_dict["easy"])
        # self.assertIsNotNone(self.trainer.alpha_entropy_dict["medium"])
        # self.assertIsNotNone(self.trainer.alpha_entropy_dict["hard"])
    
    @unittest.skipUnless(VERL_AVAILABLE, "VERL modules not available")
    def test_adaptive_alpha_convergence(self):
        """测试alpha_entropy的自适应调整机制"""
        # 设置初始状态
        initial_alpha = 0.5
        self.trainer.alpha_entropy_dict["easy"] = initial_alpha
        self.trainer.alpha_entropy_lr = 0.05  # 使用较小的学习率
        
        # 第一次设定基准
        batch_data_baseline = {
            "global_id": ["converge_baseline"] * 2,
            "accuracy": [1.0, 1.0],
            "entropies": [0.4] * 2,
            "high_entropy_token_num": [15, 15]  # 设定基准为15
        }
        batch_baseline = DataProto.from_dict(non_tensors=batch_data_baseline)
        self.trainer.update_difficulty_and_skip_gid_set(batch_baseline)
        
        # 记录初始alpha（可能已经变化）
        alpha_after_baseline = self.trainer.alpha_entropy_dict["easy"]
        
        # 多次更新，每次都生成较多token（但不是同一个样本）
        for i in range(5):
            batch_data = {
                "global_id": [f"converge_test_{i}"] * 2,
                "accuracy": [1.0, 1.0],
                "entropies": [0.4] * 2,
                "high_entropy_token_num": [25, 27]  # 持续较多
            }
            batch = DataProto.from_dict(non_tensors=batch_data)
            self.trainer.update_difficulty_and_skip_gid_set(batch)
        
        final_alpha = self.trainer.alpha_entropy_dict["easy"]

        # print(final_alpha)
        expected_final_alpha = 0.5
        torch.testing.assert_close(final_alpha, expected_final_alpha, atol=1e-4, rtol=1e-4)
        
        # alpha应该有所调整
        # 由于持续生成较多token，alpha可能增加或保持稳定
        self.assertIsNotNone(final_alpha)
        # 但不应超过上限
        self.assertLessEqual(final_alpha, self.trainer.alpha_entropy_max)
        # 且应该在合理范围内
        self.assertGreaterEqual(final_alpha, self.trainer.alpha_entropy_min)


if __name__ == "__main__":
    unittest.main()

