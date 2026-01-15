import unittest
import sys
import os
import json

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import detection_score


class TestDetectionScore(unittest.TestCase):
    """测试detection_score函数 - VLM-R1文档描述的目标检测综合得分计算"""
    
    def test_perfect_match(self):
        """测试完美匹配的情况"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 完美匹配：position_score=1.0, label_score=1.0, completeness_score=1.0
        # final_score = (0.7*1.0 + 0.0*1.0 + 0.3*1.0) / 1.0 = 1.0
        self.assertAlmostEqual(score, 1.0, places=2)
    
    def test_partial_match(self):
        """测试部分匹配的情况"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 1个匹配，1个误检
        # position_score = 1.0/1 = 1.0
        # completeness_score = 1.0 - (0 + 1/2) / 2 = 0.75
        # final_score = (0.7*1.0 + 0.3*0.75) / 1.0 = 0.925
        self.assertGreater(score, 0.8)
        self.assertLess(score, 1.0)
    
    def test_no_match(self):
        """测试完全不匹配的情况"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [500, 500, 600, 600], "label": "person"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol, iou_threshold=0.5)
        
        # 完全不匹配：IoU < 0.5，没有匹配对
        # position_score = 0, completeness_score很低
        self.assertLess(score, 0.5)
    
    def test_empty_prediction(self):
        """测试空预测的情况"""
        pred_boxes = []
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 空预测，全部漏检
        self.assertEqual(score, 0.0)
    
    def test_empty_ground_truth_empty_prediction(self):
        """测试空真实框和空预测的情况"""
        pred_boxes = []
        gt_boxes = []
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 都为空，完美匹配
        self.assertEqual(score, 1.0)
    
    def test_empty_ground_truth_with_prediction(self):
        """测试空真实框但有预测的情况"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = []
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 有误检
        self.assertEqual(score, 0.0)
    
    def test_label_mismatch(self):
        """测试标签不匹配的情况"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "person"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol, alpha=0.5, beta=0.5, gamma=0.0)
        
        # IoU很高但标签不匹配
        # position_score = 0 (因为label_correct=False时iou设为0)
        # label_score = 0
        # 得分应该很低
        self.assertLess(score, 0.5)
    
    def test_iou_threshold(self):
        """测试IoU阈值的影响"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [150, 150, 250, 250], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        # 使用较低的阈值
        score_low = detection_score(content, sol, iou_threshold=0.1)
        # 使用较高的阈值
        score_high = detection_score(content, sol, iou_threshold=0.9)
        
        # 低阈值应该能匹配，高阈值可能无法匹配
        self.assertGreater(score_low, score_high)
    
    def test_custom_weights(self):
        """测试自定义权重"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        # 只考虑位置
        score_position = detection_score(content, sol, alpha=1.0, beta=0.0, gamma=0.0)
        # 只考虑完整性
        score_completeness = detection_score(content, sol, alpha=0.0, beta=0.0, gamma=1.0)
        
        # 完美匹配时，两者都应该是1.0
        self.assertAlmostEqual(score_position, 1.0, places=2)
        self.assertAlmostEqual(score_completeness, 1.0, places=2)
    
    def test_multiple_objects(self):
        """测试多个目标的情况"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"},
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"},
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 所有目标都完美匹配
        self.assertAlmostEqual(score, 1.0, places=2)
    
    def test_greedy_matching(self):
        """测试贪婪匹配算法"""
        # 测试贪婪匹配是否选择最佳匹配
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [110, 110, 210, 210], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 应该选择IoU最高的匹配
        # 1个匹配，1个误检
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)
    
    def test_invalid_json_format(self):
        """测试无效JSON格式"""
        content = "invalid json"
        sol = f"```json\n{json.dumps([{'bbox_2d': [100, 100, 200, 200], 'label': 'car'}])}\n```"
        
        score = detection_score(content, sol)
        
        # 无效格式应该返回0.0
        self.assertEqual(score, 0.0)
    
    def test_missing_bbox_field(self):
        """测试缺少bbox_2d字段"""
        pred_boxes = [
            {"label": "car"}  # 缺少bbox_2d
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 缺少必要字段，应该处理异常
        self.assertLessEqual(score, 0.5)
    
    def test_missing_label_field(self):
        """测试缺少label字段"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200]}  # 缺少label
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 缺少label字段，应该处理异常
        self.assertGreaterEqual(score, 0.0)
    
    def test_case_insensitive_labels(self):
        """测试标签大小写不敏感"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "CAR"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol, alpha=0.0, beta=1.0, gamma=0.0)
        
        # 标签应该转换为小写进行比较
        self.assertAlmostEqual(score, 1.0, places=2)


class TestDetectionScoreIntegration(unittest.TestCase):
    """集成测试：测试detection_score在实际场景中的表现"""
    
    def test_object_detection_scenario(self):
        """测试目标检测场景"""
        # 模拟一个真实的目标检测任务
        pred_boxes = [
            {"bbox_2d": [50, 50, 150, 150], "label": "car"},
            {"bbox_2d": [200, 200, 300, 300], "label": "person"},
            {"bbox_2d": [400, 400, 500, 500], "label": "bike"}
        ]
        gt_boxes = [
            {"bbox_2d": [50, 50, 150, 150], "label": "car"},
            {"bbox_2d": [200, 200, 300, 300], "label": "person"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # 2个正确，1个误检
        self.assertGreater(score, 0.7)
        self.assertLess(score, 1.0)
    
    def test_quality_control_scenario(self):
        """测试质量控制场景"""
        # 测试不同质量的检测结果
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        
        # 好的检测
        good_pred = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        
        # 差的检测
        bad_pred = [
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        good_score = detection_score(f"```json\n{json.dumps(good_pred)}\n```", sol)
        bad_score = detection_score(f"```json\n{json.dumps(bad_pred)}\n```", sol)
        
        # 好的检测应该得分更高
        self.assertGreater(good_score, bad_score)
        self.assertGreater(good_score, 0.9)
        self.assertLess(bad_score, 0.5)


if __name__ == "__main__":
    unittest.main()
