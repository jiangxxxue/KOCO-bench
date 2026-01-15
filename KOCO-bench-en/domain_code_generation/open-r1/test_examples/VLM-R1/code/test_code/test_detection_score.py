import unittest
import sys
import os
import json

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import detection_score


class TestDetectionScore(unittest.TestCase):
    """Test detection_score function - comprehensive object detection score calculation as described in VLM-R1 documentation"""
    
    def test_perfect_match(self):
        """Test perfect match case"""
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
        
        # Perfect match: position_score=1.0, label_score=1.0, completeness_score=1.0
        # final_score = (0.7*1.0 + 0.0*1.0 + 0.3*1.0) / 1.0 = 1.0
        self.assertAlmostEqual(score, 1.0, places=4)
    
    def test_partial_match(self):
        """Test partial match case"""
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
        
        # print(score)

        expected_score = 0.9249999999999999
        # 1 match, 1 false positive
        # position_score = 1.0/1 = 1.0
        # completeness_score = 1.0 - (0 + 1/2) / 2 = 0.75
        # final_score = (0.7*1.0 + 0.3*0.75) / 1.0 = 0.925
        self.assertAlmostEqual(score, expected_score, places=4)
        self.assertLess(score, 1.0)
    
    def test_no_match(self):
        """Test complete mismatch case"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [500, 500, 600, 600], "label": "person"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol, iou_threshold=0.5)
        
        # print(score)

        expected_score = 0.0
        # Complete mismatch: IoU < 0.5, no matching pairs
        # position_score = 0, completeness_score very low
        self.assertAlmostEqual(score, expected_score, places=4)
    
    def test_empty_prediction(self):
        """Test empty prediction case"""
        pred_boxes = []
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # Empty prediction, all false negatives
        self.assertEqual(score, 0.0)
    
    def test_empty_ground_truth_empty_prediction(self):
        """Test empty ground truth and empty prediction case"""
        pred_boxes = []
        gt_boxes = []
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # Both empty, perfect match
        self.assertEqual(score, 1.0)
    
    def test_empty_ground_truth_with_prediction(self):
        """Test empty ground truth with prediction case"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = []
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # Has false positives
        self.assertEqual(score, 0.0)
    
    def test_label_mismatch(self):
        """Test label mismatch case"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "person"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol, alpha=0.5, beta=0.5, gamma=0.0)
        
        # print(score)

        expected_score = 0.0
        # High IoU but label mismatch
        # position_score = 0 (because iou is set to 0 when label_correct=False)
        # label_score = 0
        # Score should be very low
        self.assertAlmostEqual(score, expected_score, places=4)
    
    def test_iou_threshold(self):
        """Test IoU threshold impact"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [150, 150, 250, 250], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        # Use lower threshold
        score_low = detection_score(content, sol, iou_threshold=0.1)
        # Use higher threshold
        score_high = detection_score(content, sol, iou_threshold=0.9)
        
        # print(score_low, score_high)

        expected_score_low = 0.39999999999999997
        expected_score_high = 0.0
        # Low threshold should match, high threshold may not match
        self.assertAlmostEqual(score_low, expected_score_low, places=4)
        self.assertAlmostEqual(score_high, expected_score_high, places=4)
    
    def test_custom_weights(self):
        """Test custom weights"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        # Only consider position
        score_position = detection_score(content, sol, alpha=1.0, beta=0.0, gamma=0.0)
        # Only consider completeness
        score_completeness = detection_score(content, sol, alpha=0.0, beta=0.0, gamma=1.0)
        
        # For perfect match, both should be 1.0
        self.assertAlmostEqual(score_position, 1.0, places=4)
        self.assertAlmostEqual(score_completeness, 1.0, places=4)
    
    def test_multiple_objects(self):
        """Test multiple objects case"""
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
        
        # All objects perfectly matched
        self.assertAlmostEqual(score, 1.0, places=4)
    
    def test_greedy_matching(self):
        """Test greedy matching algorithm"""
        # Test if greedy matching selects the best match
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
        
        # Should select the match with highest IoU
        # 1 match, 1 false positive
        expected_score = 0.9249999999999999
        self.assertAlmostEqual(score, expected_score, places=4)
        self.assertLess(score, 1.0)
    
    def test_invalid_json_format(self):
        """Test invalid JSON format"""
        content = "invalid json"
        sol = f"```json\n{json.dumps([{'bbox_2d': [100, 100, 200, 200], 'label': 'car'}])}\n```"
        
        score = detection_score(content, sol)
        
        # Invalid format should return 0.0
        self.assertEqual(score, 0.0)
    
    def test_missing_bbox_field(self):
        """Test missing bbox_2d field"""
        pred_boxes = [
            {"label": "car"}  # Missing bbox_2d
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # print(score)

        expected_score = 0.0
        # Missing required field, should handle exception
        self.assertAlmostEqual(score, expected_score, places=4)
    
    def test_missing_label_field(self):
        """Test missing label field"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200]}  # Missing label
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol)
        
        # Missing label field, should handle exception
        expected_score = 0.3
        self.assertAlmostEqual(score, expected_score, places=4)
    
    def test_case_insensitive_labels(self):
        """Test case insensitive labels"""
        pred_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "CAR"}
        ]
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        
        content = f"```json\n{json.dumps(pred_boxes)}\n```"
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        score = detection_score(content, sol, alpha=0.0, beta=1.0, gamma=0.0)
        
        # Labels should be converted to lowercase for comparison
        self.assertAlmostEqual(score, 1.0, places=4)


class TestDetectionScoreIntegration(unittest.TestCase):
    """Integration tests: test detection_score performance in real scenarios"""
    
    def test_object_detection_scenario(self):
        """Test object detection scenario"""
        # Simulate a real object detection task
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
        
        # print(score)
        
        expected_score = 0.95
        # 2 correct, 1 false positive
        self.assertAlmostEqual(score, expected_score, places=4)
    
    def test_quality_control_scenario(self):
        """Test quality control scenario"""
        # Test different quality detection results
        gt_boxes = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        
        # Good detection
        good_pred = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        
        # Bad detection
        bad_pred = [
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        
        sol = f"```json\n{json.dumps(gt_boxes)}\n```"
        
        good_score = detection_score(f"```json\n{json.dumps(good_pred)}\n```", sol)
        bad_score = detection_score(f"```json\n{json.dumps(bad_pred)}\n```", sol)
        
        # print(good_score, bad_score)

        expected_good_score = 1.0
        expected_bad_score = 0.0
        # Good detection should score higher
        self.assertAlmostEqual(good_score, expected_good_score, places=4)
        self.assertAlmostEqual(bad_score, expected_bad_score, places=4)


if __name__ == "__main__":
    unittest.main()
