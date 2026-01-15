from math import e
import unittest
import sys
import os
import json

# Add the parent directory to Python path to import open_r1 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'open-r1-multimodal', 'src'))

# Import the actual function we want to test
from open_r1.grpo_jsonl import repetition_reward


class TestRepetitionReward(unittest.TestCase):
    """Test repetition_reward function - repetition penalty as described in VLM-R1 documentation"""
    
    def test_no_repetition_json(self):
        """Test JSON format with no repetition"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"},
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # No repetition, should return 0.0
        self.assertEqual(reward, 0.0)
    
    def test_full_repetition_json(self):
        """Test JSON format with full repetition"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        expected_reward = -0.6666666666666667
        # Full repetition, should be close to -1.0
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_partial_repetition_json(self):
        """Test JSON format with partial repetition"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        expected_reward = -0.3333333333333333
        # Partial repetition, should have some penalty
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_no_repetition_text(self):
        """Test plain text with no repetition"""
        content = "This is a unique sentence with no repeated phrases at all"
        
        reward = repetition_reward(content)
        
        # No repetition, should return 0.0 or close to 0.0
        expected_reward = 0.0
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_full_repetition_text(self):
        """Test plain text with full repetition"""
        phrase = "repeat this phrase "
        content = phrase * 20  # Repeat 20 times
        
        reward = repetition_reward(content)
        
        # Full repetition, should be close to -1.0
        expected_reward = -0.9454545454545454
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_partial_repetition_text(self):
        """Test plain text with partial repetition"""
        content = "some unique text " * 3 + "different content here"
        
        reward = repetition_reward(content)

        # Partial repetition, should have some penalty
        expected_reward = -0.1428571428571429
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_empty_content(self):
        """Test empty content"""
        content = ""
        
        reward = repetition_reward(content)
        
        # Empty content should return 0.0
        self.assertEqual(reward, 0.0)
    
    def test_short_text(self):
        """Test short text (less than ngram_size)"""
        content = "short"  # Only 1 word, less than 6-gram
        
        reward = repetition_reward(content)
        
        # Short text should return 0.0
        self.assertEqual(reward, 0.0)
    
    def test_json_without_markers(self):
        """Test JSON without ```json markers"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = json.dumps(data)
        
        reward = repetition_reward(content)
        
        # Should be able to recognize JSON and detect repetition
        expected_reward = -0.5 
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_json_with_generic_markers(self):
        """Test JSON with generic ``` markers"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = f"```\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # Should be able to recognize JSON and detect repetition
        expected_reward = -0.5
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_invalid_json(self):
        """Test invalid JSON (fallback to text processing)"""
        content = "```json\ninvalid json content\n```"
        
        reward = repetition_reward(content)
        
        expected_reward = 0.0
        # Invalid JSON should fallback to text processing
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_json_missing_fields(self):
        """Test JSON with missing required fields"""
        data = [
            {"bbox_2d": [100, 100, 200, 200]},  # Missing label
            {"label": "car"}  # Missing bbox_2d
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        # Missing fields should fallback to text processing
        expected_reward = 0.0
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_mixed_content(self):
        """Test mixed content (JSON + text)"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        content = f"Some text before\n```json\n{json.dumps(data)}\n```\nSome text after"
        
        reward = repetition_reward(content)
        
        expected_reward = -0.0
        # Should be able to extract JSON part
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_case_sensitivity_text(self):
        """Test text case sensitivity"""
        # Need long enough text to form 6-gram
        content = "REPEAT REPEAT repeat repeat word1 word2 word3 word4"
        
        reward = repetition_reward(content)
        
        # After converting to lowercase should detect repetition
        expected_reward = -0.0
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_ngram_size_6(self):
        """Test 6-gram detection"""
        # Create text containing repeated 6-grams
        base = "word1 word2 word3 word4 word5 word6 "
        content = base * 5 + "different words here now"
        
        reward = repetition_reward(content)

        # Should detect 6-gram repetition
        expected_reward = -0.6551724137931034
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_different_labels_same_bbox(self):
        """Test same bbox but different labels"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "person"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        
        expected_reward = -0.0
        # Same bbox but different label, not full repetition
        self.assertAlmostEqual(reward, expected_reward, places=4)
    
    def test_same_label_different_bbox(self):
        """Test same label but different bbox"""
        data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "car"}
        ]
        content = f"```json\n{json.dumps(data)}\n```"
        
        reward = repetition_reward(content)
        expected_reward = -0.0
        # Same label but different bbox, not repetition 
        self.assertAlmostEqual(reward, expected_reward, places=4)


class TestRepetitionRewardIntegration(unittest.TestCase):
    """Integration tests: test repetition_reward performance in real scenarios"""
    
    def test_quality_control_scenario(self):
        """Test quality control scenario"""
        # Good generation (no repetition)
        good_data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [300, 300, 400, 400], "label": "person"},
            {"bbox_2d": [500, 500, 600, 600], "label": "bike"}
        ]
        good_content = f"```json\n{json.dumps(good_data)}\n```"
        
        # Bad generation (with repetition)
        bad_data = [
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"},
            {"bbox_2d": [100, 100, 200, 200], "label": "car"}
        ]
        bad_content = f"```json\n{json.dumps(bad_data)}\n```"
        
        good_reward = repetition_reward(good_content)
        bad_reward = repetition_reward(bad_content)
        
        expected_good_reward = -0.0
        expected_bad_reward = -0.6666666666666667
        # Good generation should score higher (smaller penalty)
        self.assertAlmostEqual(good_reward, expected_good_reward, places=4)
        self.assertAlmostEqual(bad_reward, expected_bad_reward, places=4)
    
    def test_text_generation_scenario(self):
        """Test text generation scenario"""
        # Good generation (diverse)
        good_text = "The cat sat on the mat. The dog played in the yard. Birds flew overhead."
        
        # Bad generation (repetitive)
        bad_text = "I think I think I think I think I think I think I think"
        
        good_reward = repetition_reward(good_text)
        bad_reward = repetition_reward(bad_text)
        
        expected_good_reward = -0.0
        expected_bad_reward = -0.7777777777777778
        # Good generation should score higher
        self.assertAlmostEqual(good_reward, expected_good_reward, places=4)
        self.assertAlmostEqual(bad_reward, expected_bad_reward, places=4)
    
    def test_filter_repetitive_outputs(self):
        """Test filtering repetitive outputs"""
        outputs = [
            f"```json\n{json.dumps([{'bbox_2d': [i*100, i*100, (i+1)*100, (i+1)*100], 'label': f'obj{i}'} for i in range(3)])}\n```",
            f"```json\n{json.dumps([{'bbox_2d': [100, 100, 200, 200], 'label': 'car'}] * 5)}\n```",
            f"```json\n{json.dumps([{'bbox_2d': [i*50, i*50, (i+1)*50, (i+1)*50], 'label': 'item'} for i in range(4)])}\n```"
        ]
        
        rewards = [repetition_reward(output) for output in outputs]
        
        # print(rewards)

        expected_rewards =[-0.0, -0.8, -0.0]

        for i, reward in enumerate(rewards):
            self.assertAlmostEqual(reward, expected_rewards[i], places=4)
        # Can be used for filtering: only keep outputs with reward close to 0



if __name__ == "__main__":
    unittest.main()
