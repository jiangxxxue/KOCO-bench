import unittest
import sys
import os
import torch
# Add the parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual function we want to test
from examples.reward_function.aepo import compute_score


class TestComputeScore(unittest.TestCase):
    """Test compute_score function - Hierarchical entropy reward calculation described in ARES document Section 3"""
    
    def setUp(self):
        """Set up test environment"""
        pass
    
    def test_compute_score_basic_structure(self):
        """Test basic input/output structure"""
        reward_inputs = [
            {
                "response": "<think>Reasoning process</think>\\boxed{42}",
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
        # Verify output structure
        # self.assertEqual(len(scores), 1)
        # self.assertIn("overall", scores[0])
        # self.assertIn("accuracy", scores[0])
        # self.assertIn("format", scores[0])
        # self.assertIn("high_entropy_token_num_score", scores[0])
        
        # # Verify all scores are finite numbers
        # self.assertTrue(isinstance(scores[0]["overall"], float))
        # self.assertTrue(isinstance(scores[0]["accuracy"], float))
        # self.assertIn(scores[0]["accuracy"], [0.0, 1.0])
    
    def test_compute_score_invalid_input(self):
        """Test invalid input"""
        # Test that non-list input should raise ValueError
        with self.assertRaises(ValueError):
            compute_score({"response": "test"})
    
    def test_compute_score_easy_correct_in_tolerance(self):
        """Test Easy task, correct answer, HWE token count within tolerance band"""
        reward_inputs = [
            {
                "response": "<think>Simple reasoning</think>\\boxed{100}",
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
        # Easy task, correct answer, delta=0 within tolerance band, entropy reward should be close to 0 or very small
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], -0.1)
        # self.assertLessEqual(scores[0]["high_entropy_token_num_score"], 0.1)
        # # overall = accuracy + entropy_score ≈ 1.0
        # self.assertAlmostEqual(scores[0]["overall"], 1.0, delta=0.2)
    
    def test_compute_score_easy_correct_over_thinking(self):
        """Test Easy task, correct answer but over-thinking (excessive HWE token count)"""
        reward_inputs = [
            {
                "response": "<think>Very long reasoning process</think>\\boxed{50}",
                "ground_truth": "50",
                "difficulty": "easy",
                "high_entropy_token_num": 30,  # Far exceeds target
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
        # Easy task, correct but over-thinking, should apply penalty
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertLess(scores[0]["high_entropy_token_num_score"], 0.0)  # Negative reward (penalty)
        # # overall = 1.0 + negative reward < 1.0
        # self.assertLess(scores[0]["overall"], 1.0)
    
    def test_compute_score_easy_incorrect_some_exploration(self):
        """Test Easy task, incorrect answer but with some exploration"""
        reward_inputs = [
            {
                "response": "<think>Incorrect reasoning</think>\\boxed{999}",
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
        # Easy task, incorrect answer, but with exploration, give small encouragement
        # self.assertEqual(scores[0]["accuracy"], 0.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)  # Give encouragement
        # # overall = 0.0 + positive reward
        # self.assertGreater(scores[0]["overall"], 0.0)
        # self.assertLess(scores[0]["overall"], 0.5)  # Small encouragement magnitude
    
    def test_compute_score_medium_correct_in_tolerance(self):
        """Test Medium task, correct answer, HWE token count within tolerance band"""
        reward_inputs = [
            {
                "response": "<think>Medium difficulty reasoning</think>\\boxed{200}",
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
        # Medium task, correct answer, delta=0 within tolerance band, entropy reward should be close to 0
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], -0.1)
        # self.assertLessEqual(scores[0]["high_entropy_token_num_score"], 0.1)
        # self.assertAlmostEqual(scores[0]["overall"], 1.0, delta=0.2)
    
    def test_compute_score_medium_correct_deviation(self):
        """Test Medium task, correct answer but large deviation (too many or too few)"""
        # Test excessive case
        reward_inputs_over = [
            {
                "response": "<think>Excessive reasoning</think>\\boxed{300}",
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
        # self.assertLess(scores_over[0]["high_entropy_token_num_score"], 0.0)  # Penalty
        
        # Test insufficient case
        reward_inputs_under = [
            {
                "response": "<think>Brief</think>\\boxed{300}",
                "ground_truth": "300",
                "difficulty": "medium",
                "high_entropy_token_num": 5,
                "target_high_entropy_token_num": 20,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_under = compute_score(reward_inputs_under)
        self.assertEqual(scores_under[0]["accuracy"], 1.0)
        self.assertLess(scores_under[0]["high_entropy_token_num_score"], 0.0)  # Penalty
    
    def test_compute_score_medium_incorrect_exploration(self):
        """Test Medium task, incorrect answer but with exploration"""
        reward_inputs = [
            {
                "response": "<think>Incorrect reasoning process</think>\\boxed{999}",
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
        # Medium task, incorrect answer but with exploration, give medium encouragement
        self.assertEqual(scores[0]["accuracy"], 0.0)
        self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)
        # Encouragement magnitude should be larger than Easy
        self.assertGreater(scores[0]["overall"], 0.0)
    
    def test_compute_score_hard_correct_sufficient_exploration(self):
        """Test Hard task, correct answer with sufficient exploration"""
        reward_inputs = [
            {
                "response": "<think>Deep detailed reasoning process</think>\\boxed{500}",
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
        
        # # Hard task, correct with sufficient exploration, should give positive reward
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)  # Positive reward
        # # overall = 1.0 + positive reward > 1.0
        # self.assertGreater(scores[0]["overall"], 1.0)
    
    def test_compute_score_hard_correct_insufficient_exploration(self):
        """Test Hard task, correct answer but insufficient exploration"""
        reward_inputs = [
            {
                "response": "<think>Brief</think>\\boxed{500}",
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
        # Hard task, correct but insufficient exploration, should apply light penalty
        # self.assertEqual(scores[0]["accuracy"], 1.0)
        # self.assertLess(scores[0]["high_entropy_token_num_score"], 0.0)  # Light penalty
        # # overall = 1.0 + negative reward
        # self.assertLess(scores[0]["overall"], 1.0)
        # self.assertGreater(scores[0]["overall"], 0.5)  # But still relatively high
    
    def test_compute_score_hard_incorrect_with_exploration(self):
        """Test Hard task, incorrect answer but with sufficient exploration"""
        reward_inputs = [
            {
                "response": "<think>Extensive incorrect reasoning</think>\\boxed{999}",
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
        # Hard task, incorrect answer but with sufficient exploration, give encouragement
        self.assertEqual(scores[0]["accuracy"], 0.0)
        self.assertGreaterEqual(scores[0]["high_entropy_token_num_score"], 0.0)
        # Encouragement magnitude should be large
        self.assertGreater(scores[0]["overall"], 0.0)
    
    def test_compute_score_batch_processing(self):
        """Test batch processing of multiple samples"""
        reward_inputs = [
            {
                "response": "<think>Reasoning 1</think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            },
            {
                "response": "<think>Reasoning 2</think>\\boxed{200}",
                "ground_truth": "200",
                "difficulty": "medium",
                "high_entropy_token_num": 20,
                "target_high_entropy_token_num": 20,
                "alpha_entropy": 0.5
            },
            {
                "response": "<think>Reasoning 3</think>\\boxed{300}",
                "ground_truth": "300",
                "difficulty": "hard",
                "high_entropy_token_num": 40,
                "target_high_entropy_token_num": 40,
                "alpha_entropy": 0.6
            }
        ]
        
        scores = compute_score(reward_inputs)
        
        # Verify batch processing
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
        # Verify results for each sample
        # for score in scores:
        #     self.assertIn("overall", score)
        #     self.assertIn("accuracy", score)
        #     self.assertIn("high_entropy_token_num_score", score)
        #     self.assertEqual(score["accuracy"], 1.0)
    
    def test_compute_score_different_alpha_entropy(self):
        """Test effect of different alpha_entropy values"""
        base_input = {
            "response": "<think>Reasoning process</think>\\boxed{100}",
            "ground_truth": "999",  # Incorrect answer
            "difficulty": "easy",
            "high_entropy_token_num": 20,
            "target_high_entropy_token_num": 10
        }
        
        # Small alpha
        input_small_alpha = {**base_input, "alpha_entropy": 0.1}
        scores_small = compute_score([input_small_alpha])
        
        # Large alpha
        input_large_alpha = {**base_input, "alpha_entropy": 1.0}
        scores_large = compute_score([input_large_alpha])

        
        # print(scores_small[0]["high_entropy_token_num_score"])
        # print(scores_large[0]["high_entropy_token_num_score"])
        expected_high_entropy_token_num_score_small = 0.05915618237740526
        expected_high_entropy_token_num_score_large = 0.5915618237740526
        torch.testing.assert_close(scores_small[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score_small, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(scores_large[0]["high_entropy_token_num_score"], expected_high_entropy_token_num_score_large, atol=1e-5, rtol=1e-5)
        
        # Larger alpha should have greater impact on entropy reward/penalty
        entropy_score_small = abs(scores_small[0]["high_entropy_token_num_score"])
        entropy_score_large = abs(scores_large[0]["high_entropy_token_num_score"])
        # self.assertLess(entropy_score_small, entropy_score_large)
    
    def test_compute_score_format_detection(self):
        """Test format detection functionality"""
        # Correct format
        reward_inputs_correct = [
            {
                "response": "<think>Reasoning</think>\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_correct = compute_score(reward_inputs_correct)
        self.assertEqual(scores_correct[0]["format"], 1.0)
        
        # Incorrect format (missing think tag)
        reward_inputs_wrong = [
            {
                "response": "Direct answer\\boxed{100}",
                "ground_truth": "100",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores_wrong = compute_score(reward_inputs_wrong)
        # Incorrect format, format should be 0
        self.assertEqual(scores_wrong[0]["format"], 0.0)
    
    def test_compute_score_accuracy_extraction(self):
        """Test automatic accuracy extraction functionality"""
        # Don't provide accuracy, let function extract automatically
        reward_inputs = [
            {
                "response": "<think>Reasoning</think>\\boxed{42}",
                "ground_truth": "42",
                "difficulty": "easy",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10,
                "alpha_entropy": 0.5
            }
        ]
        
        scores = compute_score(reward_inputs)
        
        # Should automatically determine as correct
        self.assertEqual(scores[0]["accuracy"], 1.0)
        
        # Test incorrect answer
        reward_inputs_wrong = [
            {
                "response": "<think>Reasoning</think>\\boxed{999}",
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
        """Test edge cases"""
        # Test HWE token count of 0
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
        # self.assertTrue(abs(scores_zero[0]["overall"]) < 100)  # Reasonable range
        
        # Test target of 0
        reward_inputs_target_zero = [
            {
                "response": "<think>Reasoning</think>\\boxed{100}",
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
        """Test default parameters"""
        # Don't provide difficulty information
        reward_inputs = [
            {
                "response": "<think>Reasoning</think>\\boxed{100}",
                "ground_truth": "100",
                "high_entropy_token_num": 10,
                "target_high_entropy_token_num": 10
            }
        ]
        
        # Use default alpha_entropy
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
    """Integration test: test complex scenarios of compute_score"""
    
    def test_difficulty_progression(self):
        """Test reward changes across different difficulties"""
        # Same conditions, different difficulties
        base_correct = {
            "response": "<think>Sufficient reasoning</think>\\boxed{100}",
            "ground_truth": "100",
            "high_entropy_token_num": 30,
            "target_high_entropy_token_num": 20,
            "alpha_entropy": 0.5
        }
        
        # Easy: over-thinking will be penalized
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
        # Medium: deviation will be penalized
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
        # Hard: sufficient exploration will be rewarded
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
        # # Verify trend: Hard should be highest, Easy should be penalized
        # self.assertGreater(hard_scores[0]["overall"], medium_scores[0]["overall"])
        # self.assertLess(easy_scores[0]["overall"], 1.0)  # Easy penalized
        # self.assertGreater(hard_scores[0]["overall"], 1.0)  # Hard rewarded
    
    def test_accuracy_dominant_factor(self):
        """Test that accuracy is the dominant factor"""
        # Correct answer, even with negative entropy reward, overall is still high
        correct_input = [
            {
                "response": "<think>Excessive reasoning</think>\\boxed{100}",
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
        
        # Incorrect answer, even with positive entropy reward, overall is still low
        incorrect_input = [
            {
                "response": "<think>Sufficient reasoning</think>\\boxed{999}",
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
        # Correct answer's overall should be greater than incorrect answer
        self.assertGreater(correct_scores[0]["overall"], incorrect_scores[0]["overall"])
    
    def test_margin_tolerance(self):
        """Test tolerance band mechanism"""
        # Test at tolerance band edge
        base_input = {
            "response": "<think>Reasoning</think>\\boxed{100}",
            "ground_truth": "100",
            "difficulty": "easy",
            "target_high_entropy_token_num": 100,
            "alpha_entropy": 0.5
        }
        
        # Exactly at target value
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
        # Slightly exceeds (within tolerance band)
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
        # Far exceeds (beyond tolerance band)
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
        # Verify: penalty within tolerance band should be very small or 0
        self.assertAlmostEqual(exact_scores[0]["high_entropy_token_num_score"], 0.0, delta=0.1)
        self.assertGreater(slightly_over_scores[0]["high_entropy_token_num_score"], 
                          far_over_scores[0]["high_entropy_token_num_score"])


if __name__ == "__main__":
    unittest.main()

