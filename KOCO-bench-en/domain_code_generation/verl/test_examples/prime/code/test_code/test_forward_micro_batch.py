import unittest
import torch
import sys
import os
from unittest.mock import MagicMock
from omegaconf import OmegaConf

# Add the recipe directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'recipe'))

# Import ground truth class
from prime.prime_dp_rm import DataParallelPRIMERewardModel


class TestForwardMicroBatch(unittest.TestCase):
    """Test _forward_micro_batch function - implicit process reward computation described in Section 1 of documentation"""
    
    def setUp(self):
        torch.manual_seed(42)
        
        # Create mock models
        self.vocab_size = 100
        self.hidden_size = 64
        
        # Create mock reward model
        self.reward_model = self._create_mock_model()
        self.ref_model = self._create_mock_model()
        
        # Create mock optimizer
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-4)
        
        # Create minimal config
        self.config = OmegaConf.create({
            'model': {
                'use_remove_padding': False,
                'use_fused_kernels': False,
                'beta_train': 0.05
            },
            'lambda': 0.0,
            'prime_use_gt': False,
            'prime_granularity': 'token',
            'ulysses_sequence_parallel_size': 1
        })
        
        # Create DataParallelPRIMERewardModel instance
        self.rm_instance = DataParallelPRIMERewardModel(
            config=self.config,
            reward_module=self.reward_model,
            ref_module=self.ref_model,
            reward_optimizer=self.optimizer
        )
    
    def _create_mock_model(self):
        """Create a simple mock model for testing"""
        class MockModel(torch.nn.Module):
            def __init__(self, vocab_size, hidden_size):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
                self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
                
            def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, return_dict=False):
                hidden = self.embedding(input_ids)
                logits = self.lm_head(hidden)
                
                if return_dict:
                    output = MagicMock()
                    output.logits = logits
                    return output
                else:
                    output = MagicMock()
                    output.logits = logits
                    return output
        
        return MockModel(self.vocab_size, self.hidden_size)
    
    def test_forward_micro_batch_basic(self):
        """Test _forward_micro_batch basic functionality - using ground truth code"""
        batch_size, seq_len = 2, 10
        prompt_length = 3
        
        # Create test input - use long type to ensure correct indexing
        micro_batch = {
            "input_ids": torch.tensor([[36, 98, 30, 77, 38, 37, 79, 84, 67, 82],
        [64, 92, 99, 41, 84, 69, 23, 51,  9, 15]]),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
            "acc": torch.tensor([0.8, 0.6])
        }

        # print(micro_batch["input_ids"])
        # print(micro_batch["position_ids"])
        
        
        # Call ground truth function
        token_level_scores, q_values = self.rm_instance._forward_micro_batch(micro_batch, prompt_length)

        # print(token_level_scores)
        # print(f"token_level_scores: {token_level_scores[0][0]:.30f}")
        # print(f"token_level_scores: {token_level_scores[0][1]:.30f}")
        # print(f"token_level_scores: {token_level_scores[0][2]:.30f}")
        # print(f"token_level_scores: {token_level_scores[0][3]:.30f}")
        # print(f"token_level_scores: {token_level_scores[0][4]:.30f}")
        # print(f"token_level_scores: {token_level_scores[0][5]:.30f}")
        # print(f"token_level_scores: {token_level_scores[0][6]:.30f}")
        # print(q_values)
        # print(f"q_values: {q_values[0][0]:.30f}")
        # print(f"q_values: {q_values[0][1]:.30f}")
        # print(f"q_values: {q_values[0][2]:.30f}")
        # print(f"q_values: {q_values[0][3]:.30f}")
        # print(f"q_values: {q_values[0][4]:.30f}")
        # print(f"q_values: {q_values[0][5]:.30f}")
        # print(f"q_values: {q_values[0][6]:.30f}")
        # print(token_level_scores)
        # print(f"token_level_scores: {token_level_scores[1][0]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][1]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][2]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][3]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][4]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][5]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][6]:.30f}")
        # print(q_values)
        # print(f"q_values: {q_values[1][0]:.30f}")
        # print(f"q_values: {q_values[1][1]:.30f}")
        # print(f"q_values: {q_values[1][2]:.30f}")
        # print(f"q_values: {q_values[1][3]:.30f}")
        # print(f"q_values: {q_values[1][4]:.30f}")
        # print(f"q_values: {q_values[1][5]:.30f}")
        # print(f"q_values: {q_values[1][6]:.30f}")
        
        expected_token_level_scores = torch.tensor([[-0.020435070618987083435058593750,  0.014614701271057128906250000000,  -0.011621952056884765625000000000,  0.005395698826760053634643554688,  0.020277500152587890625000000000,  0.006968331523239612579345703125,  0.0000],
        [-0.026112342253327369689941406250,  0.021826243028044700622558593750,  0.033189535140991210937500000000,  -0.052226688712835311889648437500, -0.006393861956894397735595703125, -0.031713128089904785156250000000,  0.0000]])
        expected_q_values = torch.tensor([[ -0.408701419830322265625000000000,   0.292294025421142578125000000000, -0.232439041137695312500000000000,  0.107913970947265625000000000000,  0.405550003051757812500000000000,  0.139366626739501953125000000000, -0.625825405120849609375000000000],
        [ -0.522246837615966796875000000000,  0.436524868011474609375000000000,  0.663790702819824218750000000000, -1.044533729553222656250000000000,  -0.127877235412597656250000000000,  -0.634262561798095703125000000000,  1.390359401702880859375000000000]])
        torch.testing.assert_close(token_level_scores, expected_token_level_scores, atol=1e-30, rtol=1e-30)
        torch.testing.assert_close(q_values, expected_q_values, atol=1e-30, rtol=1e-30)
    
    def test_gae_mode_processing(self):
        """Test GAE mode process reward generation - using ground truth code"""
        batch_size, seq_len = 2, 8
        prompt_length = 2
        
        # Configure GAE mode
        self.config.lambda_ = 0.95
        self.config.prime_granularity = 'token'
        
        # Create test input, first sequence has padding - use long type
        micro_batch = {
            "input_ids": torch.tensor([[36, 98, 30, 77, 38, 37, 79, 84],
        [67, 82, 64, 92, 99, 41, 84, 69]]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 0, 0],  # First sequence has padding
                [1, 1, 1, 1, 1, 1, 1, 1]   # Second sequence is fully valid
            ], dtype=torch.long),
            "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
            "acc": torch.tensor([0.7, 0.9])
        }
        
        # print(micro_batch["input_ids"])
        # Call ground truth function
        token_level_scores, q_values = self.rm_instance._forward_micro_batch(micro_batch, prompt_length)
        
        # print(token_level_scores)
        # print(f"token_level_scores: {token_level_scores[1][0]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][1]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][2]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][3]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][4]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][5]:.30f}")
        # print(q_values)
        # print(f"q_values: {q_values[1][0]:.30f}")
        # print(f"q_values: {q_values[1][1]:.30f}")
        # print(f"q_values: {q_values[1][2]:.30f}")
        # print(f"q_values: {q_values[1][3]:.30f}")
        # print(f"q_values: {q_values[1][4]:.30f}")
        # print(f"q_values: {q_values[1][5]:.30f}")


        expected_token_level_scores = torch.tensor([[ 0.010057640261948108673095703125,-0.020435070618987083435058593750,  0.014614701271057128906250000000,  0.0000,  0.0000,  0.0000],
        [ -0.066051840782165527343750000000,  0.027718162164092063903808593750,  0.081534914672374725341796875000, -0.026112342253327369689941406250,   0.021826243028044700622558593750,  0.0000]])
        expected_q_values = torch.tensor([[  0.201152801513671875000000000000, -0.408701419830322265625000000000,  0.292294025421142578125000000000, -0.232439041137695312500000000000,  0.0000,  0.0000],
        [-1.321036815643310546875000000000,   0.554363250732421875000000000000,  1.630698204040527343750000000000, -0.522246837615966796875000000000,  0.436524868011474609375000000000,   0.663790702819824218750000000000]])
        torch.testing.assert_close(token_level_scores, expected_token_level_scores, atol=1e-30, rtol=1e-30)
        torch.testing.assert_close(q_values, expected_q_values, atol=1e-30, rtol=1e-30)   


    def test_different_prompt_lengths(self):
        """Test handling of different prompt lengths - using ground truth code"""
        batch_size, seq_len = 2, 12

        token_level_scores_list = []
        q_values_list = []
        
        # Test different prompt lengths
        for prompt_length in [2, 4, 6]:
            with self.subTest(prompt_length=prompt_length):
                micro_batch = {
                    "input_ids": torch.tensor([[36, 98, 30, 77, 38, 37, 79, 84],
        [67, 82, 64, 92, 99, 41, 84, 69]]),
                    "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
                    "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
                    "acc": torch.rand(batch_size)
                }
                
                token_level_scores, q_values = self.rm_instance._forward_micro_batch(micro_batch, prompt_length)
                
                token_level_scores_list.append(token_level_scores)
                q_values_list.append(q_values)

        # print(token_level_scores_list)
        # print(q_values_list)
        # print(token_level_scores_list:.30f)
        # print(f"token_level_scores: {token_level_scores_list[2][1][0]:.30f}")
        # print(f"token_level_scores: {token_level_scores_list[2][1][1]:.30f}")
        # # print(f"token_level_scores: {token_level_scores_list[1][1][2]:.30f}")
        # # print(f"token_level_scores: {token_level_scores_list[1][1][3]:.30f}")
        # # print(f"token_level_scores: {token_level_scores_list[1][0][4]:.30f}")
        # # print(f"token_level_scores: {token_level_scores_list[1][0][5]:.30f}")
        # print(f"q_values: {q_values_list[2][1][0]:.30f}")
        # print(f"q_values: {q_values_list[2][1][1]:.30f}")
        # print(f"q_values: {q_values_list[1][1][2]:.30f}")
        # print(f"q_values: {q_values_list[1][1][3]:.30f}")
        # print(f"q_values: {q_values_list[1][0][4]:.30f}")
        # print(f"q_values: {q_values_list[1][0][5]:.30f}")
        expected_token_level_scores_list = [torch.tensor([[  0.010057640261948108673095703125, -0.020435070618987083435058593750,  0.014614701271057128906250000000, -0.011621952056884765625000000000,  0.005395698826760053634643554688,  0.020277500152587890625000000000],
        [-0.066051840782165527343750000000,  0.027718162164092063903808593750,  0.081534914672374725341796875000, -0.026112342253327369689941406250,  0.021826243028044700622558593750,   0.033189535140991210937500000000]]),
         torch.tensor([[ 0.014614701271057128906250000000, -0.011621952056884765625000000000,  0.005395698826760053634643554688,  0.020277500152587890625000000000],
        [ 0.081534914672374725341796875000,  -0.026112342253327369689941406250,  0.021826243028044700622558593750,  0.033189535140991210937500000000]]), 
        torch.tensor([[0.005395698826760053634643554688, 0.020277500152587890625000000000],
        [0.021826243028044700622558593750, 0.033189535140991210937500000000]])]
        expected_q_values_list = [torch.tensor([[  0.201152801513671875000000000000, -0.408701419830322265625000000000,  0.292294025421142578125000000000, -0.232439041137695312500000000000,  0.107913970947265625000000000000,  0.405550003051757812500000000000],
        [-1.321036815643310546875000000000,  0.554363250732421875000000000000,  1.630698204040527343750000000000,-0.522246837615966796875000000000, 0.436524868011474609375000000000,  0.663790702819824218750000000000]]),
         torch.tensor([[  0.292294025421142578125000000000,  -0.232439041137695312500000000000,  0.107913970947265625000000000000, 0.405550003051757812500000000000],
        [ 1.630698204040527343750000000000,  -0.522246837615966796875000000000, 0.436524868011474609375000000000,  0.663790702819824218750000000000]]), 
        torch.tensor([[0.107913970947265625000000000000, 0.405550003051757812500000000000],
        [0.436524868011474609375000000000,  0.663790702819824218750000000000]])]
        for i in range(len(token_level_scores_list)):
            torch.testing.assert_close(token_level_scores_list[i], expected_token_level_scores_list[i], atol=1e-30, rtol=1e-30)
            torch.testing.assert_close(q_values_list[i], expected_q_values_list[i], atol=1e-30, rtol=1e-30)


    def test_input_output_samples_basic(self):
        """Test basic input/output samples - using ground truth code"""
        batch_size, seq_len = 3, 10
        prompt_length = 3
        
        # Create test input - use long type
        micro_batch = {
            "input_ids": torch.randint(0, self.vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
            "acc": torch.tensor([0.8, 0.5, 0.3])
        }
        
        # Call ground truth function
        token_level_scores, q_values = self.rm_instance._forward_micro_batch(micro_batch, prompt_length)
        
        # Verify output
        expected_response_len = seq_len - prompt_length
        self.assertEqual(token_level_scores.shape, (batch_size, expected_response_len))
        self.assertEqual(q_values.shape, (batch_size, expected_response_len))
        self.assertTrue(torch.isfinite(token_level_scores).all())
        self.assertTrue(torch.isfinite(q_values).all())
    
    def test_reward_granularity_token(self):
        """Test token-level reward allocation - using ground truth code"""
        batch_size, seq_len = 2, 8
        prompt_length = 2
        
        # Configure token granularity
        self.config.prime_granularity = 'token'
        self.config.lambda_ = 0.0  # Do not use GAE
        
        micro_batch = {
            "input_ids": torch.tensor([[36, 98, 30, 77, 38, 37, 79, 84],
        [67, 82, 64, 92, 99, 41, 84, 69]]),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
            "acc": torch.rand(batch_size)
        }

        # print(micro_batch["input_ids"])
        
        token_level_scores, q_values = self.rm_instance._forward_micro_batch(micro_batch, prompt_length)

        # print(token_level_scores)
        # print(f"token_level_scores: {token_level_scores[1][0]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][1]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][2]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][3]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][4]:.30f}")
        # print(f"token_level_scores: {token_level_scores[1][5]:.30f}")
        # print(q_values)
        # print(f"q_values: {q_values[1][0]:.30f}")
        # print(f"q_values: {q_values[1][1]:.30f}")
        # print(f"q_values: {q_values[1][2]:.30f}")
        # print(f"q_values: {q_values[1][3]:.30f}")
        # print(f"q_values: {q_values[1][4]:.30f}")
        # print(f"q_values: {q_values[1][5]:.30f}")
        expected_token_level_scores = torch.tensor([[ 0.010057640261948108673095703125,-0.020435070618987083435058593750,  0.014614701271057128906250000000, -0.011621952056884765625000000000,  0.005395698826760053634643554688,  0.0000],
        [-0.066051840782165527343750000000,  0.027718162164092063903808593750, 0.081534914672374725341796875000, -0.026112342253327369689941406250, 0.021826243028044700622558593750,  0.0000]])

        torch.testing.assert_close(token_level_scores, expected_token_level_scores, atol=1e-30, rtol=1e-30)

        
        # Verify token granularity: each position should have a reward value (except padding)
        self.assertTrue((token_level_scores != 0).any())
    

    
    def test_padding_handling(self):
        """Test padding handling - using ground truth code"""
        batch_size, seq_len = 2, 10
        prompt_length = 2
        
        # Create input with padding - use long type
        micro_batch = {
            "input_ids": torch.tensor([[36, 98, 30, 77, 38, 37, 79, 84, 67, 82],
        [64, 92, 99, 41, 84, 69, 23, 51,  9, 15]]),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # First sequence has padding
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # Second sequence is fully valid
            ], dtype=torch.long),
            "position_ids": torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1),
            "acc": torch.rand(batch_size)
        }

        # print(micro_batch["input_ids"])
        
        token_level_scores, q_values = self.rm_instance._forward_micro_batch(micro_batch, prompt_length)
        
        torch.set_printoptions(precision=30, sci_mode=False)
        # print(token_level_scores)
        expected_token_level_scores = torch.tensor([[ 0.010057640261948108673095703125, -0.020435070618987083435058593750,
          0.014614701271057128906250000000, -0.011621952056884765625000000000,
          0.000000000000000000000000000000,  0.000000000000000000000000000000,
          0.000000000000000000000000000000,  0.000000000000000000000000000000],
        [ 0.081534914672374725341796875000, -0.026112342253327369689941406250,
          0.021826243028044700622558593750,  0.033189535140991210937500000000,
         -0.052226688712835311889648437500, -0.006393861956894397735595703125,
         -0.031713128089904785156250000000,  0.000000000000000000000000000000]])
        torch.testing.assert_close(token_level_scores, expected_token_level_scores, atol=1e-30, rtol=1e-30)

        # Verify that rewards at padding positions are 0
        # The last 3 response positions of the first sequence should be 0
        response_len = seq_len - prompt_length
        self.assertEqual(token_level_scores[0, -3:].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
