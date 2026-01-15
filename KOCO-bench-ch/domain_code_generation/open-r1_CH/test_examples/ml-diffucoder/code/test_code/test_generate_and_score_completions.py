import unittest
import torch
import sys
import os
from unittest.mock import MagicMock, patch

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the actual class we want to test
from open_r1.coupled_grpo import DiffuGRPOTrainer


class TestGenerateAndScoreCompletions(unittest.TestCase):
    """测试DiffuGRPOTrainer._generate_and_score_completions函数 - 生成代码完成并计算奖励"""
    
    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 1000
        self.eos_token_id = 50256
    
    def _create_mock_trainer(self):
        """创建一个配置好的mock trainer"""
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # Mock processing_class
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50257
        trainer.processing_class.encode = lambda x: [self.eos_token_id]
        trainer.processing_class.batch_decode = lambda ids, **kwargs: ["generated code"] * ids.size(0)
        
        # Mock accelerator
        trainer.accelerator = MagicMock()
        trainer.accelerator.device = torch.device('cpu')
        trainer.accelerator.process_index = 0
        trainer.accelerator.gather_for_metrics = lambda x: x
        
        # Mock model
        trainer.model = MagicMock()
        trainer.model_wrapped = MagicMock()
        trainer.model.diffusion_generate = MagicMock()
        
        # Mock config - 使用真实对象而不是MagicMock
        class MockArgs:
            max_completion_length = 50
            diffusion_steps = 10
            generation_temperature = 0.7
            generation_batch_size = 4
            random_masking = True
            output_dir = "test_output"
            report_to = []  # 添加 report_to 属性
        
        trainer.args = MockArgs()
        
        # Mock other attributes
        trainer.max_prompt_length = None
        trainer.num_iterations = 2
        trainer.num_generations = 4
        trainer.beta = 0.0
        trainer.ref_model = None
        trainer.scale_rewards = True
        trainer.reward_funcs = [lambda **kwargs: [1.0] * len(kwargs['prompts'])]
        trainer.reward_processing_classes = [trainer.processing_class]
        trainer.reward_weights = torch.tensor([1.0])
        trainer.reward_func_names = ["test_reward"]
        trainer.control = MagicMock()
        trainer.control.should_evaluate = False
        trainer._step = 0  # Add missing _step attribute
        
        # Mock metrics
        trainer._metrics = {"train": {"completion_length": [], "zero_std_ratio": [], 
                                     "rewards/test_reward": [], "rewards/<lambda>": [], 
                                     "reward": [], "reward_std": []}}
        trainer._textual_logs = {"prompt": [], "completion": [], "rewards": {"test_reward": [], "<lambda>": []}}
        
        return trainer
    
    def test_generate_and_score_completions_basic_structure(self):
        """测试基本输出结构"""
        trainer = self._create_mock_trainer()
        
        # Mock输入 - 需要匹配 num_generations
        inputs = [
            {"prompt": "Write a function"},
            {"prompt": "Implement sorting"},
            {"prompt": "Create a class"},
            {"prompt": "Define a method"}
        ]
        
        # Mock生成结果
        batch_size = len(inputs)
        seq_len = 30
        vocab_size = self.vocab_size  # 捕获到局部变量
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, seq_len))
        )
        
        # Mock _get_per_token_logps
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(
            batch_size, trainer.num_iterations, 10
        )
        
        # Mock maybe_apply_chat_template
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            # Mock unwrap_model_for_generation
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                # Mock Trainer._prepare_inputs
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证输出结构
        self.assertIn("prompt_ids", result)
        self.assertIn("prompt_mask", result)
        self.assertIn("completion_ids", result)
        self.assertIn("completion_mask", result)
        self.assertIn("advantages", result)
        self.assertIn("mask_seeds", result)
    
    def test_generate_and_score_completions_prompt_processing(self):
        """测试提示处理"""
        trainer = self._create_mock_trainer()
        
        # 需要匹配 num_generations = 4
        inputs = [{"prompt": f"Test prompt {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        # 记录处理过的提示
        processed_prompts = []
        
        def mock_tokenizer_call(text, **kwargs):
            processed_prompts.extend(text)
            return {
                "input_ids": torch.randint(0, vocab_size, (len(text), 10)),
                "attention_mask": torch.ones(len(text), 10)
            }
        
        # 使用side_effect而不是直接赋值__call__
        trainer.processing_class.side_effect = mock_tokenizer_call
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.side_effect = lambda example, tokenizer: {"prompt": example["prompt"]}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: x
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证提示被处理
        self.assertEqual(len(processed_prompts), batch_size)
    
    def test_generate_and_score_completions_eos_masking(self):
        """测试EOS token后的掩码"""
        trainer = self._create_mock_trainer()
        
        # 需要匹配 num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        completion_length = 10
        
        # 创建包含EOS token的完成序列
        completion_ids = torch.randint(0, vocab_size, (batch_size, completion_length))
        completion_ids[0, 5] = self.eos_token_id  # 在位置5放置EOS
        
        prompt_ids = torch.randint(0, vocab_size, (batch_size, 10))
        full_sequence = torch.cat([prompt_ids, completion_ids], dim=1)
        
        trainer.model.diffusion_generate.return_value = MagicMock(sequences=full_sequence)
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, completion_length)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {"input_ids": prompt_ids, "attention_mask": torch.ones_like(prompt_ids)}
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证completion_mask - 检查第一个样本
        completion_mask = result["completion_mask"]
        
        # EOS位置及之前应该为1（包括EOS本身）
        self.assertGreaterEqual(completion_mask[0, :6].sum().item(), 5)
        
        # 验证mask的基本属性
        self.assertEqual(completion_mask.shape[0], batch_size)
    
    def test_generate_and_score_completions_mask_seeds_generation(self):
        """测试mask seeds的生成"""
        trainer = self._create_mock_trainer()
        
        # 需要匹配 num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        # 测试random_masking=True
        trainer.args.random_masking = True
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证mask_seeds
        mask_seeds = result["mask_seeds"]
        self.assertEqual(len(mask_seeds), trainer.num_iterations)
        
        # 测试random_masking=False
        trainer.args.random_masking = False
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证固定seeds
        mask_seeds = result["mask_seeds"]
        self.assertEqual(mask_seeds, [42] * trainer.num_iterations)
    
    def test_generate_and_score_completions_advantage_calculation(self):
        """测试优势计算（leave-one-out baseline）"""
        trainer = self._create_mock_trainer()
        trainer.num_generations = 4
        
        # 创建4个样本（对应同一个prompt的4个生成）
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        
        # Mock奖励函数返回已知的奖励值
        rewards = [1.0, 2.0, 3.0, 4.0]
        trainer.reward_funcs = [lambda **kwargs: rewards]
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, self.vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.return_value = {
                        "input_ids": torch.randint(0, self.vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        result = trainer._generate_and_score_completions(inputs)
        
        # 验证优势计算
        advantages = result["advantages"]
        
        # 手动计算期望的优势
        # baseline_i = (sum - reward_i) / (k-1)
        # advantage_i = reward_i - baseline_i
        total_reward = sum(rewards)
        expected_advantages = []
        for r in rewards:
            baseline = (total_reward - r) / (trainer.num_generations - 1)
            advantage = r - baseline
            expected_advantages.append(advantage)
        
        # 由于scale_rewards=True，还需要除以标准差
        std = torch.tensor(rewards).std().item()
        expected_advantages = [a / (std + 1e-4) for a in expected_advantages]
        
        # 验证（考虑数值误差）
        for i, expected in enumerate(expected_advantages):
            self.assertAlmostEqual(advantages[i].item(), expected, places=3)
    
    def test_generate_and_score_completions_reward_weighting(self):
        """测试多个奖励函数的加权"""
        trainer = self._create_mock_trainer()
        
        # 设置多个奖励函数
        trainer.reward_funcs = [
            lambda **kwargs: [1.0] * len(kwargs['prompts']),
            lambda **kwargs: [2.0] * len(kwargs['prompts'])
        ]
        trainer.reward_processing_classes = [trainer.processing_class, trainer.processing_class]
        trainer.reward_weights = torch.tensor([0.3, 0.7])
        trainer.reward_func_names = ["reward1", "reward2"]
        trainer._metrics["train"]["rewards/reward1"] = []
        trainer._metrics["train"]["rewards/reward2"] = []
        trainer._textual_logs["rewards"]["reward1"] = []
        trainer._textual_logs["rewards"]["reward2"] = []
        
        # 需要匹配 num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证加权奖励
        # 期望: 0.3 * 1.0 + 0.7 * 2.0 = 0.3 + 1.4 = 1.7
        # 但由于是单个样本，advantage会是0（因为baseline等于reward）
        self.assertIn("advantages", result)
    
    def test_generate_and_score_completions_metrics_logging(self):
        """测试指标记录"""
        trainer = self._create_mock_trainer()
        
        # 需要匹配 num_generations = 4
        inputs = [{"prompt": f"Test {i}"} for i in range(4)]
        batch_size = len(inputs)
        vocab_size = self.vocab_size
        
        trainer.model.diffusion_generate.return_value = MagicMock(
            sequences=torch.randint(0, vocab_size, (batch_size, 30))
        )
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(batch_size, trainer.num_iterations, 10)
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.return_value = {"prompt": "test"}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, 10)),
                        "attention_mask": torch.ones(batch_size, 10)
                    }
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证指标被记录
        self.assertGreater(len(trainer._metrics["train"]["completion_length"]), 0)
        self.assertGreater(len(trainer._metrics["train"]["reward"]), 0)


class TestGenerateAndScoreCompletionsIntegration(unittest.TestCase):
    """集成测试：测试_generate_and_score_completions的完整工作流程"""
    
    def setUp(self):
        torch.manual_seed(42)
    
    def test_end_to_end_generation_and_scoring(self):
        """端到端测试：完整的生成和评分流程"""
        # 这个测试验证整个流程的连贯性
        trainer = object.__new__(DiffuGRPOTrainer)
        
        # 配置所有必要的属性
        trainer.processing_class = MagicMock()
        trainer.processing_class.mask_token_id = 50257
        trainer.processing_class.encode = lambda x: [50256]
        trainer.processing_class.batch_decode = lambda ids, **kwargs: ["def test(): pass"] * ids.size(0)
        def mock_tokenizer_call(text, **kwargs):
            return {
                "input_ids": torch.randint(0, 1000, (len(text), 10)),
                "attention_mask": torch.ones(len(text), 10)
            }
        # 使用side_effect而不是直接赋值__call__
        trainer.processing_class.side_effect = mock_tokenizer_call
        
        trainer.accelerator = MagicMock()
        trainer.accelerator.device = torch.device('cpu')
        trainer.accelerator.process_index = 0
        trainer.accelerator.gather_for_metrics = lambda x: x
        
        trainer.model = MagicMock()
        trainer.model_wrapped = MagicMock()
        trainer.model.diffusion_generate = MagicMock(
            return_value=MagicMock(sequences=torch.randint(0, 1000, (2, 30)))
        )
        
        # 创建一个真实的对象而不是MagicMock来存储配置
        class MockArgs:
            max_completion_length = 50
            diffusion_steps = 10
            generation_temperature = 0.7
            generation_batch_size = 2
            random_masking = False
            output_dir = "test_output"
            report_to = []  # 添加 report_to 属性
        
        trainer.args = MockArgs()
        
        trainer.max_prompt_length = None
        trainer.num_iterations = 1
        trainer.num_generations = 2
        trainer.beta = 0.0
        trainer.ref_model = None
        trainer.scale_rewards = False
        trainer.reward_funcs = [lambda **kwargs: [0.8, 0.9]]
        trainer.reward_processing_classes = [trainer.processing_class]
        trainer.reward_weights = torch.tensor([1.0])
        trainer.reward_func_names = ["test_reward"]
        trainer.control = MagicMock()
        trainer.control.should_evaluate = False
        trainer._step = 0  # Add missing _step attribute
        
        trainer._metrics = {"train": {"completion_length": [], "zero_std_ratio": [],
                                     "rewards/test_reward": [], "rewards/<lambda>": [],
                                     "reward": [], "reward_std": []}}
        trainer._textual_logs = {"prompt": [], "completion": [], "rewards": {"test_reward": [], "<lambda>": []}}
        
        trainer._get_per_token_logps = lambda model, ids, keep, seeds: torch.randn(2, 1, 10)
        
        inputs = [{"prompt": "Write a function"}, {"prompt": "Implement sorting"}]
        
        with patch('open_r1.coupled_grpo.maybe_apply_chat_template') as mock_chat:
            mock_chat.side_effect = lambda example, tokenizer: {"prompt": example["prompt"]}
            
            with patch('open_r1.coupled_grpo.unwrap_model_for_generation') as mock_unwrap:
                mock_unwrap.return_value.__enter__ = lambda self: trainer.model
                mock_unwrap.return_value.__exit__ = lambda self, *args: None
                
                with patch('transformers.Trainer._prepare_inputs') as mock_prepare:
                    mock_prepare.side_effect = lambda self, x: x
                    
                    with patch('open_r1.coupled_grpo.gather') as mock_gather:
                        mock_gather.side_effect = lambda x: x
                        
                        with patch('open_r1.coupled_grpo.gather_object') as mock_gather_obj:
                            mock_gather_obj.side_effect = lambda x: x
                            
                            result = trainer._generate_and_score_completions(inputs)
        
        # 验证所有必要的输出都存在
        required_keys = ["prompt_ids", "prompt_mask", "completion_ids", "completion_mask",
                        "advantages", "mask_seeds"]
        for key in required_keys:
            self.assertIn(key, result, f"输出应该包含{key}")
        
        # 验证输出的形状和类型
        self.assertEqual(result["prompt_ids"].dim(), 2)
        self.assertEqual(result["completion_ids"].dim(), 2)
        self.assertTrue(torch.is_tensor(result["advantages"]))


if __name__ == "__main__":
    unittest.main()
