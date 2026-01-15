#!/usr/bin/env python3
"""快速测试修复后的代码"""
import sys
import os
import torch

# Add the verl directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'luffy', 'verl'))

print("测试1: compute_grpo_outcome_advantage_split - index类型修复")
try:
    from verl.mix_src.mix_core_alg import compute_grpo_outcome_advantage_split
    
    batch_size, response_length = 2, 4
    token_level_rewards = torch.tensor([
        [0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 5.0],
    ])
    eos_mask = torch.ones(batch_size, response_length)
    index = [0, 0]  # 使用列表而不是tensor
    on_policy_mask = torch.tensor([True, True])
    
    advantages, returns = compute_grpo_outcome_advantage_split(
        token_level_rewards, eos_mask, index, on_policy_mask
    )
    
    print(f"✓ 测试1通过 - advantages shape: {advantages.shape}")
except Exception as e:
    print(f"✗ 测试1失败: {e}")
    sys.exit(1)

print("\n测试2: compute_token_on_off_policy_loss - prefix_mask类型修复")
try:
    from verl.mix_src.mix_core_alg import compute_token_on_off_policy_loss
    
    batch_size, response_length = 2, 4
    old_log_prob = torch.tensor([
        [-0.5, -0.3, -0.4, -0.6],
        [-0.2, -0.5, -0.3, -0.4]
    ])
    log_prob = torch.tensor([
        [-0.3, -0.2, -0.3, -0.5],
        [-0.1, -0.4, -0.2, -0.3]
    ])
    advantages = torch.ones(batch_size, response_length)
    eos_mask = torch.ones(batch_size, response_length)
    prefix_mask = torch.zeros(batch_size, response_length, dtype=torch.bool)  # 使用bool类型
    
    result = compute_token_on_off_policy_loss(
        old_log_prob, log_prob, advantages, eos_mask,
        cliprange=0.2, clip_upper_bound=100.0, prefix_mask=prefix_mask,
        off_cliprange=None
    )
    
    print(f"✓ 测试2通过 - pg_loss: {result['pg_loss'].item():.4f}")
except Exception as e:
    print(f"✗ 测试2失败: {e}")
    sys.exit(1)

print("\n测试3: compute_sft_pure_loss")
try:
    from verl.mix_src.mix_core_alg import compute_sft_pure_loss
    
    batch_size, response_length = 2, 4
    log_prob = torch.tensor([
        [-0.5, -0.3, -0.4, -0.6],
        [-0.2, -0.5, -0.3, -0.4]
    ])
    eos_mask = torch.ones(batch_size, response_length)
    
    sft_loss = compute_sft_pure_loss(log_prob, eos_mask)
    
    print(f"✓ 测试3通过 - sft_loss: {sft_loss.item():.4f}")
except Exception as e:
    print(f"✗ 测试3失败: {e}")
    sys.exit(1)

print("\n✓ 所有快速测试通过！")
