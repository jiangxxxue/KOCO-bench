import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import BCEWithLogitsLoss




def compute_rloo_outcome_advantage(
    reward,
    rollout_n,
):
    reward = reward.reshape(-1, rollout_n, reward.size(-1))  # (bs, rollout_n, 1)
    total_reward = torch.sum(reward, dim=1, keepdim=True)  # (bsz, 1, 1)
    total_reward = total_reward.repeat(1, rollout_n, 1)  # (bsz, rollout_n, 1)
    mean_reward = (total_reward - reward) / (rollout_n - 1)  # (bsz, rollout_n, 1)
    advantage = (reward - mean_reward).reshape(-1, 1)  # (bsz, 1)
    return advantage


def compute_grpo_outcome_advantage(
    reward,
    rollout_n,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    reward = reward.reshape(-1, rollout_n, reward.size(-1))
    mean_reward = torch.mean(reward, dim=1, keepdim=True)  # (bsz, 1, 1)
    if norm_adv_by_std_in_grpo:
        std_kl = torch.std(reward, dim=1, keepdim=True)  # (bsz, 1, 1)
        advantage = (reward - mean_reward) / (std_kl + epsilon)
    else:
        advantage = reward - mean_reward
    advantage = advantage.reshape(-1, advantage.size(-1))  # (bsz, 1)
    return advantage


def compute_naive_advantage(
    reward,
):
    return reward




