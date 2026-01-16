import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_sequence_ppl(log_probs: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None,
                        response_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute sequence perplexity (Perplexity)
    
    Args:
        log_probs: shape [batch_size, response_length] - log probabilities for each position
        attention_mask: shape [batch_size, sequence_length] - attention mask, optional
        response_mask: shape [batch_size, response_length] - response mask, optional
        
    Returns:
        ppl: shape [batch_size] - perplexity for each sequence
    """
    if response_mask is not None:
        # Use response mask, only calculate response part
        mask = response_mask
    elif attention_mask is not None:
        # Extract response part from attention mask
        response_length = log_probs.size(-1)
        mask = attention_mask[:, -response_length:]
    else:
        # If no mask, assume all positions are valid
        mask = torch.ones_like(log_probs, dtype=torch.bool)
    
    # Calculate number of valid tokens for each sequence
    valid_tokens = mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
    # print(f"valid_tokens: {valid_tokens}")
    
    # Calculate total log probability for each sequence
    masked_log_probs = log_probs * mask
    total_log_prob = masked_log_probs.sum(dim=-1)  # [batch_size]
    
    # Calculate average log probability
    avg_log_prob = total_log_prob / valid_tokens.squeeze(-1)  # [batch_size]
    
    # Calculate perplexity: PPL = exp(-avg_log_prob)
    ppl = torch.exp(-avg_log_prob)  # [batch_size]
    
    return ppl


def compute_sequence_entropy(entropy: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           response_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute average entropy for sequence
    
    Args:
        entropy: shape [batch_size, response_length] - entropy for each position
        attention_mask: shape [batch_size, sequence_length] - attention mask, optional
        response_mask: shape [batch_size, response_length] - response mask, optional
        
    Returns:
        avg_entropy: shape [batch_size] - average entropy for each sequence
    """
    if response_mask is not None:
        # Use response mask, only calculate response part
        mask = response_mask
    elif attention_mask is not None:
        # Extract response part from attention mask
        response_length = entropy.size(-1)
        mask = attention_mask[:, -response_length:]
    else:
        # If no mask, assume all positions are valid
        mask = torch.ones_like(entropy, dtype=torch.bool)
    
    # Calculate number of valid tokens for each sequence
    valid_tokens = mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
    # print(f"valid_tokens: {valid_tokens}")
    
    # Calculate total entropy for each sequence
    masked_entropy = entropy * mask
    total_entropy = masked_entropy.sum(dim=-1)  # [batch_size]
    
    # Calculate average entropy
    avg_entropy = total_entropy / valid_tokens.squeeze(-1)  # [batch_size]
    
    return avg_entropy


def compute_sequence_metrics(log_probs: torch.Tensor,
                           entropy: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None,
                           response_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both sequence perplexity and average entropy
    
    Args:
        log_probs: shape [batch_size, response_length] - log probabilities for each position
        entropy: shape [batch_size, response_length] - entropy for each position
        attention_mask: shape [batch_size, sequence_length] - attention mask, optional
        response_mask: shape [batch_size, response_length] - response mask, optional
        
    Returns:
        ppl: shape [batch_size] - perplexity for each sequence
        avg_entropy: shape [batch_size] - average entropy for each sequence
    """
    ppl = compute_sequence_ppl(log_probs, attention_mask, response_mask)
    avg_entropy = compute_sequence_entropy(entropy, attention_mask, response_mask)
    
    return ppl, avg_entropy


def compute_batch_metrics(log_probs: torch.Tensor,
                         entropy: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         response_mask: Optional[torch.Tensor] = None) -> dict:
    """
    Compute batch-level metric statistics
    
    Args:
        log_probs: shape [batch_size, response_length] - log probabilities for each position
        entropy: shape [batch_size, response_length] - entropy for each position
        attention_mask: shape [batch_size, sequence_length] - attention mask, optional
        response_mask: shape [batch_size, response_length] - response mask, optional
        
    Returns:
        metrics: dictionary containing various statistical metrics
    """
    ppl, avg_entropy = compute_sequence_metrics(log_probs, entropy, attention_mask, response_mask)
    
    # Calculate batch statistics
    metrics = {
        'ppl_mean': ppl.mean().item(),
        'ppl_std': ppl.std().item(),
        'ppl_min': ppl.min().item(),
        'ppl_max': ppl.max().item(),
        'entropy_mean': avg_entropy.mean().item(),
        'entropy_std': avg_entropy.std().item(),
        'entropy_min': avg_entropy.min().item(),
        'entropy_max': avg_entropy.max().item(),
        'batch_size': ppl.size(0)
    }
    
    return metrics
