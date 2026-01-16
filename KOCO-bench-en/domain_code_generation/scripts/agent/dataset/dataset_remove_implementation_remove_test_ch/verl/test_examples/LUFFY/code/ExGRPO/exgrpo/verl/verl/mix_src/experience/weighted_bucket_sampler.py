import random
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict


class WeightedBucketSampler:
    """
    A sampler that samples data from different buckets according to weights.
    
    Supports:
    - Different sampling weights for different buckets
    - Uneven data distribution across buckets 
    - Handling empty buckets
    - Allowing repeated samples
    """
    
    def __init__(self, buckets: Dict[int, List[Any]], n_rollout, weighting_method="linear", beta: float = 0.01):
        """
        Initialize the sampler
        
        Args:
            buckets: dict, key is bucket name, value is list of data in the bucket
            weights: dict, key is bucket name, value is sampling weight for that bucket
        """
        def _normal_mapping(x, rollout):
            mu = rollout / 2
            sigma = 1
            coeff = 1 / np.sqrt(2 * np.pi * sigma**2)
            exponent = -((x - mu) ** 2) / (2 * sigma**2)
            return coeff * np.exp(exponent)
        
        def _linear_clip_mapping(x, rollout):
            if x > rollout/2:
                return 0.5
            else:
                return x / rollout
        
        def _sqrt_mapping(x, rollout):
            return np.sqrt(x/rollout)

        self.buckets = buckets.copy()
        self.beta = float(beta)
        
        if weighting_method == "linear":
            self.weights = {k: k/n_rollout for k in buckets.keys()}
        elif weighting_method == "normal":
            self.weights = {k: _normal_mapping(k, n_rollout) for k in buckets.keys()}
        elif weighting_method == "sqrt":
            self.weights = {k: _sqrt_mapping(k, n_rollout) for k in buckets.keys()}
        elif weighting_method == "linear_clip":
            self.weights = {k: _linear_clip_mapping(k, n_rollout) for k in buckets.keys()}
        else:
            raise ValueError(f"Invalid weighting method: {weighting_method}")
        
        # Filter out empty buckets
        self.valid_buckets = {k: v for k, v in self.buckets.items() if len(v) > 0}
        self.valid_weights = {k: self.weights.get(k, 0) for k in self.valid_buckets.keys()}
        
        # Calculate total weight
        self.total_weight = sum(self.valid_weights.values())
        
        if self.total_weight == 0:
            raise ValueError("All buckets are empty or have zero weight")
        
        # Calculate normalized probabilities
        self.probabilities = {k: w / self.total_weight for k, w in self.valid_weights.items()}

        self.total_valid_items = sum(len(v) for v in self.valid_buckets.values())

        print(f"Initialization complete:")
        print(f"  Number of valid buckets: {len(self.valid_buckets)}")
        print(f"  Data size per bucket: {[(k, len(v)) for k, v in self.valid_buckets.items()]}")
        print(f"  Weight per bucket: {[(k, w) for k, w in self.valid_weights.items()]}")
        print(f"  Probability per bucket: {[(k, f'{p:.3f}') for k, p in self.probabilities.items()]}")


    def sample(self, n: int, seed: Optional[int] = None) -> tuple[List[Any], List[float]]:
        """
        Sample n samples according to weights (with replacement)
        Use multinomial distribution to determine number of samples from each bucket at once
        Handle oversampling and ensure no duplicates within buckets
        
        Args:
            n: number of samples to draw
            seed: random seed
            
        Returns:
            tuple of (list of sampled items, list of importance sampling weights)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if len(self.valid_buckets) == 0:
            return [], []
        
        # Use multinomial to determine number of samples from each bucket at once
        bucket_names = list(self.valid_buckets.keys())
        bucket_probs = [self.probabilities[name] for name in bucket_names]
        
        # multinomial returns count for each bucket
        counts = np.random.multinomial(n, bucket_probs)
        
        # Handle oversampling: when allocated count exceeds actual bucket size
        adjusted_counts = []
        excess_samples = 0
        
        for i, count in enumerate(counts):
            bucket_name = bucket_names[i]
            bucket_size = len(self.valid_buckets[bucket_name])
            
            if count > bucket_size:
                # Oversampling: can only sample bucket_size items, record excess
                adjusted_counts.append(bucket_size)
                excess_samples += count - bucket_size
            else:
                adjusted_counts.append(count)
        
        # Redistribute excess samples to other buckets with remaining capacity
        while excess_samples > 0:
            # Find buckets with remaining capacity
            available_buckets = []
            available_probs = []
            
            for i, bucket_name in enumerate(bucket_names):
                bucket_size = len(self.valid_buckets[bucket_name])
                current_count = adjusted_counts[i]
                
                if current_count < bucket_size:
                    available_buckets.append(i)
                    # Redistribute according to original weights
                    available_probs.append(self.probabilities[bucket_name])
            
            if not available_buckets:
                # All buckets are full, cannot allocate more
                print(f"Warning: Cannot allocate {excess_samples} samples, all buckets at max capacity")
                break
            
            # Renormalize probabilities
            total_prob = sum(available_probs)
            available_probs = [p / total_prob for p in available_probs]
            
            # Randomly choose an available bucket to allocate one sample
            chosen_idx = np.random.choice(available_buckets, p=available_probs)
            adjusted_counts[chosen_idx] += 1
            excess_samples -= 1
        
        # Sample from each bucket according to adjusted counts (without replacement)
        samples = []
        sample_bucket_indices = []  # Track which bucket each sample comes from
        
        for i, count in enumerate(adjusted_counts):
            if count > 0:
                bucket_name = bucket_names[i]
                bucket_data = self.valid_buckets[bucket_name]
                
                # Sample count items from current bucket without replacement
                if count >= len(bucket_data):
                    # Take all samples
                    bucket_samples = bucket_data.copy()
                else:
                    # Randomly choose count unique samples
                    bucket_samples = random.sample(bucket_data, count)
                
                samples.extend(bucket_samples)
                # Record bucket index for each sample
                sample_bucket_indices.extend([i] * count)
        
        weights = []

        return samples, weights
    
    def get_bucket_info(self) -> Dict[int, Dict[str, Any]]:
        """Get bucket statistics"""
        info = {}
        for bucket_name in self.buckets:
            info[bucket_name] = {
                'size': len(self.buckets[bucket_name]),
                'weight': self.weights.get(bucket_name, 0),
                'probability': self.probabilities.get(bucket_name, 0),
                'is_empty': len(self.buckets[bucket_name]) == 0
            }
        return info
    