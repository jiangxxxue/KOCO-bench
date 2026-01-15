from collections import defaultdict

CONSTANT_DATA_UID = -90000


class ExperienceBucketManager:
    """Experience Bucket Manager for organizing successful experiences.
    
    This class manages a bucketing system for experience pool organization where:
    - Buckets are keyed by the number of successful rollouts (num_success)
    - Each bucket contains a list of dataframe UIDs (df_uid) 
    - Provides efficient access and statistics for experience-based sampling
    
    The bucket system helps in:
    1. Organizing experiences by their rollout correctness
    2. Enabling targeted sampling from specific correctness levels
    3. Maintaining consistency between experience pool and bucket organization
    """
    
    def __init__(self):
        """Initialize the experience bucket manager.
        
        Creates empty bucket structure with two main data structures:
        - experience_bucket: Maps num_success -> list of df_uids
        - uid_to_bucket: Maps df_uid -> current bucket (num_success)
        """
        self.experience_bucket = defaultdict(list)  # key: num_success, value: list of df_uid
        self.uid_to_bucket = {}  # key: df_uid, value: current bucket (num_success)
    
    def update_bucket(self, df_uid, num_success):
        """Update the bucket assignment for a specific dataframe UID.
        
        If the number of successes has changed, automatically removes the UID from 
        the old bucket and adds it to the new bucket. This ensures bucket consistency
        as rollout correctness rates evolve during training.
        
        Args:
            df_uid: Unique identifier from the dataframe
            num_success: Number of successful rollouts, used as bucket key.
                        Can be int or tensor (will be converted to int)
        
        Note:
            Ignores updates for CONSTANT_DATA_UID to avoid polluting buckets
            with special constant data entries.
        """
        if df_uid == CONSTANT_DATA_UID:
            return  
            
        num_success_item = num_success if isinstance(num_success, int) else num_success.item()
        
        # Check if this df_uid is already in other buckets
        if df_uid in self.uid_to_bucket:
            old_bucket = self.uid_to_bucket[df_uid]
            # If bucket changed, need to remove from old bucket
            if old_bucket != num_success_item:
                if df_uid in self.experience_bucket[old_bucket]:
                    self.experience_bucket[old_bucket].remove(df_uid)
                    # print(f"[Bucket Update] Moved df_uid {df_uid} from bucket {old_bucket} to bucket {num_success_item}")
        
        # Update bucket information
        if df_uid not in self.experience_bucket[num_success_item]:
            self.experience_bucket[num_success_item].append(df_uid)
        self.uid_to_bucket[df_uid] = num_success_item
    
    def remove_from_bucket(self, df_uid):
        """Remove a specific dataframe UID from all bucket structures.
        
        Completely removes the UID from both the bucket lists and the UID mapping.
        Used when experiences are no longer valid or needed in the experience pool.
        
        Args:
            df_uid: The dataframe UID to remove from bucket system
        """
        if df_uid in self.uid_to_bucket:
            bucket_key = self.uid_to_bucket[df_uid]
            if df_uid in self.experience_bucket[bucket_key]:
                self.experience_bucket[bucket_key].remove(df_uid)
                # print(f"[Bucket Cleanup] Removed df_uid {df_uid} from bucket {bucket_key}")
            del self.uid_to_bucket[df_uid]
    
    def cleanup_consistency(self, experience_pool):
        """Maintain bucket consistency by removing UIDs not present in experience pool.
        
        Performs a consistency check between the bucket system and the actual experience
        pool, removing any stale references. This prevents memory leaks and ensures
        bucket statistics remain accurate.
        
        Args:
            experience_pool (dict): The main experience pool dictionary to check against
        
        Note:
            This is typically called periodically during training to maintain consistency
            as the experience pool evolves.
        """
        # Clean up non-existent df_uid
        to_remove_from_bucket = []
        for bucket_key, df_uid_list in self.experience_bucket.items():
            for df_uid in df_uid_list:
                if df_uid not in experience_pool:
                    to_remove_from_bucket.append((bucket_key, df_uid))
        
        for bucket_key, df_uid in to_remove_from_bucket:
            self.experience_bucket[bucket_key].remove(df_uid)
            if df_uid in self.uid_to_bucket:
                del self.uid_to_bucket[df_uid]
        
        # Clean up uid_to_bucket with non-existent df_uid
        to_remove_from_mapping = []
        for df_uid in self.uid_to_bucket.keys():
            if df_uid not in experience_pool:
                to_remove_from_mapping.append(df_uid)
        
        for df_uid in to_remove_from_mapping:
            del self.uid_to_bucket[df_uid]
    
    def get_bucket_stats(self):
        """Get bucket statistics for metrics logging and monitoring.
        
        Returns detailed statistics about the current bucket distribution,
        which can be used for training metrics, logging, and debugging.
        
        Returns:
            dict: Bucket statistics in format {'bucket/{key}_count': count, ...}
                 where key is the num_success value and count is number of UIDs
                 in that bucket. Returns empty dict if no buckets exist.
        """
        if len(self.experience_bucket) > 0:
            bucket_stats = {}
            for bucket_key, df_uid_list in self.experience_bucket.items():
                bucket_stats[f'bucket/{bucket_key}_count'] = len(df_uid_list)
            return bucket_stats
        else:
            return {}
    
    def print_bucket_stats(self):
        """Print human-readable bucket statistics to console.
        
        Displays bucket distribution in a compact format for debugging and monitoring.
        Only prints buckets that contain at least one UID.
        
        Example output:
            [Bucket Stats] bucket_0:15, bucket_1:8, bucket_2:3
        """
        if len(self.experience_bucket) > 0:
            bucket_info = []
            for bucket_key in sorted(self.experience_bucket.keys()):
                count = len(self.experience_bucket[bucket_key])
                if count > 0:
                    bucket_info.append(f"bucket_{bucket_key}:{count}")
            if bucket_info:
                print(f"[Bucket Stats] {', '.join(bucket_info)}")
    
    def get_bucket_uids(self, bucket_key):
        """Get all dataframe UIDs in a specific bucket.
        
        Retrieves the list of all UIDs currently assigned to the specified bucket.
        Useful for targeted sampling with specific correctness levels.
        
        Args:
            bucket_key: The bucket identifier (num_success value)
            
        Returns:
            list: List of dataframe UIDs in the specified bucket.
                 Returns empty list if bucket doesn't exist.
        """
        return self.experience_bucket.get(bucket_key, [])
    
    def get_total_buckets(self):
        """Get the total number of active buckets.
        
        Returns:
            int: Number of buckets that contain at least one UID
        """
        return len(self.experience_bucket)
    
    def get_total_uids(self):
        """Get the total number of UIDs across all buckets.
        
        Returns:
            int: Total count of unique dataframe UIDs managed by the bucket system
        """
        return len(self.uid_to_bucket)
    
    def clear_all(self):
        """Clear all bucket data and reset the manager to initial state.
        
        Removes all bucket assignments and UID mappings. Typically used during
        training resets or when starting fresh experience collection phases.
        """
        self.experience_bucket.clear()
        self.uid_to_bucket.clear()
        print("[Bucket Manager] All buckets cleared") 