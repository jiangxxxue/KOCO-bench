from collections import defaultdict
from typing import Dict, List

import numpy as np
from verl import DataProto


class BaseCurriculumLearning:
    """Base curriculum learning module
    """
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, batch: DataProto):
        outcome_level_vr = batch.batch['verifiable_rewards']
        index = batch.non_tensor_batch['uid']
        bsz = len(index)
        
        # gather rewards in the same group
        id2vr = defaultdict(list)  # {uid (prompt): [r_1, ..., r_n] (group vr)} 
        for i in range(bsz):
            id2vr[index[i]].append(outcome_level_vr[i].item())
        
        # compute mean group reward
        mean_group_reward = {}  # {uid (prompt): mean_r}
        for uid, group_r in id2vr.items():
            mean_group_reward[uid] = np.mean(group_r)
        
        # call DIY function to get retained prompt uids
        retained_uids = self.get_retained_prompt_uids(mean_group_reward)

        # get data ids corresponding to retained uids
        retained_ids = []
        for idx in range(bsz):
            uid = index[idx]
            if uid in retained_uids:
                retained_ids.append(idx)
        return batch.get_dataproto_item(retained_ids)

    def get_retained_prompt_uids(self, mean_group_reward: Dict[str, float]) -> List[int]:
        """Abstract method to be implemented by subclasses to determine which prompt UIDs to retain.
        
        Args:
            mean_group_reward: Dictionary mapping uid (str) to mean reward (float)
            
        Returns:
            List of prompt UIDs to retain in the batch
        """
        raise NotImplementedError