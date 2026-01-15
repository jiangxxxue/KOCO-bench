from typing import Dict, List

import numpy as np

from .base import BaseCurriculumLearning


class InterquartileCurriculumLearning(BaseCurriculumLearning):
    """Curriculum learning implementation that retains data within percentile bounds.
    """
    def __init__(self, lower_percentile_bound=25, upper_percentile_bound=75, *args, **kwargs):
        self.lower_percentile_bound = lower_percentile_bound
        self.upper_percentile_bound = upper_percentile_bound

    def get_retained_prompt_uids(self, mean_group_reward: Dict[str, float]) -> List[int]:
        """Retain prompt where mean group reward is between 25th and 75th percentiles.
        """
        # bound for filtering prompts, similar to IQM
        group_rewards = np.array(list(mean_group_reward.values()))
        lower_bound = np.percentile(group_rewards, self.lower_percentile_bound)
        upper_bound = np.percentile(group_rewards, self.upper_percentile_bound)

        # retained prompts
        retained_uids = [uid for uid, mean_r in mean_group_reward.items() 
                        if lower_bound <= mean_r < upper_bound]
        return retained_uids