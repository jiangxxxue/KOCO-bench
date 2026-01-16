from typing import Dict, List

from .base import BaseCurriculumLearning


class RemoveExtremeRewardCurriculumLearning(BaseCurriculumLearning):
    """Curriculum learning implementation that remove prompt with all VR=0 or 1.
    """
    def get_retained_prompt_uids(self, mean_group_reward: Dict[str, float]) -> List[int]:
        # retained prompts
        retained_uids = [uid for uid, mean_r in mean_group_reward.items() 
                        if 0 < mean_r < 1]
        return retained_uids