"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END

from deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType

from math_verify import parse, verify

def labeling_responses(responses: list[str], golden_answer: str):
    try:
        predict_answers = list(map(parse, responses))
        golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
        labels = list(map(verify, golden_answers, predict_answers))
        return labels
    except OverflowError as e:
        print(f"Error: {e}")
        return [False]
    except Exception as e:
        print(f"Other Error: {e}")
        return [False]

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # print("think_format", self.config.think_format)
        if self.config.think_format:
            # Extract solution.
            if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
                model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
            else:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        else:
            model_solution = model_response
        # print(model_solution)

        labels = labeling_responses([model_solution,], input.ground_truth["answer"])
        if labels[0] is True:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def reward_fn_math_verify(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

def reward_fn_math_verify_no_think(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.think_format = False
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct
