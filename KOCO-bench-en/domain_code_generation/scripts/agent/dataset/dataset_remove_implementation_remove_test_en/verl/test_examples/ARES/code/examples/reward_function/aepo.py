# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import numpy as np
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer
import math

DEFAULT_TARGET_ENTROPY = 0.2

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def laser_d_length_reward(gen_len, target_L, alpha=0.5):
    if target_L > gen_len:
        return alpha
    else:
        return 0.0


def _huber_penalty(x: float, kappa: float) -> float:
    """Smooth penalty for x>=0; quadratic near 0, linear in tail."""
    x = max(float(x), 0.0)
    return 0.5 * x * x / kappa if x <= kappa else x - 0.5 * kappa

def _sigmoid_saturate(x: float, temp: float) -> float:
    """Saturating [0,1] growth; temp controls softness."""
    t = max(float(temp), 1e-6)
    return 1.0 / (1.0 + math.exp(-x / t))

def _margin_from_target(target: float, frac: float, min_margin: float = 1.0) -> float:
    """Difficulty-aware tolerance band around target."""
    tgt = max(float(target), 1.0)
    return max(min_margin, frac * tgt)


