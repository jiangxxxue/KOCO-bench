meta_action_prompt = """
You are an expert driving assistant. \
Your current speed is {}m/s, the navigation command is '{}', \
based on the understanding of the driving scene and the navigation information, \
what is your driving plan for the next three seconds? \
Output the planning reasoning process in <think> </think> and final planning answer in <answer> </answer> tags, respectively. \
Planning answer consists of SPEED plan and PATH plan, SPEED includes KEEP, ACCELERATE, DECELERATE, and STOP. \
PATH includes STRAIGHT, RIGHT_CHANGE, LEFT_CHANGE, RIGHT_TURN, and LEFT_TURN. \
For example, a correct answer format is like '<think> planning reasoning process here </think> <answer> KEEP, LEFT_TURN </answer>'.
"""

plan_reason_prompt = """
You are an expert driving assistant. \
Your current speed is {}m/s, the navigation command is '{}', \
and your driving decision for the next three seconds is '{}'. \
Based on your understanding of the driving scene, briefly explain why you made the above driving decisions in one or two sentences.
"""
