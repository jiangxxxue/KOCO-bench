# CTRL Algorithm Core Function Descriptions

This document describes implementation details of core algorithm functions in RL stage (verl-based).

---

# FUNCTION: RewardModelForCritic.build_revision_messages

## Function Overview
Builds prompt messages for generating revised code. Combines original problem, initial solution, and critique into chat template format.

## Function Signature
def build_revision_messages(self, init_messages, critique):

## Input Parameters
- `init_messages`: Initial dialogue history list containing problem (user role) and initial solution (assistant role)
- `critique`: Model-generated critique text string

## Detailed Description

Function receives initial dialogue history and critique text, validates that initial dialogue history is list type and last message role is assistant, calls related functions for content cleaning (keeping only code portion) and updating, then converts processed plain text critique with no leading/trailing whitespace and revision instruction "Please finalize your answer accordingly using the same format." into new message with user role appended to end of dialogue.

## Output
- Complete dialogue message list (List[Dict]) containing original problem, cleaned initial code, and revision instruction

## Function Implementation
code/ctrl/rl/critic_rm.py:line 136-150

## Test Code
code/test_code/test_build_revision_messages.py

---

# FUNCTION: RewardModelForCritic.get_reward_all

## Function Overview
Concurrently executes sandbox tests for multiple critique-revision pairs, computing verifiable rewards. Improves execution efficiency through asynchronous I/O mechanism, supporting large-scale parallel evaluation.

## Function Signature
async def get_reward_all(
        self, critiques: List[str], revisions: List[str], samples: List[Any]
    ) -> List[float]:

## Input Parameters
- `critiques`: Critique text list, each element corresponding to a generated critique
- `revisions`: Revised code list, one-to-one correspondence with critiques
- `samples`: Test sample list containing test cases and problem metadata

## Detailed Description

Function calls underlying reward computation method for each triplet (sample, critique, revision), evaluating improvement quality of revised version relative to original sample, returning numerical reward scores. All computation tasks execute asynchronously and concurrently, maximizing processing throughput while ensuring resource control, with default maximum concurrency set to MAX_REQUESTS (256), requiring progress bar ("Generating rewards") for real-time completion feedback.

## Output
- Float list, each element representing reward value for corresponding critique-revision pair, range [0.0, 1.0]

## Function Implementation
code/ctrl/rl/critic_rm.py:line 198-209

## Test Code
code/test_code/test_get_reward_all.py

---

# FUNCTION: sanitize

## Function Overview
Extracts pure code blocks from mixed text containing code and explanations, removes explanatory text, preserves markdown format code block structure.

## Function Signature
def sanitize(text):

## Input Parameters
- `text`: Mixed text string containing code blocks

## Detailed Description

Function uses regular expressions to identify and extract markdown format code blocks, matching code blocks wrapped in triple backticks through regex, recognizing cases with or without python identifier. Matching process is case-insensitive. After successful match, extracts content of first code block and reformats it to standard markdown format ````python\n{code}\n``` for return. If no code block found in text or code block is empty, directly returns original text.

## Output
- Formatted code block string (````python\n{code}\n```) or original text (when no code block)

## Use Case
In `build_revision_messages` function, used to clean model-generated initial solution, remove explanatory text, keep only code portion, ensuring conciseness of revision prompt.

## Function Implementation
code/ctrl/rl/critic_rm.py:line 76-84

## Test Code
code/test_code/test_sanitize_desanitize.py

---

# FUNCTION: desanitize

## Function Overview
Removes markdown code block markers (triple backticks) from code text, extracts pure code string.

## Function Signature
def desanitize(text):

## Input Parameters
- `text`: Text with or without markdown code block markers

## Detailed Description

Function uses regular expression to match code blocks wrapped in triple backticks, recognizing cases with or without python identifier. After successful match, extracts content inside code block, returns after removing leading/trailing whitespace. If no code block format found in text, directly returns original text.

## Output
- Pure code string with markdown markers removed

## Use Case
In `get_sandbox_response` function, used to extract pure code from revision text, remove markdown format markers, prepare code for sandbox execution.

## Function Implementation
code/ctrl/rl/critic_rm.py:line 66-73

## Test Code
code/test_code/test_sanitize_desanitize.py

---

# FUNCTION: normalize_code

## Function Overview
Normalizes Python code through Abstract Syntax Tree (AST) level transformation, unifies variable naming, improves cache hit rate.

## Function Signature
def normalize_code(code_str):

## Input Parameters
- `code_str`: Python code string to be normalized

## Detailed Description

Function uniformly renames variables in Python code to standard form (v_0, v_1, ...) through AST parsing, enabling functionally identical but differently named code to be recognized as equivalent. If code has syntax errors, function returns original code without transformation.

## Output
- Normalized code string, or original code (when parsing fails)

## Cache Optimization Principle
During training, model may generate multiple semantically identical but differently named code versions. For example, `x = 1; return x` and `y = 1; return y` are functionally completely equivalent. Through normalization, both are converted to `v_0 = 1; return v_0`, using same cache key, avoiding repeated sandbox execution, significantly improving training efficiency.

## Function Implementation
code/ctrl/rl/critic_rm.py:line 35-63


## Test Code
code/test_code/test_normalize_code.py

