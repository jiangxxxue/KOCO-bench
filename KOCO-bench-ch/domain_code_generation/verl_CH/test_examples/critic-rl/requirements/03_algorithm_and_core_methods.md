# CTRL 算法核心函数描述

本文档描述 RL 阶段（基于 verl）中核心算法函数的实现细节。

---

# FUNCTION: RewardModelForCritic.build_revision_messages

## 功能概述
构建用于生成修订代码的提示消息。将原始问题、初始解答和批评组合成符合 chat template 的格式。

## 函数签名
def build_revision_messages(self, init_messages, critique):

## 输入参数
- `init_messages`: 初始对话历史列表，包含问题（user role）和初始解答（assistant role）
- `critique`: 模型生成的批评文本字符串

## 详细描述

函数接收初始对话历史和批评文本，验证初始对话历史为列表类型且最后一条消息的角色为助手，并调用相关函数进行内容清理（仅保留代码部分）和更新，然后将处理后的纯文本、首尾没有空白字符的批评意见与修订指令 "Please finalize your answer accordingly using the same format." 转换为用户角色的新消息追加到对话末尾。

## 输出
- 完整的对话消息列表（List[Dict]），包含原始问题、清理后的初始代码和修订指令

## 函数实现
code/ctrl/rl/critic_rm.py:line 136-150

## 测试代码
code/test_code/test_build_revision_messages.py

---

# FUNCTION: RewardModelForCritic.get_reward_all

## 功能概述
并发执行多个批评-修订对的沙盒测试，计算可验证奖励。通过异步 I/O 机制提高执行效率，支持大规模并行评测。

## 函数签名
async def get_reward_all(
        self, critiques: List[str], revisions: List[str], samples: List[Any]
    ) -> List[float]:

## 输入参数
- `critiques`: 批评文本列表，每个元素对应一个生成的批评
- `revisions`: 修订代码列表，与批评一一对应
- `samples`: 测试样本列表，包含测试用例和问题元数据

## 详细描述

函数会为每个三元组 (sample, critique, revision) 调用底层的奖励计算方法，评估修订版本相对于原始样本的改进质量，并返回数值化的奖励分数。所有计算任务以异步并发方式执行，在保证资源可控的前提下最大化处理吞吐量，默认最大并发数设为 MAX_REQUESTS（256），需要有进度条（"Generating rewards"）实时反馈完成进度。

## 输出
- 浮点数列表，每个元素表示对应批评-修订对的奖励值，范围 [0.0, 1.0]

## 函数实现
code/ctrl/rl/critic_rm.py:line 198-209

## 测试代码
code/test_code/test_get_reward_all.py

---

# FUNCTION: sanitize

## 功能概述
从包含代码和解释的混合文本中提取纯代码块，去除解释性文字，保留markdown格式的代码块结构。

## 函数签名
def sanitize(text):

## 输入参数
- `text`: 包含代码块的混合文本字符串

## 详细描述

该函数使用正则表达式识别和提取markdown格式的代码块，通过正则表达式匹配以三个反引号包裹的代码块，可以识别带或不带 python 标识符的情况。匹配过程不区分大小写。匹配成功后，它会提取第一个代码块的内容，并将其重新格式化为标准的markdown格式 ````python\n{code}\n```返回。如果文本中没有找到代码块或代码块为空，则直接返回原始文本。

## 输出
- 格式化的代码块字符串（````python\n{code}\n```）或原始文本（无代码块时）

## 使用场景
在`build_revision_messages`函数中，用于清理模型生成的初始解答，去除解释性文字，只保留代码部分，确保修订提示的简洁性。

## 函数实现
code/ctrl/rl/critic_rm.py:line 76-84

## 测试代码
code/test_code/test_sanitize_desanitize.py

---

# FUNCTION: desanitize

## 功能概述
去除代码文本中的markdown代码块标记（三个反引号），提取纯代码字符串。

## 函数签名
def desanitize(text):

## 输入参数
- `text`: 包含或不包含markdown代码块标记的文本

## 详细描述

函数使用正则表达式匹配由三个反引号包裹的代码块，可以识别带或不带 python 标识符的情况。匹配成功后，它会提取代码块内部的内容，去除首尾的空白字符后返回。如果文本中没有找到代码块格式，则直接返回原始文本。

## 输出
- 去除markdown标记的纯代码字符串

## 使用场景
在`get_sandbox_response`函数中，用于从修订文本中提取纯代码，去除markdown格式标记，准备代码进行沙盒执行。

## 函数实现
code/ctrl/rl/critic_rm.py:line 66-73

## 测试代码
code/test_code/test_sanitize_desanitize.py

---

# FUNCTION: normalize_code

## 功能概述
通过抽象语法树（AST）级别的变换对Python代码进行规范化，统一变量命名，提高缓存命中率。

## 函数签名
def normalize_code(code_str):

## 输入参数
- `code_str`: 待规范化的Python代码字符串

## 详细描述

函数通过 AST 解析将 Python 代码中的变量统一重命名为标准形式（v_0, v_1, ...），使功能相同但变量名不同的代码能够被识别为等价代码。如果代码存在语法错误，函数会返回原始代码而不进行转换。

## 输出
- 规范化后的代码字符串，或原始代码（解析失败时）

## 缓存优化原理
训练过程中，模型可能生成多个语义相同但变量名不同的代码版本。例如 `x = 1; return x` 和 `y = 1; return y` 在功能上完全等价。通过规范化，两者都被转换为 `v_0 = 1; return v_0`，使用相同的缓存键，避免重复的沙盒执行，显著提升训练效率。

## 函数实现
code/ctrl/rl/critic_rm.py:line 35-63


## 测试代码
code/test_code/test_normalize_code.py
