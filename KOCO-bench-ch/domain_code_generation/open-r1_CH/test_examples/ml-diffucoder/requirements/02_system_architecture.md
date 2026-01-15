# DiffuCoder 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织（ml-diffucoder 核心）
```
code/src/open_r1/
├── grpo.py                         # 训练入口与主流程
├── coupled_grpo.py                 # DiffuGRPOTrainer 核心实现
├── configs.py                      # 训练配置（扩散生成、GRPO 参数）
├── rewards.py                      # 奖励函数注册表（代码执行、格式检查等）
└── utils/
    ├── code_providers.py           # 代码执行提供者（E2B、本地、MorphCloud）
    └── ioi.py                      # IOI 问题评估工具

recipes/
├── config_coupled_code.yaml        # 训练配置文件
├── process_data.py                 # 数据预处理脚本
└── accelerate_configs/             # 分布式训练配置（DDP、FSDP、ZeRO）

tests/
└── test_code_reward.py             # 代码奖励函数测试
```

### 1.2 角色与数据流
- **角色**
  - **Policy Model (π_θ)**：策略模型，基于扩散架构，使用 Coupled-GRPO 更新；支持 FSDP/DDP 并行训练。
  - **Reference Model (π_ref)**：参考模型，用于计算 KL 散度；可选择性同步策略模型参数（`sync_ref_model`）。
  - **Reward Functions**：奖励函数集合，包括代码执行奖励（`code_reward`）、格式奖励（`code_format_reward`）等。
  - **Code Execution Provider**：代码执行后端，支持 E2B、本地执行、MorphCloud 等。

- **关键数据**
  - **prompts/completions**：输入提示与生成的代码响应。
  - **rewards**：按样本的奖励分数（代码执行通过率、格式正确性等）。
  - **per_token_logps**：token 级对数概率，通过 coupled sampling 计算得到。
  - **advantages**：优势估计，基于 leave-one-out 基线计算。

## 2. 训练管线与配置要点

### 2.1 训练入口与模式切换
- **入口**：`python -m open_r1.grpo --config recipes/config_coupled_code.yaml`
- **奖励模式配置**（`recipes/config_coupled_code.yaml`）
  - **代码执行奖励**：`reward_funcs: [code, code_format]`，权重 `reward_weights: [2.0, 0.5]`
  - **代码执行提供者**：`code_provider: e2b`（支持 `e2b`、`local`、`morph`）
  - **并行执行数**：`parallel_code_exec_per_proc: 2`（每个进程的并行代码执行数）

### 2.2 Coupled Sampling 与掩码策略
- **配置项**：`random_masking`、`p_mask_prompt`、`diffusion_steps`
  - `random_masking: True`：在多次迭代中使用随机掩码种子，增强鲁棒性；
  - `p_mask_prompt: 0.0`：提示 token 的掩码概率（默认不掩码提示）；
  - `diffusion_steps: 256`：扩散生成的步数。

- **实现位置**：`DiffuGRPOTrainer.forward_process`（`coupled_grpo.py`）
  - 对每个响应生成三个版本的掩码序列：
    1. **Version 1**：完全掩码所有完成 token；
    2. **Version 2**：按随机比例 `mask_ratio ∈ [0.2, 0.8]` 掩码完成 token；
    3. **Version 3**：反向掩码（掩码 Version 2 未掩码的部分）。
  - 计算加权对数概率：`weights = [1, 1/mask_ratio, 1/(1-mask_ratio)]`；
  - 使用 `selective_log_softmax` 函数高效计算加权对数概率，避免内存溢出。

### 2.3 优势估计与损失
- **优势估计**：`leave-one-out` 基线
  - 对同一 prompt 的 k 个响应，计算 `baseline = (sum(rewards) - reward_i) / (k-1)`；
  - 优势 `advantage_i = reward_i - baseline`；
  - 可选归一化：`advantages = advantages / (std + 1e-4)`。

- **PPO 损失**：
  - 计算概率比 `ratio = exp(logps - old_logps)`；
  - 裁剪比率 `clipped_ratio = clip(ratio, 1-ε, 1+ε)`；
  - 损失 `loss = -min(ratio * advantages, clipped_ratio * advantages)`。

- **KL 散度**：
  - 如果 `β > 0`，计算 KL 散度 `kl = exp(ref_logps - logps) - (ref_logps - logps) - 1`；
  - 总损失 `loss = loss + β * kl`。

### 2.4 扩散生成配置
- **生成参数**：`max_completion_length`、`diffusion_steps`、`generation_temperature`
  - `max_completion_length: 256`：最大生成长度；
  - `diffusion_steps: 256`：扩散步数（步数越多，生成质量越高，但速度越慢）；
  - `generation_temperature: 1.0`：生成温度（影响采样多样性和生成顺序）。

- **生成算法**：`alg="entropy"`，`alg_temp=0.0`
  - 基于熵的掩码选择策略，优先去噪高不确定性的 token。

## 3. 奖励函数系统

### 3.1 代码执行奖励（`code_reward`）
- **功能**：执行生成的代码并计算测试用例通过率；
- **流程**：
  1. 提取代码块（使用正则表达式匹配 ` ```python\n...\n``` `）；
  2. 构建评估脚本（包含代码和测试用例）；
  3. 通过代码执行提供者（E2B/本地/MorphCloud）执行脚本；
  4. 解析输出中的通过率 `__PASS_RATE__`；
  5. 返回通过率作为奖励（范围 [0, 1]）。

### 3.2 代码格式奖励（`code_format_reward`）
- **功能**：检查代码格式是否正确（是否包含代码块标记、语法是否正确）；
- **流程**：
  1. 检查是否匹配格式模式 ` ^...```python\n...\n```\n$ `；
  2. 提取代码块；
  3. 对于 Python 代码，使用 `ast.parse` 检查语法；
  4. 返回奖励：格式正确且语法正确 → 1.0；格式正确但语法错误 → 0.5；格式错误 → 0.0。

### 3.3 其他奖励函数
- **IOI 代码奖励**（`ioi_code_reward`）：针对 IOI 竞赛问题的代码评估；
- **二值代码奖励**（`binary_code_reward`）：将代码执行奖励二值化（通过率 > 0.99 → 1.0，否则 → 0.0）；
- **过长惩罚**（`soft_overlong_punishment`）：对超长生成进行软惩罚。

## 4. 代码执行提供者

### 4.1 E2B 提供者
- **特点**：云端沙箱执行，支持多种语言（Python、JavaScript、R、Java、Bash、C++）；
- **配置**：需要 E2B API Key（环境变量 `E2B_API_KEY`）；
- **路由**：支持本地路由服务器（`e2b_router_url`）以提高并发性能。

### 4.2 本地提供者
- **特点**：本地执行，无需外部依赖；
- **限制**：安全性较低，仅适用于可信代码。

### 4.3 MorphCloud 提供者
- **特点**：支持 IOI 问题评估，提供高性能计算资源；
- **配置**：需要 MorphCloud API Key。
