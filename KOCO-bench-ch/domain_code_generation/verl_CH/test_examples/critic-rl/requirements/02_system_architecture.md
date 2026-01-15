# CTRL 系统架构与模块设计

## 1. 代码结构与主要组件

### 1.1 目录组织
```
code/ctrl/
├── sft/                                # SFT阶段训练脚本
│   └── zero3_accelerate.yaml          # DeepSpeed Zero3配置
├── rl/                                 # RL阶段核心模块（基于verl）
│   ├── __init__.py
│   ├── critic_ray_trainer.py          # 训练循环与批评-修订机制
│   ├── critic_rm.py                    # 基于沙盒执行的奖励模型
│   ├── critic_fsdp_workers.py         # FSDP Worker（Actor/Critic/Ref）
│   └── critic_vllm_rollout.py         # vLLM推理引擎
├── gen/                                # 生成与推理工具
│   ├── api.py                          # API接口
│   ├── prompt.py                       # 提示词模板
│   ├── tree.py                         # 搜索树结构
│   └── config/                         # 推理配置
│       ├── critic/                     # 批评生成配置
│       └── generator/                  # 代码生成配置
├── eval/                               # 评估工具
│   ├── eval_utils.py                   # 评估辅助函数
│   └── sandbox_utils.py                # 沙盒执行接口
└── scripts/
    ├── run_sft.py                      # SFT训练脚本
    ├── run_rl.py                       # RL训练脚本
    ├── run_gen.py                      # 生成脚本
    └── rl_config/
        └── trainer.yaml                # RL训练配置
```

### 1.2 角色与数据流（RL阶段）

#### 核心角色
- **Actor（策略模型）**：生成批评文本，使用 PPO 算法更新
- **Rollout（推理引擎）**：基于 vLLM 的高效批量生成
- **Reference/Proxy（参考模型）**：用于生成修订代码和计算 KL 散度
- **RewardModel（奖励函数）**：基于沙盒执行的可验证奖励计算

注意：CTRL 使用 **GRPO** 算法，因此不需要独立的 Critic 模型（价值网络）。

#### 数据流程
```
1. 输入批次（prompts）
   ↓
2. Actor生成批评（critiques）
   ↓
3. 构建修订提示（原问题 + 原代码 + 批评）
   ↓
4. Reference生成修订代码（revisions）
   ↓
5. 沙盒执行修订代码（sandbox execution）
   ↓
6. 计算可验证奖励（pass rate）
   ↓
7. 奖励分配到token级（last token）
   ↓
8. 计算KL惩罚（old_log_prob vs ref_log_prob）
   ↓
9. GRPO优势估计（group-based baseline）
   ↓
10. PPO更新Actor
```

## 2. 训练管线与配置要点

### 2.1 训练入口与模式

#### SFT 阶段
- **入口**：`python scripts/run_sft.py`
- **框架**：基于 DeepSpeed Zero3 的监督微调
- **数据**：合成批评数据（从执行反馈生成）
- **目标**：让模型初步学会批评的格式和内容

#### RL 阶段
- **入口**：`python scripts/run_rl.py`
- **框架**：基于 verl 的分布式 RL 训练
- **算法**：PPO + GRPO（无需 Critic）
- **配置文件**：`scripts/rl_config/trainer.yaml`

### 2.2 关键配置项

#### 数据配置（`data`）
```yaml
train_files: 训练数据路径（parquet格式）
val_files: 验证数据路径
prompt_key: 数据中prompt字段名
max_prompt_length: 最大提示长度
max_response_length: 最大响应长度（批评长度）
train_batch_size: 训练批次大小
```

#### Actor-Rollout-Ref 配置（`actor_rollout_ref`）
```yaml
hybrid_engine: True                     # 使用混合引擎（Actor + Rollout + Ref）
model.path: 模型路径
actor.ppo_mini_batch_size: PPO小批次大小
actor.ppo_epochs: PPO训练轮数
actor.clip_ratio: PPO裁剪比例
actor.entropy_coeff: 熵正则化系数
actor.use_kl_loss: False                # GRPO不使用KL损失
rollout.name: vllm                      # 使用vLLM推理
rollout.temperature: 1.0                # 采样温度
rollout.n: 1                            # 每个prompt生成的样本数（GRPO可设>1）
```

#### 算法配置（`algorithm`）
```yaml
adv_estimator: grpo                     # 使用GRPO优势估计
gamma: 1.0                              # 折扣因子
lam: 1.0                                # GAE参数（GRPO中不使用）
kl_ctrl.kl_coef: 0.001                  # KL散度惩罚系数
```

### 2.3 批评-修订机制实现

#### 核心流程（`critic_ray_trainer.py: fit()`）

训练循环在每个批次中执行批评-修订的完整流程：

**生成批评阶段**：调用 `actor_rollout_wg.generate_sequences()` 使用 Actor 模型生成批评文本。生成完成后，通过 `reward_fn.parse_response()` 解析批评内容，提取问题 ID、批评文本和有效响应长度。如果配置了多样本生成（n > 1），批次会相应扩展。为优化并行效率，对批次进行序列长度均衡，确保各 GPU 处理的总 token 数相近。

**构建修订提示**：从 `reward_fn.parse_query()` 获取原始问题和测试样本信息。调用 `build_revision_messages()` 将问题、初始代码和批评组合成对话格式。使用 tokenizer 的 `apply_chat_template()` 方法格式化提示，生成符合模型训练格式的输入序列。

**生成修订代码**：调用 `actor_rollout_wg.generate_sequences_ref()` 使用参考模型或代理模型生成修订。该方法内部会临时增加最大生成长度至 1024 tokens 以容纳完整代码，采用贪婪解码（do_sample=False）确保生成的稳定性。生成完成后，通过 tokenizer 的 `batch_decode()` 将 token 序列解码为文本。

**异步沙盒执行**：创建异步事件循环，调用 `reward_fn.get_reward_all()` 并发执行批评-修订对的测试。在等待沙盒结果时，并行计算参考策略的对数概率（compute_ref_log_prob），充分利用等待时间。完成后获取所有样本的奖励列表，每个元素表示对应修订代码的测试通过率。

**奖励分配**：创建与响应形状相同的零张量 reward_tensor。遍历批次中的每个样本，将序列级奖励分配到批评的最后一个有效 token 位置，其他位置保持为零。这种分配方式反映了整个批评序列的质量由最终结果决定。

### 2.4 可验证奖励计算（`critic_rm.py`）

#### 奖励模型设计
RewardModelForCritic 类实现基于沙盒执行的可验证奖励计算。输入为批评文本、修订代码和测试样本，输出为 0 到 1 之间的浮点数，表示测试用例的通过率。

#### 奖励计算流程（`reward_revision`）

**批评有效性检查**：首先检查批评文本是否包含明确判断标识，如 "Overall judgment: Correct" 或 "Overall judgment: Incorrect"。这些标识是批评格式的必要组成部分，表明批评者做出了明确的正误判断。若批评缺少这些标识，直接返回零奖励，鼓励模型生成结构化、明确的批评。

**代码提取与规范化**：调用 `sanitize()` 函数从修订文本中提取代码块，去除 markdown 格式标记。使用 `normalize_code()` 对代码进行抽象语法树（AST）级别的规范化：解析代码生成 AST，统一变量命名（如 v_0, v_1），去除冗余空白，重新生成标准格式的代码字符串。规范化后的代码在语义上与原代码等价，但提高了缓存命中率。

**缓存查询**：使用规范化代码和测试用例的组合作为键，在 code_to_reward 字典中查找历史执行结果。若找到匹配项，直接返回缓存的奖励值，跳过沙盒执行。这对训练后期特别有效，因为模型可能生成重复或相似的代码。

**沙盒执行**：根据数据集类型构建相应的执行请求。MBPP+ 和 APPS 使用 RunCodeRequest，将代码与测试用例拼接后执行。LiveCodeBench 使用 SubmitRequest 配合 LiveCodeBenchDataset 类型，提供问题元数据。CodeContests 使用 CommonOJDataset 进行 OJ 评测。异步调用 `submit_to_sandbox()` 将请求发送到沙盒服务，等待执行结果。

**奖励聚合**：根据执行结果计算奖励值。若配置为运行所有测试用例，计算平均通过率；否则使用二元奖励（全部通过为 1，否则为 0）。最后将计算结果更新到缓存字典，供后续查询使用。

#### 支持的数据集格式
- CodeContests：竞赛风格评测，使用 CommonOJDataset 执行
- LiveCodeBench：实时编程挑战，使用 LiveCodeBenchDataset 执行  
- MBPP+/APPS：单元测试风格，直接拼接测试代码执行

#### 性能优化
代码哈希缓存机制避免重复执行相同代码。规范化通过 AST 变换统一代码风格，使功能相同但写法不同的代码能被识别为等价。异步并发执行充分利用 I/O 等待时间，信号量控制防止沙盒服务过载。

### 2.5 混合引擎实现（`critic_fsdp_workers.py`）

#### ActorRolloutRefWorker
这是一个统一的 Worker，根据 `role` 参数可以扮演不同角色：
- `role='actor_rollout_ref'`：同时支持 Actor、Rollout 和 Ref 功能
- 使用 FSDP 进行参数管理和通信
- 支持参数/梯度/优化器卸载以节省显存

#### 关键方法
- `generate_sequences()`：使用 vLLM 生成批评
- `generate_sequences_ref()`：使用 Proxy 模型生成修订（更长的 max_tokens）
- `compute_log_prob()`：计算 old_log_probs
- `compute_ref_log_prob()`：计算 ref_log_probs（用于 KL 惩罚）
- `update_actor()`：PPO 更新

#### Proxy 模型机制

配置文件中可以为 Proxy 模型指定与 Actor 不同的路径（proxy.model.path），允许使用更强的基础模型生成修订。生成修订时通过 use_proxy 参数指示系统切换模型。内部实现中，调用 rollout_sharding_manager 的 update_module 方法，将推理引擎绑定到 proxy_module_fsdp 的参数。这种设计使得 Actor 专注于学习批评，而修订质量由更强模型保证，简化训练逻辑。

