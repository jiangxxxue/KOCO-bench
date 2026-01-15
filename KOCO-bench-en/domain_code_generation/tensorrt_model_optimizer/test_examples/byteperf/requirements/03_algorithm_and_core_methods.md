# ByteMLPerf Core Algorithms and Methods

# FUNCTION:LogitsAndIntermediatesLossBalancer.forward

## Function Overview
Loss balancer that dynamically balances distillation loss and original loss

## Function Signature
def forward(self, loss_dict: Dict[str, Tensor]) -> Tensor:

## Input Parameters
- `loss_dict`: Dictionary of all scalar losses, containing key-value pairs such as student_loss, LogitsKLLoss

## Detailed Description
Coordinates the relative weights of multiple loss terms during knowledge distillation training. Separates the student model's original learning loss from the distillation loss guided by the teacher. Dynamically adjusts their proportions based on the numerical relationship between the two types of losses to achieve balanced training. When intermediate layer distillation losses exist, scales them proportionally to maintain coordination with output layer distillation loss. Decides whether to retain the original learning loss based on training strategy.

## Output
Aggregated total scalar loss tensor

## Function Implementation
code/byte_train_perf/Megatron-LM/megatron/inference/algos/distillation.py:line 333-363

## Test Code
code/test_code/test_loss_balancer.py

---

# FUNCTION:adjust_distillation_model_for_mcore

## Function Overview
Adjust distillation model to adapt to Megatron-Core architecture

## Function Signature
def adjust_distillation_model_for_mcore(model: mtd.DistillationModel, distill_cfg: Dict[str, Any]):

## Input Parameters
- `model`: Distillation model (mtd.DistillationModel type)
- `distill_cfg`: Distillation configuration dictionary, containing layer names, loss functions, and other configurations

## Detailed Description
Adapts the distillation model to distributed parallel training architecture. Hides the teacher model so it does not participate in gradient computation and parameter updates. Adjusts model layer naming mappings to match layer structures after parallel partitioning. Ensures distillation loss computation can correctly handle parallel data flow and gradient propagation across devices.

## Output
Adjusted distillation model, adapted to MCore architecture

## Function Implementation
code/byte_train_perf/Megatron-LM/megatron/inference/algos/distillation.py:line 435-455

## Test Code
code/test_code/test_mcore_distillation_adjust.py

---

# FUNCTION:_teacher_provider

## Function Overview
Teacher model factory function for creating and loading teacher model during distillation

## Function Signature
def _teacher_provider(config: Namespace, model_kwargs: Dict[str, Any]) -> MCoreGPTModel:

## Input Parameters
- `config`: Namespace teacher model configuration
- `model_kwargs`: Dict[str, Any] model construction parameter dictionary

## Detailed Description
Creates and initializes teacher model instance for distillation training. Constructs model structure based on configuration information, loads teacher model parameter weights from trained checkpoints. Handles parameter name differences between different model formats to ensure correct loading. Supports non-homogeneous layer structures to adapt to different model architecture configurations.

## Output
Teacher model instance loaded with checkpoint

## Function Implementation
code/byte_train_perf/Megatron-LM/megatron/inference/gpt/model_provider.py:line 98-119

## Test Code
code/test_code/test_teacher_provider.py

---

# FUNCTION:model_provider

## Function Overview
Main function for constructing GPT model, supports ModelOpt distillation mode

## Function Signature
def model_provider(pre_process=True, post_process=True, parallel_output=True) -> MCoreGPTModel:

## Input Parameters
- `pre_process`: bool whether to compute embeddings
- `post_process`: bool whether to compute output logits/loss
- `parallel_output`: bool whether to perform allgather on output logits

## Detailed Description
Constructs language model instance and configures it according to training mode. Supports non-homogeneous layers when creating model structure to adapt to optimization needs. Restores model parameters when saved model state exists. Handles compatibility conversion of model parameters in different formats. Integrates teacher model in distillation training mode, configures distillation strategy including loss computation method and loss balancing mechanism. Supports exporting clean student model after distillation completion, or adjusts distillation model to adapt to distributed parallel architecture when continuing training.

## Output
MCoreGPTModel instance (may be wrapped as DistillationModel)

## Function Implementation
code/byte_train_perf/Megatron-LM/megatron/inference/gpt/model_provider.py:line 122-222

## Test Code
code/test_code/test_model_provider.py

---

# FUNCTION:loss_func

## Function Overview
Loss function, supports knowledge distillation loss computation

## Function Signature
def loss_func(loss_mask: torch.Tensor, model: GPTModel, output_tensor: torch.Tensor):

## Input Parameters
- `loss_mask`: torch.Tensor loss mask tensor
- `model`: GPTModel model object
- `output_tensor`: torch.Tensor model output tensor

## Detailed Description
Computes loss values during training and aggregates statistical information. Parses model output to obtain prediction results, applies mask to filter loss contributions from invalid positions. Computes language model prediction loss in standard training mode. In distillation training mode, additionally computes distillation loss guided by teacher model, while retaining original prediction loss for comparative analysis. Aggregates loss values across distributed devices to obtain global statistics for monitoring training progress.

## Output
Loss value tensor and report dictionary

## Function Implementation
code/byte_train_perf/Megatron-LM/megatron/inference/gpt/loss_func.py:line 60-90

## Test Code
code/test_code/test_loss_func.py
