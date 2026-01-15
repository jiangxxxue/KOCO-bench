# NeMo Core Algorithms and Methods

# FUNCTION:Quantizer.quantize

## Function Overview
Quantize and calibrate the model, supporting multiple quantization algorithms

## Function Signature
def quantize(self, model: "MegatronParallel", forward_loop=None):

## Input Parameters
- `model`: MegatronParallel model instance
- `forward_loop`: Optional calibration forward loop function

## Detailed Description
Converts the model from floating-point representation to low-precision numerical representation to reduce computation and storage overhead. Determines numerical precision and quantization method based on the selected quantization algorithm. Analyzes the model's activation distribution on actual data through the calibration process to determine appropriate scaling parameters. Post-processes quantization parameters to avoid numerical underflow or precision loss. Generates comparison information before and after quantization for validating quantization effectiveness.

## Output
Quantized model instance

## Function Implementation
code/nemo/collections/llm/modelopt/quantization/quantizer.py:line 293-336

## Test Code
code/test_code/test_quantizer.py

---

# FUNCTION:prune_language_model

## Function Overview
Perform structured pruning on GPT/Mamba models

## Function Signature
def prune_language_model(model: llm.GPTModel, pruning_config: PruningConfig, data_module: pl.LightningDataModule | None = None, trainer: nl.Trainer | None = None) -> llm.GPTModel:

## Input Parameters
- `model`: llm.GPTModel model instance
- `pruning_config`: PruningConfig pruning configuration
- `data_module`: pl.LightningDataModule data module (required when not drop_layers)
- `trainer`: nl.Trainer trainer (required when not drop_layers)

## Detailed Description
Reduces model size by removing redundant structures in the model. Supports directly dropping entire layers or performing structured pruning based on importance evaluation. In importance-based pruning, determines pruning targets by evaluating the contribution of each part on real data. Trims structures such as attention heads and intermediate layer dimensions according to the configured pruning ratio. Maintains the model's functional integrity so it can still work normally after reduction.

## Output
Pruned GPTModel instance

## Function Implementation
code/nemo/collections/llm/modelopt/prune/pruner.py:line 84-125

## Test Code
code/test_code/test_pruner.py

---

# FUNCTION:adjust_distillation_model_for_mcore

## Function Overview
Adjust distillation model to adapt to Megatron-Core architecture

## Function Signature
def adjust_distillation_model_for_mcore(model: DistillationModel, distill_cfg: DistillationConfig):

## Input Parameters
- `model`: DistillationModel distillation model instance
- `distill_cfg`: DistillationConfig distillation configuration

## Detailed Description
Adapts the distillation model to distributed parallel training architecture. Adjusts the model state saving and loading mechanism to correctly handle the parameter distribution of student and teacher models under parallel partitioning. Modifies loss calculation logic to support skipping the student model's original training loss and keeping only the distillation loss. Ensures the teacher model does not participate in gradient computation and parameter updates. Enables distillation training to run normally in multi-device parallel environments.

## Output
Adjusted distillation model, adapted for MCore architecture

## Function Implementation
code/nemo/collections/llm/modelopt/distill/utils.py:line 168-194

## Test Code
code/test_code/test_distillation_utils.py

---

# FUNCTION:teacher_provider

## Function Overview
Teacher model provider function for loading and initializing teacher models during distillation

## Function Signature
def teacher_provider(config: llm.GPTConfig, ckpt_path: str, tokenizer: "TokenizerSpec", trainer: nl.Trainer) -> "MCoreGPTModel":

## Input Parameters
- `config`: llm.GPTConfig teacher model configuration
- `ckpt_path`: str teacher model checkpoint path
- `tokenizer`: TokenizerSpec tokenizer
- `trainer`: nl.Trainer trainer object

## Detailed Description
Creates and initializes teacher model instance for distillation training. Constructs teacher model structure according to configuration and loads trained model parameters from saved checkpoint. Correctly handles parameter partitioning and loading in distributed parallel environments, ensuring each device loads corresponding parameter shards. Handles checkpoint format compatibility to support weight files saved in different ways. Releases temporarily occupied memory resources after loading is complete.

## Output
Teacher model instance with loaded weights

## Function Implementation
code/nemo/collections/llm/modelopt/distill/utils.py:line 137-165

## Test Code
code/test_code/test_teacher_provider.py

---

# FUNCTION:get_tensor_shapes_adjust_fn_for_distillation

## Function Overview
Returns a function for adjusting tensor shapes during distillation, supporting pipeline parallelism

## Function Signature
def get_tensor_shapes_adjust_fn_for_distillation(model: Union[torch.nn.Module, List[torch.nn.Module]], seq_length: int, micro_batch_size: int, decoder_seq_length: Optional[int] = None, forward_only: bool = False) -> Union[Callable, None]:

## Input Parameters
- `model`: torch.nn.Module or list of models
- `seq_length`: int sequence length
- `micro_batch_size`: int micro batch size
- `decoder_seq_length`: Optional[int] decoder sequence length
- `forward_only`: bool whether forward-only propagation

## Detailed Description
Coordinates tensor dimensions for data transfer in pipeline-parallel distillation training. When the model is partitioned into multiple pipeline stages, intermediate results from both student and teacher models need to be passed between stages. Calculates tensor dimensions for each model's output based on teacher and student model configurations, adjusting data shapes for cross-stage transfer to accommodate outputs from both models. Ensures correct tensor sizes are used in pipeline send and receive operations to avoid dimension mismatch errors. Applies adjustments only when necessary to avoid unnecessary overhead.

## Output
Tensor shape adjustment function or None

## Function Implementation
code/nemo/collections/llm/modelopt/distill/utils.py:line 230-292

## Test Code
code/test_code/test_get_tensor_shapes_adjust_fn.py
