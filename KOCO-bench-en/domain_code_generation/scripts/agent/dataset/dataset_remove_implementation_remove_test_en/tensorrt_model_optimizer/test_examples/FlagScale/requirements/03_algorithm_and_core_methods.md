# FlagScale Core Algorithms and Methods

# FUNCTION:model_provider

## Function Overview
Model provider function that selects between ModelOpt or standard model based on configuration

## Function Signature
def model_provider(pre_process=True, post_process=True, vp_stage: Optional[int] = None, is_dualpipev_first_chunk: Optional[bool] = False) -> Union[GPTModel, megatron.legacy.model.GPTModel]:

## Input Parameters
- `pre_process`: Whether to perform preprocessing
- `post_process`: Whether to perform postprocessing

## Detailed Description
Determines whether to use an optimized model or standard model based on training configuration. Provides an optimized model instance when optimization features are enabled, otherwise provides a standard model instance.

## Output
GPT model object (ModelOpt optimized or standard version)

## Function Implementation
code/flagscale/train/train_gpt.py:line 86-87

## Test Code
code/test_code/test_model_provider.py

---

# FUNCTION:loss_func

## Function Overview
Loss function computation supporting ModelOpt distillation loss

## Function Signature
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None):

## Input Parameters
- `loss_mask`: Loss mask tensor
- `output_tensor`: Model output tensor
- `model`: Model object

## Detailed Description
Computes corresponding loss values based on training mode. When using distillation mode, computes loss including teacher model guidance; otherwise computes standard training loss. Filters loss contributions from invalid positions through masking mechanism.

## Output
Loss value and related statistics

## Function Implementation
code/flagscale/train/train_gpt.py:line 234-235

## Test Code
code/test_code/test_loss_func.py

---

# FUNCTION:forward_step

## Function Overview
Forward training step function that executes model forward propagation and returns output and loss function

## Function Signature
def forward_step(data_iterator, model: GPTModel):

## Input Parameters
- `data_iterator`: Data iterator
- `model`: GPT model object

## Detailed Description
Completes data processing and model forward computation in a single training step. After obtaining training batch from data source, executes model forward propagation and selects appropriate data input method based on model type. In distillation mode, passes model information to loss computation to support teacher model guidance.

## Output
Model output tensor and partially applied loss function

## Function Implementation
code/flagscale/train/train_gpt.py:line 278-304

## Test Code
code/test_code/test_forward_step.py
