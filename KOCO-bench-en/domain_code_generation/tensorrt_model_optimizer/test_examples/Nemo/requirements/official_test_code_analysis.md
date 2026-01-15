# NeMo Core Functions Official Test Code Analysis

This document provides detailed analysis of how each official test code calls the target core functions, including source code evidence and call chain analysis.

---

## 1. FUNCTION: Quantizer.quantize

### Test Code 1: tests/export/test_quantizer.py

**Call Location**: Line 78

**Source Code Snippet**:
```python
# Lines 67-81
@patch('nemo.export.quantize.quantizer.dist')
def test_quantize_method(self, mock_dist, basic_quantization_config, basic_export_config):
    mock_dist.get_rank.return_value = 0

    # Create mock model and forward loop
    mock_model = MagicMock()
    mock_forward_loop = MagicMock()

    quantizer = Quantizer(basic_quantization_config, basic_export_config)

    with patch('modelopt.torch.quantization.quantize') as mock_quantize:
        with patch('modelopt.torch.quantization.print_quant_summary'):
            quantizer.quantize(mock_model, mock_forward_loop)  # ← Line 78: Direct call

            # Verify quantize was called with correct arguments
            mock_quantize.assert_called_once_with(mock_model, QUANT_CFG_CHOICES['int8'], mock_forward_loop)
```

**Analysis Conclusion**: ✅ **Confirmed Valid**
- This is a unit test that directly tests the `Quantizer.quantize()` method
- Uses mock objects to simulate model and forward_loop
- Verifies the call parameters of the quantize method

---

### Test Code 2: examples/nlp/language_modeling/megatron_gpt_ptq.py

**Call Location**: Line 105

**Source Code Snippet**:
```python
# Lines 73-107
# Initialize quantizer
quantizer = Quantizer(cfg.quantization, cfg.export)

# ... model loading code ...

if cfg.quantization.algorithm is not None:
    data_iter = get_calib_data_iter(...)
    dataloader = [data for data in data_iter]

    def forward_loop(model):
        model.set_inference_config(OmegaConf.to_container(cfg.inference))
        for i, batch in enumerate(tqdm(dataloader, desc="Calibrating")):
            model.predict_step(batch, i)

    model = quantizer.quantize(model, forward_loop)  # ← Line 105: Actual call

quantizer.export(model)
```

**Analysis Conclusion**: ✅ **Confirmed Valid**
- This is a complete end-to-end quantization example script
- Actually calls `quantizer.quantize()` method for model quantization
- Provides actual calibration data and forward_loop implementation
- Most suitable as integration test reference

---

## 2. FUNCTION: prune_language_model

### Test Code 1: examples/nlp/language_modeling/megatron_gpt_prune.py

**Call Location**: Lines 164, 171

**Source Code Snippet**:
```python
# Lines 159-177
drop_layers = OmegaConf.to_object(cfg.prune.drop_layers)
if drop_layers:
    assert not export_config
    mtp.plugins.megatron.drop_mcore_gpt_layers(model.model, layers_to_drop=drop_layers)  # ← Line 164
    setattr(model.cfg, "num_layers", model.model.config.num_layers)
else:
    assert cfg.model.tensor_model_parallel_size == 1
    
    mtp.prune(  # ← Line 171: Calls mtp.prune, not prune_language_model
        model,
        mode="mcore_gpt_minitron",
        constraints={"export_config": export_config},
        dummy_input=None,
        config={"forward_loop": forward_loop},
    )
```

**Analysis Conclusion**: ❌ **No Direct Call**
- This script uses **NeMo 1.0 style API**
- Directly calls underlying `mtp.plugins.megatron.drop_mcore_gpt_layers` and `mtp.prune`
- **Not** calling the target function `prune_language_model`
- However, implementation logic is identical to `prune_language_model` (can be used as reference implementation)

---

### Test Code 2: scripts/llm/gpt_prune.py

**Call Location**: Line 119

**Source Code Snippet**:
```python
# Lines 104-133
def main(args):
    pruning_config = PruningConfig(
        target_ffn_hidden_size=args.target_ffn_hidden_size,
        # ... other configurations ...
    )

    data_module = get_data_module(args) if not args.drop_layers else None

    llm.prune(  # ← Line 119: Calls llm.prune (high-level API)
        nemo_checkpoint=args.restore_path,
        save_path=args.save_path,
        pruning_config=pruning_config,
        devices=args.devices,
        # ... other parameters ...
    )
```

**Call Chain Analysis**:
```
scripts/llm/gpt_prune.py:119
  └─> llm.prune() 
      └─> nemo/collections/llm/api.py:381
          └─> prune_language_model(model, pruning_config, data, trainer)  ← Target function
```

**Analysis Conclusion**: ✅ **Indirect Call (via high-level API)**
- This is **NeMo 2.0 style high-level API** script
- After calling `llm.prune()`, internally calls `prune_language_model`
- Suitable as **end-to-end integration test** reference

---

### Test Code 3: nemo/collections/llm/api.py

**Call Location**: Line 381

**Source Code Snippet**:
```python
# Lines 366-387
def prune(
    nemo_checkpoint: str,
    save_path: str,
    pruning_config: PruningConfig,
    # ... parameters ...
) -> str:
    # ... setup code ...
    
    model, trainer = setup_trainer_and_restore_model_with_modelopt_spec(...)
    
    prune_language_model(model, pruning_config, data, trainer)  # ← Line 381: Direct call
    save_pruned_model(trainer, save_path)
    
    console = Console()
    console.print(f"[green]✓ Pruning succeded, pruned checkpoint saved to {save_path}[/green]")
    
    return save_path
```

**Analysis Conclusion**: ✅ **Confirmed Valid**
- This is the internal implementation of `llm.prune()` high-level API
- **Directly calls** the target function `prune_language_model`
- This is the best place to see how `prune_language_model` is called

---

### Test Code 4: tests/functional_tests/L2_NeMo_2_Prune_Llama_TP1PP2.sh

**Call Location**: Shell script calls gpt_prune.py

**Source Code Snippet**:
```bash
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/llm/gpt_prune.py \
  --restore_path /home/TestData/nemo2_ckpt/llama_68M_v4 \
  --target_hidden_size 64 \
  --target_ffn_hidden_size 128 \
  --target_num_attention_heads 4 \
  --target_num_query_groups 4 \
  --target_num_layers 2 \
  --save_path /tmp/pruned-llama
```

**Analysis Conclusion**: ✅ **Indirect Call (L2 level functional test)**
- This is an **L2 level functional test script**
- Triggers `prune_language_model` by calling `gpt_prune.py`
- Suitable for verifying end-to-end pruning workflow

---

## 3. FUNCTION: adjust_distillation_model_for_mcore

### Test Code 1: examples/nlp/language_modeling/megatron_gpt_distillation.py

**Call Location**: Line 167

**Source Code Snippet**:
```python
# Lines 156-169 (inside model_provider_func method)
# [ModelOpt] Distillation mode.
distill_cfg = load_distillation_config(self.transformer_config)

# Intialize DistillationModel.
kd_config = {
    "teacher_model": (_teacher_provider, [self.cfg, copy.deepcopy(self.trainer)], {}),
    "criterion": distill_cfg["criterion"],
    "loss_balancer": distill_cfg["loss_balancer"],
}
model = mtd.convert(model, mode=[("kd_loss", kd_config)])

# Additional tweaks needed for MCore/Nemo.
adjust_distillation_model_for_mcore(model, distill_cfg)  # ← Line 167: Direct call

return model
```

**Analysis Conclusion**: ✅ **Confirmed Valid**
- This is a complete distillation training example
- Called in `DistillationMegatronGPTModel.model_provider_func()`
- After creating distillation model, calls this function for MCore architecture adaptation
- Best **end-to-end distillation example**

---

### Test Code 2: tests/functional_tests/L2_NeMo_2_Distill_Llama3_TP1PP2.sh

**Call Location**: Shell script calls training script

**Source Code Snippet**:
```bash
coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo scripts/llm/gpt_train.py \
  --name nemo2_llama_distill \
  --teacher_path /home/TestData/nemo2_ckpt/llama_68M_v4 \
  --model_path /home/TestData/nemo2_ckpt/llama_68M_v4 \
  --kd_config /tmp/distill-config.yaml \
  --tp_size 1 \
  --pp_size 2 \
  --max_steps 5
```

**Call Chain Analysis**:
```
L2_NeMo_2_Distill_Llama3_TP1PP2.sh
  └─> scripts/llm/gpt_train.py (with --kd_config)
      └─> Creates DistillationMegatronGPTModel
          └─> model_provider_func()
              └─> adjust_distillation_model_for_mcore()  ← Target function
```

**Analysis Conclusion**: ✅ **Indirect Call (L2 level functional test)**
- This is **L2 level distillation functional test**
- Triggers function call through distillation training workflow
- Verifies complete distillation workflow

---

## 4. FUNCTION: teacher_provider

### Test Code 1: examples/nlp/language_modeling/megatron_gpt_distillation.py

**Call Location**: Line 503 (function definition), Line 160 (referenced)

**Source Code Snippet**:
```python
# Lines 503-515: _teacher_provider function definition
def _teacher_provider(cfg: DictConfig, trainer: Trainer) -> MCoreGPTModel:
    """Teacher model factory (must be a non-local function to pickle)."""
    logging.info("Distillation: Loading teacher weights...")
    teacher_model_cfg = _merge_model_arch_fields(cfg, cfg.kd_teacher_restore_from_path)

    model = MegatronGPTModel.restore_from(
        cfg.kd_teacher_restore_from_path,
        override_config_path=teacher_model_cfg,
        trainer=trainer,
    )
    teacher_model_module_list = model.get_model_module_list()
    logging.info("Distillation: ... teacher weights loaded.")
    return teacher_model_module_list[0]

# Lines 156-164: _teacher_provider is referenced
distill_cfg = load_distillation_config(self.transformer_config)
kd_config = {
    "teacher_model": (_teacher_provider, [self.cfg, copy.deepcopy(self.trainer)], {}),  # ← Referenced
    "criterion": distill_cfg["criterion"],
    "loss_balancer": distill_cfg["loss_balancer"],
}
model = mtd.convert(model, mode=[("kd_loss", kd_config)])
```

**Analysis Conclusion**: ⚠️ **Reference Implementation (not direct call to target function)**
- This defines `_teacher_provider`, not the target function `teacher_provider`
- However, **functionality is identical**, implementation logic is consistent
- Different file paths:
  - Target function: `nemo/collections/llm/modelopt/distill/utils.py`
  - Function here: `examples/nlp/language_modeling/megatron_gpt_distillation.py`
- Can be used as **reference implementation** to understand teacher_provider usage

---

### Test Code 2: tests/functional_tests/L2_NeMo_2_Distill_Llama3_TP1PP2.sh

**Call Location**: Shell script triggers distillation training

**Analysis Conclusion**: ✅ **Indirect Call (through distillation workflow)**
- Shares the same test script with `adjust_distillation_model_for_mcore`
- teacher_provider is called to load teacher model during distillation model initialization

---

## 5. FUNCTION: get_tensor_shapes_adjust_fn_for_distillation

### Test Code 1: nemo/lightning/megatron_parallel.py

**Call Location**: Lines 1449-1457

**Source Code Snippet**:
```python
# Lines 1439-1457
@property
def adjust_tensor_shapes_fn(self) -> Union[Callable, None]:
    """
    Retrieves the function to adjust send and receive tensor shapes in Megatron-Core's forward pass.

    Currently only used during non-interleaved pipelining for Distillation.

    Returns:
        Union[Callable, None]: The function which takes in tensor shapes and returns updated shapes,
                               or None if not applicable.
    """
    from nemo.collections.llm.modelopt.distill.utils import get_tensor_shapes_adjust_fn_for_distillation

    return get_tensor_shapes_adjust_fn_for_distillation(  # ← Line 1451: Direct call
        self.model,
        self.seq_length,
        self.micro_batch_size,
        self.decoder_seq_length,
        self.forward_only,
    )
```

**Analysis Conclusion**: ✅ **Confirmed Valid (automatically called by framework)**
- Called in `MegatronStep.adjust_tensor_shapes_fn` property
- This is a **property attribute**, automatically retrieved during training
- When using pipeline parallel distillation training, framework automatically calls this function
- This is the most realistic usage scenario

---

### Test Code 2: examples/nlp/language_modeling/megatron_gpt_distillation.py

**Call Location**: Indirectly triggered (through pipeline parallel distillation)

**Analysis Conclusion**: ✅ **Indirect Call (automatically triggered at runtime)**
- When running distillation script with `pipeline_model_parallel_size > 1` configured
- Framework automatically calls `get_tensor_shapes_adjust_fn_for_distillation`
- No explicit call needed, automatically handled by `MegatronStep`

---

### Test Code 3: tests/functional_tests/L2_NeMo_2_Distill_Llama3_TP1PP2.sh

**Call Location**: Shell script (triggered by PP=2)

**Source Code Snippet**:
```bash
--pp_size 2  # ← Pipeline Parallel = 2, triggers function call
```

**Analysis Conclusion**: ✅ **Indirect Call (L2 functional test trigger)**
- Script sets `--pp_size 2`, enabling pipeline parallelism
- In PP > 1 distillation training, this function is automatically called
- Verifies complete pipeline parallel distillation workflow

---

## 📊 Summary and Recommendations

### Valid Official Test Code (Recommended)

| Function | Recommended Official Test | Type | Rating |
|----------|--------------------------|------|--------|
| `Quantizer.quantize` | `tests/export/test_quantizer.py` | Unit test | ⭐⭐⭐⭐⭐ |
| `Quantizer.quantize` | `examples/nlp/language_modeling/megatron_gpt_ptq.py` | Integration example | ⭐⭐⭐⭐⭐ |
| `prune_language_model` | `nemo/collections/llm/api.py` | Implementation location | ⭐⭐⭐⭐⭐ |
| `prune_language_model` | `scripts/llm/gpt_prune.py` | Integration script | ⭐⭐⭐⭐ |
| `prune_language_model` | `tests/functional_tests/L2_NeMo_2_Prune_Llama_TP1PP2.sh` | L2 test | ⭐⭐⭐⭐ |
| `adjust_distillation_model_for_mcore` | `examples/nlp/language_modeling/megatron_gpt_distillation.py` | Integration example | ⭐⭐⭐⭐⭐ |
| `adjust_distillation_model_for_mcore` | `tests/functional_tests/L2_NeMo_2_Distill_Llama3_TP1PP2.sh` | L2 test | ⭐⭐⭐⭐ |
| `teacher_provider` | `examples/nlp/language_modeling/megatron_gpt_distillation.py` | Reference implementation | ⭐⭐⭐⭐ |
| `get_tensor_shapes_adjust_fn_for_distillation` | `nemo/lightning/megatron_parallel.py` | Framework call | ⭐⭐⭐⭐⭐ |
| `get_tensor_shapes_adjust_fn_for_distillation` | `examples/nlp/language_modeling/megatron_gpt_distillation.py` | Runtime trigger | ⭐⭐⭐⭐ |

### Best Learning Path

1. **Quantizer.quantize**: First look at unit test to understand interface → Then look at integration example to understand actual usage
2. **prune_language_model**: Directly look at `api.py` call method → Then look at L2 test to verify workflow
3. **adjust_distillation_model_for_mcore**: Look at complete distillation example
4. **teacher_provider**: Reference `_teacher_provider` implementation in distillation example
5. **get_tensor_shapes_adjust_fn_for_distillation**: Understand framework automatic call mechanism
