./build/xllm/core/server/xllm \
    --model=/path/VLM-R1-Qwen2.5VL-3B-OVD-0321 \  
    --backend=vlm \                        # backend type as VLM
    --port=8000 \                          # Set service port to 8000
    --max_memory_utilization 0.90 \        # Set maximum memory utilization to 90%
    --model_id=VLM-R1-Qwen2.5VL-3B-OVD-032 # Specify model ID (adjust based on your model)