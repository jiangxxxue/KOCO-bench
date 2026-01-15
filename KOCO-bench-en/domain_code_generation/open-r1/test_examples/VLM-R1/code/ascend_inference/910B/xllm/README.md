# Run VLM-R1-OVD on Altlas 800T A2 by XLLM


## Overview  
XLLM is an efficient and user-friendly open-source intelligent inference framework that provides enterprise-level service guarantees and high-performance engine computing capabilities for model inference on domestic chips.  


## Background  
Currently, large language models (LLMs) with parameter scales ranging from tens of billions to trillions are rapidly being deployed in core business scenarios such as intelligent customer service, real-time recommendations, and content generation. Efficient support for domestic computing hardware has become a core requirement for low-cost inference deployment.  

Existing inference engines struggle to effectively adapt to the architectural characteristics of dedicated accelerators (e.g., domestic chips), leading to issues such as low utilization of hardware computing units, unbalanced loads under MoE (Mixture of Experts) architecture, communication overhead bottlenecks, and difficult KV cache management. These problems restrict efficient request inference and system scalability.  

The XLLM inference engine improves the resource utilization efficiency of the entire "communication-computation-storage" link, providing key technical support for the large-scale implementation of large language models in practical business scenarios.  


## Installation of XLLM  

### 1. Download the Official Repository and Dependent Modules  
First, clone the XLLM repository and initialize/update its submodules:  
```bash
git clone https://github.com/jd-opensource/xllm
cd xllm
git submodule init
git submodule update
```  


### 2. Install vcpkg (Compilation Dependency)  
vcpkg is a package manager required for compilation. It will be downloaded automatically during the compilation process by default. Alternatively, you can download it in advance and set the environment variable:  
```bash
git clone https://github.com/microsoft/vcpkg.git
export VCPKG_ROOT=/your/path/to/vcpkg  # Replace with your actual vcpkg path
```  


### 3. Compile and Install the ARM Package for Triton from Source  
Triton’s ARM package needs to be compiled from source before proceeding. Execute the following commands:  
```bash
git clone https://github.com/triton-lang/triton.git
cd triton/python
pip install ninja cmake wheel  # Install build-time dependencies
pip install -e .
```  
Proceed to the next step only after Triton is installed successfully.  


### 4. Install Python Dependencies  
Navigate back to the XLLM root directory and install the required Python packages (using Tsinghua University’s PyPI mirror for faster downloads):  
```bash
cd xllm  # Ensure you are in the XLLM root directory
pip install -r cibuild/requirements-dev.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
pip install --upgrade setuptools wheel
```  


### 5. Compile XLLM  
Before compilation, complete the following preparations and execute the compilation commands:  

#### Preparation  
- Set the `XLLM_KERNELS_PATH` environment variable (this points to the directory of XLLM kernel files).  
- Install the `pre-commit` package (required for compilation):  
  ```bash
  pip install pre-commit
  ```  

#### Compilation Commands  
Choose one of the following commands based on your needs:  

1. Compile to generate an executable file (stored in `build/xllm/core/server/`):  
   ```bash
   python setup.py build
   ```  
   The executable file path will be: `build/xllm/core/server/xllm`  

2. Compile to generate a WHL package (stored in the `dist/` directory, for easy installation via `pip`):  
   ```bash
   python setup.py bdist_wheel
   ```  


#### Troubleshooting: Missing .h Files  
During compilation, you may encounter an error where `.h` files cannot be found. Resolve this by following these steps:  
1. Copy the `xllm_kernels` folder into the XLLM project directory.  
2. Copy all subfolders under `xllm_kernels/include/xllm_kernels` to `xllm/xllm_kernels/`.  
3. Reset the `XLLM_KERNELS_PATH` environment variable to point to the new `xllm_kernels` directory (example below):  
   ```bash
   export XLLM_KERNELS_PATH=/home/xllm/xllm_kernels  # Replace with your actual path
   ```  


## Online Inference of XLLM  
Run the following command to start the XLLM engine:  

```bash
./build/xllm/core/server/xllm \
    --model=/path/VLM-R1-Qwen2.5VL-3B-OVD-0321 \  
    --backend=vlm \                        # backend type as VLM
    --port=8000 \                          # Set service port to 8000
    --max_memory_utilization 0.90 \        # Set maximum memory utilization to 90%
    --model_id=VLM-R1-Qwen2.5VL-3B-OVD-032 # Specify model ID (adjust based on your model)
```  

Once your server is started, you can send requests to the server using the following command:

```
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "VLM-R1-Qwen2.5VL-3B-OVD-0321",
    "messages": [
      {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://cbu01.alicdn.com/img/ibank/2018/960/214/9170412069_1052252572.jpg"  
                }
            },
            {
                "type": "text",
                "text": "请分析图像并回答以下问题。您的回答应包含对图像内容的简要描述和最终答案。描述使用 `<description></description>` 标签包裹，答案使用 `</think></think>` 标签包裹。答案必须以 JSON 格式输出，包含 \"yes\" 或 \"no\"，并提供相关物体的边界框坐标作为解释。如果没有涉及具体物体，则将 \"explanations\" 设为 \"None\"。输出格式如下：\n\n<description>对图像内容的简要描述写在这里</description>\n\n</think>\n```json\n{{\"answer\": \"yes or no\",\n\"explanations\": [\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }},\n    {{\n    \"bbox_2d\": [xx, xx, xx, xx], \"label\": \"xxx\"\n    }}]\n}}\n```\n<|FunctionCallEnd|>\n\n具体问题:根据规则或识别要求，{describe}。 图中是否出现{event}？"
            }  
        ]
      }
    ]
}'
```

## Performance

We conducted performance stress testing on **VLM R1** using `evalscope`. Since the VLM backend of `xllm` does not currently support pure text stress testing, we need to make some modifications to the source code of `evalscope`. The specific adjustment is as follows:  

In the `attempt_connection` function of the file `evalscope/perf/http_client.py`, change the format of the `message` to **text-image (multimodal) format**.  


The following is the provided `evalscope` stress testing script:  

```bash
evalscope perf \
  --parallel 1 2 4 8 16 32 \
  --number 4 8 16 32 64 128 \
  --model VLM-R1-Qwen2.5VL-3B-OVD-0321 \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --dataset random_vl \
  --max-tokens 512 \
  --min-tokens 512 \
  --prefix-length 0 \
  --min-prompt-length 100 \
  --max-prompt-length 100 \
  --image-width 640 \
  --image-height 640 \
  --image-format RGB \
  --image-num 1 \
  --tokenizer-path /path/VLM-R1-Qwen2.5VL-3B-OVD-0321 \
  --extra-args '{"ignore_eos": true}'
```  

The `xllm` performance result is follow:

| Conc. | RPS  | Avg Lat.(s) | P99 Lat.(s) | Gen. toks/s | Avg TTFT(s) | P99 TTFT(s) | Avg TPOT(s) | P99 TPOT(s) | Success Rate |
|-------|------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 1     | 0.12 | 8.663        | 9.156       | 59.10       | 0.775       | 1.002       | 0.015       | 0.016       | 100.0%      |
| 2     | 0.23 | 8.840        | 9.006       | 115.81      | 0.215       | 0.285       | 0.017       | 0.017       | 100.0%      |
| 4     | 0.43 | 9.397        | 9.563       | 217.85      | 0.367       | 0.445       | 0.018       | 0.018       | 100.0%      |
| 8     | 0.79 | 10.172       | 10.254      | 402.47      | 0.677       | 0.805       | 0.019       | 0.019       | 100.0%      |
| 16    | 1.36 | 11.789       | 12.267      | 694.39      | 1.295       | 1.941       | 0.021       | 0.023       | 100.0%      |
| 32    | 2.22 | 14.373       | 14.793      | 1138.47     | 2.739       | 3.262       | 0.023       | 0.028       | 100.0%      |

And the `vllm-ascend` performance result is follow:

| Conc. | RPS  | Avg Lat.(s) | P99 Lat.(s) | Gen. toks/s | Avg TTFT(s) | P99 TTFT(s) | Avg TPOT(s) | P99 TPOT(s) | Success Rate |
|-------|------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 1     | 0.04 | 28.567      | 32.103      | 17.92       | 1.333       | 4.572       | 0.053       | 0.054       | 100.0%      |
| 2     | 0.07 | 28.246      | 28.683      | 36.24       | 0.301       | 0.449       | 0.055       | 0.055       | 100.0%      |
| 4     | 0.14 | 28.239      | 28.729      | 72.47       | 0.391       | 0.684       | 0.054       | 0.055       | 100.0%      |
| 8     | 0.27 | 29.139      | 29.387      | 140.36      | 0.479       | 1.001       | 0.056       | 0.057       | 100.0%      |
| 16    | 0.53 | 30.338      | 31.959      | 269.22      | 0.875       | 2.093       | 0.058       | 0.059       | 100.0%      |
| 32    | 0.98 | 32.465      | 35.385      | 501.47      | 1.462       | 3.703       | 0.061       | 0.063       | 100.0%      |

We have found that compared with `vllm-ascend`, `xllm` achieves a 227% increase in Gen. toks/s (Generation tokens per second) under single concurrency, a 127% increase in overall Gen. toks/s under 32 concurrencies, and a 50% reduction in average TTFT (Time to First Token). These improvements have significantly enhanced the inference speed and efficiency of `VLM-R1`. We would like to express our gratitude to the `xllm team` from `JD.com` for their support!

### Notes  
- **GPU Memory Usage**: The memory occupation is related to the `--max_memory_utilization 0.90` setting. Adjust this parameter according to your hardware’s available memory.  
- **Common Error Fix**: If an error occurs during execution (e.g., device initialization failure), run the following command to set the NPU device:  
  ```bash
  python3 -c "import torch; import torch_npu; torch_npu.npu.set_device('npu:0') "
  ```

