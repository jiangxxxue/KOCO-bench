# FlagScale Project Overview

## 1. Algorithm Overview

FlagScale is an open-source framework for large model training, inference, and serving that achieves seamless scalability across diverse hardware architectures through unified multi-backend management and hardware adaptation mechanisms. The framework adopts a Hydra-based configuration management system, defining experiment directories, backend engines, task types, and other environmental parameters through experiment-level configurations, and specifying models, datasets, training parameters, and other specific task parameters through task-level configurations.

The core value of FlagScale lies in providing unified multi-backend support, including the Megatron-LM distributed training framework, vLLM high-performance inference engine, SGLang structured generation language, llama.cpp lightweight inference, and LeRobot robotics learning framework. The framework supports multiple chip architectures, including mainstream hardware vendors such as NVIDIA, Huawei, Kunlun, and Cambricon, achieving heterogeneous training and inference across chips through heterogeneous computing technology.

## 2. Workflow

FlagScale's workflow is based on a configuration-driven task execution model. The system first selects the corresponding backend engine (Megatron-LM, vLLM, SGLang, or llama.cpp) according to configuration parameters, then performs hardware adaptation and optimization. For training tasks, the system executes distributed training workflows, supporting mixed precision training, gradient checkpointing, and dynamic learning rate scheduling. For inference tasks, the system performs model inference, supporting multimodal inference, batch processing optimization, and memory management. For serving tasks, the system starts a Ray service cluster, deploys models to service nodes, handles client requests, and implements load balancing and auto-scaling. The entire workflow achieves a complete pipeline from model training to inference serving through unified configuration management and modular design.

## 3. Application Scenarios

### Large Model Training Tasks
- Input: Large-scale text datasets
- Output: Pre-trained or fine-tuned models
- Supported Models: GPT, LLaMA, Qwen, DeepSeek, etc.
- Data Sources: Various open-source and proprietary datasets

### Multimodal Inference Tasks
- Input: Multimodal data such as text, images, videos
- Output: Multimodal understanding and generation results
- Supported Models: Vision-language models, robotics models
- Application Domains: Image captioning, visual question answering, robot control

### Model Serving Tasks
- Input: Client requests and queries
- Output: Real-time inference results
- Supported Interfaces: RESTful API, WebSocket
- Deployment Methods: Distributed serving, edge deployment
