# ByteMLPerf Benchmark Pipeline

## 1. Project Overview

ByteMLPerf is an AI accelerator benchmark suite that focuses on evaluating AI accelerator performance from a practical production perspective. Compared to traditional MLPerf, ByteMLPerf features: models and runtime environments that are closer to real business scenarios; for ASIC hardware evaluation, in addition to performance and accuracy, it also evaluates compiler usability and coverage metrics.

## 2. Workflow

ByteMLPerf's workflow is divided into three main stages:

### Model Preparation and Compilation
- Load model configuration and datasets
- Compile models on target hardware to generate executable code
- Configure test parameters (batch size, iteration count, etc.)

### Performance Testing Execution
- **Inference Testing**: Measure latency and throughput under different batch sizes
- **Micro Testing**: Test performance of basic computation operations and communication operations
- **Training Testing**: Evaluate distributed training performance and scalability

### Result Analysis and Reporting
- Collect performance metric data
- Perform accuracy verification (if needed)
- Generate performance reports and comparative analysis

## 3. Benchmark Categories

### Inference Performance Testing (Inference)
- **General Performance Testing (General Perf)**: Evaluate accelerator inference capabilities using common models
- **Large Language Model Performance Testing (LLM Perf)**: Specifically evaluate accelerator capabilities in handling large language models
- **Test Metrics**: Latency, throughput, accuracy, memory usage

### Micro Performance Testing (Micro)
- **Computation Operations**: Basic computation operations such as Gemm, Softmax, Conv2D, BatchNorm, LayerNorm
- **Communication Operations**: Distributed communication operations such as AllReduce, AllGather, Broadcast, ReduceScatter
- **Test Metrics**: Latency, memory bandwidth, computational power, communication bandwidth, algorithm efficiency

### Training Performance Testing (Training)
- **Distributed Training**: Large-scale model training performance evaluation based on Megatron-LM
- **Test Metrics**: Training throughput, scalability, memory efficiency, communication efficiency

## 4. Hardware Evaluation Dimensions

- **Performance Metrics**: Computational power, memory bandwidth, latency, throughput
- **Accuracy Metrics**: Model accuracy, numerical precision, convergence
- **Usability Metrics**: Compiler coverage, deployment convenience, development efficiency
