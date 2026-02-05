# NeMo Full Environment Docker Image
# 基于 NVIDIA CUDA 12.1 运行时镜像，包含 Python 3.11

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 设置 CUDA 环境变量
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libsndfile1 \
    libsox-dev \
    sox \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    zlib1g-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# 复制环境文件
COPY environment.yml /tmp/environment.yml

# 创建 conda 环境
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm /tmp/environment.yml

# 激活环境
ENV CONDA_DEFAULT_ENV=nemo_full_v311
ENV PATH=${CONDA_DIR}/envs/${CONDA_DEFAULT_ENV}/bin:${PATH}

# 设置工作目录
WORKDIR /workspace

# 设置默认 shell 使用 conda 环境
SHELL ["conda", "run", "-n", "nemo_full_v311", "/bin/bash", "-c"]

# 验证安装
RUN python --version && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import nemo; print(f'NeMo: {nemo.__version__}')"

# 默认命令
CMD ["bash"]
