#!/bin/bash
# NeMo 环境 Docker 容器运行脚本

set -e

# 默认配置
IMAGE_NAME="tensorrt"
IMAGE_TAG="latest"
CONTAINER_NAME="tensorrt-env"
WORKSPACE_DIR=""
DATA_DIR=""
DETACH=false

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -n, --name NAME           镜像名称 (默认: ${IMAGE_NAME})"
    echo "  -t, --tag TAG             镜像标签 (默认: ${IMAGE_TAG})"
    echo "  -c, --container NAME      容器名称 (默认: ${CONTAINER_NAME})"
    echo "  -w, --workspace DIR       工作目录挂载路径"
    echo "  -d, --data DIR            数据目录挂载路径"
    echo "  -D, --detach              后台运行"
    echo "  -h, --help                显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                      # 交互式运行"
    echo "  $0 -w /path/to/code -d /path/to/data   # 挂载目录运行"
    echo "  $0 -D                                   # 后台运行"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -c|--container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        -d|--data)
            DATA_DIR="$2"
            shift 2
            ;;
        -D|--detach)
            DETACH=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 构建 docker run 命令
DOCKER_CMD="docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"

# 添加容器名称
DOCKER_CMD="${DOCKER_CMD} --name ${CONTAINER_NAME}"

# 添加挂载目录
if [ -n "$WORKSPACE_DIR" ]; then
    DOCKER_CMD="${DOCKER_CMD} -v ${WORKSPACE_DIR}:/workspace/code"
fi

if [ -n "$DATA_DIR" ]; then
    DOCKER_CMD="${DOCKER_CMD} -v ${DATA_DIR}:/workspace/data"
fi

# 交互模式或后台模式
if [ "$DETACH" = true ]; then
    DOCKER_CMD="${DOCKER_CMD} -d"
else
    DOCKER_CMD="${DOCKER_CMD} -it --rm"
fi

# 添加镜像名称
DOCKER_CMD="${DOCKER_CMD} ${IMAGE_NAME}:${IMAGE_TAG}"

echo "执行命令: ${DOCKER_CMD}"
echo ""

# 执行命令
eval ${DOCKER_CMD}
