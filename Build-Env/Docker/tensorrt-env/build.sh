#!/bin/bash
# NeMo 环境 Docker 镜像构建脚本

set -e

# 默认配置
IMAGE_NAME="tensorrt"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"
SAVE_TO_FILE=false

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -n, --name NAME     镜像名称 (默认: ${IMAGE_NAME})"
    echo "  -t, --tag TAG       镜像标签 (默认: ${IMAGE_TAG})"
    echo "  -p, --pip           使用 pip 版本的 Dockerfile (更轻量)"
    echo "  -s, --save          构建后保存镜像到 tar.gz 文件"
    echo "  -h, --help          显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                          # 使用 conda 版本构建"
    echo "  $0 -p                       # 使用 pip 版本构建"
    echo "  $0 -s                       # 构建并保存镜像到文件"
    echo "  $0 -n myimage -t v1.0       # 自定义镜像名称和标签"
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
        -p|--pip)
            DOCKERFILE="Dockerfile.pip"
            shift
            ;;
        -s|--save)
            SAVE_TO_FILE=true
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

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "构建 NeMo 环境 Docker 镜像"
echo "================================================"
echo "镜像名称: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Dockerfile: ${DOCKERFILE}"
echo "构建目录: ${SCRIPT_DIR}"
echo "================================================"

# 构建镜像
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" .

echo ""
echo "================================================"
echo "构建完成！"
echo "================================================"

# 保存镜像到文件
if [ "$SAVE_TO_FILE" = true ]; then
    OUTPUT_FILE="${IMAGE_NAME}-${IMAGE_TAG}.tar.gz"
    echo ""
    echo "正在保存镜像到文件: ${OUTPUT_FILE}"
    docker save "${IMAGE_NAME}:${IMAGE_TAG}" | gzip > "${OUTPUT_FILE}"
    echo "镜像已保存到: ${SCRIPT_DIR}/${OUTPUT_FILE}"
    echo "文件大小: $(du -h "${OUTPUT_FILE}" | cut -f1)"
fi

echo ""
echo "运行镜像:"
echo "  docker run --gpus all -it ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "挂载目录运行:"
echo "  docker run --gpus all -it -v /your/data:/workspace/data ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "保存镜像到文件:"
echo "  docker save ${IMAGE_NAME}:${IMAGE_TAG} | gzip > ${IMAGE_NAME}-${IMAGE_TAG}.tar.gz"
echo ""
