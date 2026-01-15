#!/usr/bin/env python3
"""
从 Hugging Face 下载 Qwen2.5-Coder-7B 模型，并自动保存在 ~/models/下新的模型目录中
"""

import os
import argparse
from pathlib import Path

def get_default_dir_name(model_name: str):
    # 将 HuggingFace 仓库名中的 / 替换为 - 作为本地目录名
    # Qwen/Qwen2.5-Coder-7B -> Qwen-Qwen2.5-Coder-7B
    return model_name.split("/")[-1]

def download_model(model_name: str, root_save_dir: str, cache_dir: str = None):
    """
    从 Hugging Face 下载模型
    
    Args:
        model_name: 模型名称，如 "Qwen/Qwen2.5-Coder-7B"
        root_save_dir: 根保存目录，所有模型下载都会在此下面各自新建子目录
        cache_dir: 缓存目录，如果为None则使用默认缓存
    """
    from huggingface_hub import snapshot_download

    # 自动新建以模型名为名的目录
    dir_name = get_default_dir_name(model_name)
    model_save_path = os.path.join(root_save_dir, dir_name)
    print(f"开始下载模型: {model_name}")
    print(f"保存路径: {model_save_path}")
    if cache_dir:
        print(f"缓存路径: {cache_dir}")

    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        # 下载模型，所有内容保存在 model_save_path 目录下
        download_kwargs = {
            "repo_id": model_name,
            "local_dir": model_save_path,
            "local_dir_use_symlinks": False,
            "resume_download": True,
        }
        
        # 如果指定了缓存目录，添加到参数中
        if cache_dir:
            download_kwargs["cache_dir"] = cache_dir
            
        snapshot_download(**download_kwargs)
        print(f"\n✅ 模型下载成功！")
        print(f"模型路径: {model_save_path}")
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n💡 提示：")
        print("1. 确保已安装: pip install huggingface_hub")
        print("2. 如果需要登录: huggingface-cli login")
        print("3. 检查网络连接")
        return False, model_save_path

    return True, model_save_path

def main():
    parser = argparse.ArgumentParser(
        description="下载 Hugging Face 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python download_model.py Qwen/Qwen2.5-Coder-7B-Instruct ~/models
  
这将会下载模型到: ~/models/Qwen2.5-Coder-7B-Instruct/
        """
    )
    
    # 位置参数
    parser.add_argument(
        "model_name",
        type=str,
        help="模型名称，例如: Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument(
        "save_dir",
        type=str,
        nargs='?',  # 可选的位置参数
        default=os.path.expanduser("~/models"),
        help="保存主目录 (默认: ~/models)"
    )
    
    # 可选参数
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="缓存目录 (如果不指定，使用HuggingFace默认缓存路径)"
    )

    args = parser.parse_args()

    # 展开 ~ 符号
    save_dir = os.path.expanduser(args.save_dir)
    
    # 确保主 save_dir 存在
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 显示将要保存的完整路径
    final_dir = os.path.join(save_dir, get_default_dir_name(args.model_name))
    print(f"📦 模型将保存到: {final_dir}\n")

    # 下载模型到 ~/models/<模型名目录> 下
    success, final_path = download_model(args.model_name, save_dir, args.cache_dir)

    if success:
        print(f"\n🎉 可以使用以下路径进行推理：")
        print(f"   {final_path}")

if __name__ == "__main__":
    main()

