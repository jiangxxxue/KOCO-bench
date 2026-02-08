#!/usr/bin/env python3
"""
Download models from Hugging Face and automatically save them in ~/models/ under a new model directory
"""

import os
import argparse
from pathlib import Path

def get_default_dir_name(model_name: str):
    # Replace / in HuggingFace repository name with - as local directory name
    # Qwen/Qwen2.5-Coder-7B -> Qwen2.5-Coder-7B
    return model_name.split("/")[-1]

def download_model(model_name: str, root_save_dir: str, cache_dir: str = None):
    """
    Download model from Hugging Face
    
    Args:
        model_name: Model name, e.g., "Qwen/Qwen2.5-Coder-7B"
        root_save_dir: Root save directory, all model downloads will create subdirectories under this
        cache_dir: Cache directory, if None then use default cache
    """
    from huggingface_hub import snapshot_download

    # Automatically create directory named after model
    dir_name = get_default_dir_name(model_name)
    model_save_path = os.path.join(root_save_dir, dir_name)
    print(f"Starting model download: {model_name}")
    print(f"Save path: {model_save_path}")
    if cache_dir:
        print(f"Cache path: {cache_dir}")

    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Download model, all content saved in model_save_path directory
        download_kwargs = {
            "repo_id": model_name,
            "local_dir": model_save_path,
            "local_dir_use_symlinks": False,
            "resume_download": True,
        }
        
        # If cache directory is specified, add to parameters
        if cache_dir:
            download_kwargs["cache_dir"] = cache_dir
            
        snapshot_download(**download_kwargs)
        print(f"\n✅ Model download successful!")
        print(f"Model path: {model_save_path}")
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\n💡 Tips:")
        print("1. Make sure installed: pip install huggingface_hub")
        print("2. If login required: huggingface-cli login")
        print("3. Check network connection")
        return False, model_save_path

    return True, model_save_path

def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python download_model.py Qwen/Qwen2.5-Coder-7B-Instruct ~/models
  
This will download the model to: ~/models/Qwen2.5-Coder-7B-Instruct/
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name, e.g., Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    parser.add_argument(
        "save_dir",
        type=str,
        nargs='?',  # Optional positional argument
        default=os.path.expanduser("~/models"),
        help="Root save directory (default: ~/models)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory (if not specified, use HuggingFace default cache path)"
    )

    args = parser.parse_args()

    # Expand ~ symbol
    save_dir = os.path.expanduser(args.save_dir)
    
    # Ensure main save_dir exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Display full path where model will be saved
    final_dir = os.path.join(save_dir, get_default_dir_name(args.model_name))
    print(f"📦 Model will be saved to: {final_dir}\n")

    # Download model to ~/models/<model_name_dir>/
    success, final_path = download_model(args.model_name, save_dir, args.cache_dir)

    if success:
        print(f"\n🎉 You can use the following path for inference:")
        print(f"   {final_path}")

if __name__ == "__main__":
    main()

