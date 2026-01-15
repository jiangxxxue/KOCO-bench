import os
import requests

def depthanythingv2_download_weights(url, local_filename):
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename} from {url}...")

        # Stream download to handle large files
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad status codes
            with open(local_filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

if __name__ == "__main__":
    checkpoints_folder = "Baselines/Depth-Anything-V2/checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder, exist_ok=True)

    url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
    local_filename = os.path.join(checkpoints_folder, 'depth_anything_v2_vitl.pth')
    depthanythingv2_download_weights(url, local_filename)

    checkpoints_folder = "Baselines/Depth-Anything-V2/metric_depth/checkpoints"
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder, exist_ok=True)

    url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true"
    local_filename = os.path.join(checkpoints_folder, 'depth_anything_v2_metric_hypersim_vitl.pth')
    depthanythingv2_download_weights(url, local_filename)

