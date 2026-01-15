from sklearn.linear_model import TheilSenRegressor
from tqdm import tqdm
import numpy as np
import argparse
import shutil
import torch
import json
import sys
import cv2
import os

Depth_Anything_V2_path = os.path.join(os.getcwd(), 'Baselines', 'Depth-Anything-V2')
checkpoints_path = os.path.join(Depth_Anything_V2_path, 'checkpoints')
sys.path.append(Depth_Anything_V2_path)
from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')

    parser.add_argument('--sequence_path', type=str)

    parser.add_argument('--rgb_txt', type=str, default="none", help="Specify the RGB text (default: 'none')")
    parser.add_argument('--anchor', action='store_true', default=False, help="Scale predicted depth with information from a sensor")
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--max-depth', type=float, default=8.0)

    args = parser.parse_args()
    sequence_path = args.sequence_path

    if args.rgb_txt == 'none':
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
    else:
        rgb_txt = args.rgb_txt

    # Constants
    depth_folder_name = 'depth_anything_v2'
    subsample_size = 1000
    max_depth = args.max_depth
    min_depth = 0.5
    scale_factor = 6553.5

    # Prepare depth folder
    depth_folder = os.path.join(sequence_path, depth_folder_name)
    depth_folder_anchor = os.path.join(sequence_path, 'depth')

    if os.path.exists(depth_folder):
        shutil.rmtree(depth_folder)
    os.makedirs(depth_folder, exist_ok=True)

    rgbd_txt = os.path.join(sequence_path, f'rgbd_{depth_folder_name}.txt')
    if os.path.isfile(rgbd_txt):
        os.remove(rgbd_txt)

    # Create model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder]})
    depth_anything.load_state_dict(torch.load(os.path.join(checkpoints_path, f'depth_anything_v2_{args.encoder}.pth'), map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Load images from rgb.txt
    rgb_paths = []
    rgb_timestamps = []
    with open(rgb_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            rgb_paths.append(os.path.join(sequence_path, parts[1]))
            rgb_timestamps.append(parts[0])

    rgbd_assoc = []
    for k, filename in enumerate(tqdm(rgb_paths)):
        # Paths
        rgbImage_path = filename
        scaledDepthImage_path = os.path.join(depth_folder, os.path.splitext(os.path.basename(filename))[0] + '.png')
        depthImageAnchor_path = os.path.join(depth_folder_anchor, os.path.splitext(os.path.basename(filename))[0] + '.png')
        # jsonFile = os.path.join(depth_folder, os.path.splitext(os.path.basename(filename))[0] + '.json')

        # Predict disparities
        raw_image = cv2.imread(filename)
        invDepthImage = depth_anything.infer_image(raw_image, args.input_size)

        # Anchoring
        depthImageAnchor = cv2.imread(depthImageAnchor_path, cv2.IMREAD_UNCHANGED) / 6553.5

        # Filter invalid values
        valid_mask = (depthImageAnchor > 0) & (invDepthImage > 0)
        depthImageAnchor_flat = depthImageAnchor[valid_mask].reshape(-1, 1)
        invDepthImage_flat = invDepthImage[valid_mask].reshape(-1, 1)

        # Subsampling for efficiency
        if len(depthImageAnchor_flat) > subsample_size:
            indices = np.random.choice(len(depthImageAnchor_flat), subsample_size, replace=False)
            depthImageAnchor_flat = depthImageAnchor_flat[indices]
            invDepthImage_flat = invDepthImage_flat[indices]

        # Invert anchor depth
        invDepthImageAnchor_flat = 1.0 / depthImageAnchor_flat

        # Estimate regression between inverse depths
        regressor = TheilSenRegressor(random_state=42)
        regressor.fit(invDepthImage_flat, invDepthImageAnchor_flat.ravel())
        scaledInvDepthImage = regressor.predict(invDepthImage.flatten().reshape(-1, 1)).reshape(invDepthImage.shape)
        print(f"Regression Coefficients:")
        print(f"  Slope: {regressor.coef_[0]:.4f}")
        print(f"  Intercept: {regressor.intercept_:.4f}")

        # Save predicted depth
        scaledDepthImage = np.divide(1.0, scaledInvDepthImage, where=scaledInvDepthImage != 0.0)
        scaledDepthImage = np.where((scaledDepthImage < min_depth) | (scaledDepthImage > max_depth), 0.0, scaledDepthImage)

        scaledDepthImage = scaledDepthImage * scale_factor
        scaledDepthImage = scaledDepthImage.astype(np.uint16)
        cv2.imwrite(scaledDepthImage_path, scaledDepthImage)

        # Save metadata
        # metadata = {
        #     "depthMin": str(depth.min()),
        #     "depthMax": str(depth.max()),
        #     "normalizationFactor": str(65535),
        #     "rgbImage": str(rgbImage),
        #     "depthImage": str(depthImage),
        #     "normalization": str("depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65535"),
        #     "type": str("uint16"),
        #     "Description": "Normalized Depth image"
        # }

        # with open(jsonFile, 'w') as json_file:
        #     json.dump(metadata, json_file)

        rgbd_assoc.append(f"{rgb_timestamps[k]} rgb/{os.path.basename(rgbImage_path)} "
                          f"{rgb_timestamps[k]} {depth_folder_name}/{os.path.basename(scaledDepthImage_path)}")

    # Save associations file
    with open(rgbd_txt, 'w') as file:
        for line in rgbd_assoc:
            file.write(line + '\n')
