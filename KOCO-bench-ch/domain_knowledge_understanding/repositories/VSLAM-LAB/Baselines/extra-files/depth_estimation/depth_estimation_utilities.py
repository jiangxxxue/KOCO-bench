from sklearn.linear_model import TheilSenRegressor
import matplotlib.pyplot as plt
import numpy as np
import argparse
import shutil
import os
import cv2
import yaml


def argument_parser(baseline_name):
    parser = argparse.ArgumentParser(description=baseline_name)

    parser.add_argument('--sequence_path', type=str)
    parser.add_argument('--rgb_txt', type=str, default="none", help="Specify the RGB text (default: 'none')")
    parser.add_argument('--max-depth', type=float, default=8.0)
    parser.add_argument('--min-depth', type=float, default=0.5)
    parser.add_argument('--verbose', type=str)
    parser.add_argument('--anchor', type=str, default="0")

    args, unknown = parser.parse_known_args()
    sequence_path = args.sequence_path

    if args.rgb_txt == 'none':
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
    else:
        rgb_txt = args.rgb_txt

    calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')

    anchor = bool(int(args.anchor))
    print(anchor)
    if anchor:
        depth_folder_name = f'{baseline_name}_anchored'
    else:
        depth_folder_name = baseline_name

    max_depth = args.max_depth
    min_depth = args.min_depth

    return (sequence_path, rgb_txt, calibration_yaml, max_depth, min_depth, depth_folder_name,
            bool(int(args.verbose)), anchor)


def prepare_depth_folder(sequence_path, depth_folder_name):
    depth_folder = os.path.join(sequence_path, depth_folder_name)

    if os.path.exists(depth_folder):
        shutil.rmtree(depth_folder)
    os.makedirs(depth_folder, exist_ok=True)

    rgbd_txt = os.path.join(sequence_path, f'rgbd_{depth_folder_name}.txt')
    if os.path.isfile(rgbd_txt):
        os.remove(rgbd_txt)

    return depth_folder, rgbd_txt


def load_rgb_txt(rgb_txt, sequence_path):
    rgb_paths = []
    rgb_timestamps = []
    with open(rgb_txt, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            rgb_paths.append(os.path.join(sequence_path, parts[1]))
            rgb_timestamps.append(parts[0])

    return rgb_paths, rgb_timestamps


def save_depth_image(depth_np, depthImage_path, scale_factor,
                     rgbd_assoc, rgb_timestamp, rgbImage_path, depth_folder_name):
    depth_np = scale_factor * depth_np
    depth_np = depth_np.astype(np.uint16)
    cv2.imwrite(depthImage_path, depth_np)
    print(depthImage_path)
    rgbd_assoc.append(f"{rgb_timestamp} rgb/{os.path.basename(rgbImage_path)} "
                      f"{rgb_timestamp} {depth_folder_name}/{os.path.basename(depthImage_path)}")

    return rgbd_assoc


def load_calibration_yaml(calibration_yaml):
    with open(calibration_yaml, 'r') as file:
        lines = file.readlines()
    if lines and lines[0].strip() == '%YAML:1.0':
        lines = lines[1:]

    calibration = yaml.safe_load(''.join(lines))

    fx, fy, cx, cy = (calibration["Camera.fx"], calibration["Camera.fy"],
                      calibration["Camera.cx"], calibration["Camera.cy"])

    f = 0.5 * (fx + fy)

    depth_factor = calibration["depth_factor"]
    return f, depth_factor


def print_statistics(depth_np):
    min_val = np.min(depth_np)
    max_val = np.max(depth_np)
    median_val = np.median(depth_np)

    print(f"depth_np: {depth_np.shape}")
    print(f"Min Depth Value: {min_val:.4f} m")
    print(f"Max Depth Value: {max_val:.4f} m")
    print(f"Median Depth Value: {median_val:.4f} m")


def anchor_depth(depth_np, sequence_path, filename, scale_factor, verbose):
    #
    epsilon = 1e-6

    # Load anchor depth image
    depth_folder = os.path.join(sequence_path, 'depth')
    depthImageAnchor_path = os.path.join(depth_folder, os.path.splitext(os.path.basename(filename))[0] + '.png')
    depthImageAnchor = np.array(cv2.imread(depthImageAnchor_path, cv2.IMREAD_UNCHANGED)) / scale_factor

    # Invert depth images
    invDepthImage = np.where(depth_np > epsilon, depth_np, epsilon)
    invDepthImageAnchor = np.where(depthImageAnchor > epsilon, depthImageAnchor, epsilon)

    # Filter values
    valid_mask = (invDepthImageAnchor > epsilon) & (invDepthImage > epsilon)
    invDepthImageAnchor_flat = invDepthImageAnchor[valid_mask].reshape(-1, 1)
    invDepthImage_flat = invDepthImage[valid_mask].reshape(-1, 1)

    # Subsampling
    subsample_size = 1000
    if len(invDepthImageAnchor_flat) > subsample_size:
        indices = np.random.choice(len(invDepthImageAnchor_flat), subsample_size, replace=False)
        invDepthImageAnchor_flat = invDepthImageAnchor_flat[indices]
        invDepthImage_flat = invDepthImage_flat[indices]

    # Estimate regression between inverse depths
    regressor = TheilSenRegressor(random_state=42)
    regressor.fit(invDepthImage_flat, invDepthImageAnchor_flat.ravel())
    scaledInvDepthImage = regressor.predict(invDepthImage.flatten().reshape(-1, 1)).reshape(invDepthImage.shape)

    scaledDepthImage = np.where(scaledInvDepthImage > epsilon, scaledInvDepthImage, epsilon)
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    if verbose:
        print(f"Regression Coefficients:")
        print(f"  Slope: {slope:.4f}")
        print(f"  Intercept: {intercept:.4f}")

    # # Visualization of depth images and differences
    #
    # # Calculate global min and max for consistent color scale
    # min_val = min(depthImageAnchor.min(), scaledDepthImage.min())
    # max_val = max(depthImageAnchor.max(), scaledDepthImage.max())
    #
    # plt.figure(figsize=(18, 6))
    #
    # # Show depthImageAnchor
    # plt.subplot(1, 3, 1)
    # plt.imshow(depthImageAnchor, cmap='viridis', vmin=min_val, vmax=max_val)
    # plt.title('Depth Image Anchor')
    # plt.colorbar()
    #
    # # Show scaledDepthImage
    # plt.subplot(1, 3, 2)
    # plt.imshow(scaledDepthImage, cmap='viridis', vmin=min_val, vmax=max_val)
    # plt.title('Scaled Depth Image')
    # plt.colorbar()
    #
    # # Show difference
    # difference = scaledDepthImage - depthImageAnchor
    # plt.subplot(1, 3, 3)
    # plt.imshow(difference, cmap='coolwarm')
    # plt.title('Difference (Scaled - Anchor)')
    # plt.colorbar()
    #
    # plt.tight_layout()
    # plt.show()

    return scaledDepthImage, slope, intercept
