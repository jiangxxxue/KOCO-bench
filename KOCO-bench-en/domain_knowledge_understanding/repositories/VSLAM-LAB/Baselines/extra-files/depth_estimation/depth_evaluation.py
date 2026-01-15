import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_depth_image(image_path, scale_factor = 1.0):
    # Load the depth image in grayscale (depth images are single-channel)
    depth_image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH) / scale_factor
    if depth_image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    return depth_image

def compare_depth_images(image1, image2):
    # Ensure the images are of the same shape
    if image1.shape != image2.shape:
        raise ValueError("The depth images must have the same dimensions for comparison.")

    # Compute the absolute difference
    diff = cv2.absdiff(image1, image2)

    # Calculate metrics
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    min_diff = np.min(diff)
    median_diff = np.median(diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    metrics = {
        "Mean Difference": mean_diff,
        "Median Difference": median_diff,
        "Max Difference": max_diff,
        "Min Difference": min_diff,
        "RMSE": rmse
    }

    return diff, metrics

def visualize_images(image1, image2, diff, clip_value):
    # Set consistent scale limits for the depth images
    vmin_depth = min(image1.min(), image2.min())
    vmax_depth = max(image1.max(), image2.max())

    # Set the scale limits for the difference image
    vmin_diff = 0
    vmax_diff = diff.max()

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title("Depth Image 1")
    plt.imshow(image1, cmap='gray', vmin=vmin_depth, vmax=vmax_depth)
    plt.colorbar(label='Depth Value')

    plt.subplot(1, 4, 2)
    plt.title("Depth Image 2")
    plt.imshow(image2, cmap='gray', vmin=vmin_depth, vmax=vmax_depth)
    plt.colorbar(label='Depth Value')

    diff_saturated = np.clip(diff, vmin_diff, clip_value)
    print(vmax_diff)
    plt.subplot(1, 4, 3)
    plt.title("Difference Image")
    plt.imshow(diff_saturated, cmap='jet', vmin=vmin_diff, vmax=clip_value)
    plt.colorbar(label='Difference Value')


    plt.subplot(1, 4, 4)
    plt.title("Histogram of Depth Differences")
    plt.hist(diff.flatten(), bins=50, color='skyblue', edgecolor='black')

    plt.xlabel("Depth Difference (Y-axis)")
    plt.ylabel("Frequency")

    plt.tight_layout()


def compute_inverse_depth(depth_image, min_valid_depth=1e-6):

    mask = depth_image > min_valid_depth
    inverse_depth = np.zeros_like(depth_image, dtype=np.float32)
    inverse_depth[mask] = 1.0 / depth_image[mask]

    return inverse_depth, mask

# Example usage
if __name__ == "__main__":
    # Paths to the depth images
    image1_path = "/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office3/depth/000100.png"
    image2_path = "/media/fontan/data/VSLAM-LAB-Benchmark/REPLICA/office3/depth_pro_anchored/000100.png"
    scale_factor = 6553.5
    try:
        # Load the images
        depth_image1 = load_depth_image(image1_path, scale_factor)
        depth_image2 = load_depth_image(image2_path, scale_factor)

        # Compare the images
        diff, metrics = compare_depth_images(depth_image1, depth_image2)

        # Print metrics
        print("Comparison Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Visualize the images
        visualize_images(depth_image1, depth_image2, diff, 0.3)

        inverse_depth1, mask = compute_inverse_depth(depth_image1)
        inverse_depth2, mask = compute_inverse_depth(depth_image2)
        diff, metrics = compare_depth_images(inverse_depth1, inverse_depth2)

        # Print metrics
        print("Comparison Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        visualize_images(inverse_depth1, inverse_depth2, diff, 0.1)
        plt.show()

    except Exception as e:
            print(e)
