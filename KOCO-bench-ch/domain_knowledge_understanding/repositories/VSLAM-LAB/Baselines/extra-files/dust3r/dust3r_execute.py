
import os
import sys
import torch
import subprocess
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

dust3r_path = os.path.join(os.getcwd(), 'Baselines', 'dust3r')
sys.path.append(dust3r_path)

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--sequence_path", type=str, help="path to image directory")
    parser.add_argument("--calibration_yaml", type=str, help="path to calibration file")
    parser.add_argument("--rgb_txt", type=str, help="path to image list")
    parser.add_argument("--exp_folder", type=str, help="path to save results")
    parser.add_argument("--exp_it", type=str, help="experiment iteration")
    parser.add_argument("--settings_yaml", type=str, help="settings_yaml")
    parser.add_argument("--verbose", type=str, help="verbose")

    parser.add_argument('--device', type=str, default='cuda', help="")
    parser.add_argument('--batch_size', type=int, default=1, help="")
    parser.add_argument('--schedule', type=str, default='cosine', help="")
    parser.add_argument('--lr', type=float, default=0.01, help="")
    parser.add_argument('--niter', type=int, default=300, help="")
    parser.add_argument('--img_size', type=int, default=512, help="")

    args, unknown = parser.parse_known_args()
    sequence_path = args.sequence_path
    rgb_txt = args.rgb_txt
    exp_folder = args.exp_folder
    exp_id = args.exp_it
    verbose = bool(int(args.verbose))

    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr = args.lr
    niter = args.niter
    img_size = args.img_size

    # Verify if PyTorch is compiled with CUDA
    print("\nVerify if PyTorch is compiled with CUDA: ")
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        print(output)
    except FileNotFoundError:
        print("    nvcc (NVIDIA CUDA Compiler) not found, CUDA may not be installed.")

    print(f"    Torch with CUDA is available: {torch.cuda.is_available()}")
    print(f"    CUDA version: {torch.version.cuda}")
    print(f"    Device capability: {torch.cuda.get_device_capability(0)}")

    image_list = []
    timestamps = []
    with open(rgb_txt, 'r') as file:
        for line in file:
            timestamp, path, *extra = line.strip().split(' ')
            image_list.append(os.path.join(sequence_path, path))
            timestamps.append(timestamp)

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    images = load_images(image_list, size=img_size)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    poses = scene.get_im_poses()
    keyFrameTrajectory_txt = os.path.join(exp_folder, exp_id.zfill(5) + '_KeyFrameTrajectory' + '.txt')
    with open(keyFrameTrajectory_txt, 'w') as file:
        for i_pose, pose in enumerate(poses):
            tx, ty, tz = pose[0, 3].item(), pose[1, 3].item(), pose[2, 3].item()
            rotation_matrix = np.array([[pose[0, 0].item(), pose[0, 1].item(), pose[0, 2].item()],
                                        [pose[1, 0].item(), pose[1, 1].item(), pose[1, 2].item()],
                                        [pose[2, 0].item(), pose[2, 1].item(), pose[2, 2].item()]])
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()
            qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            ts = timestamps[i_pose]
            line = str(ts) + " " + str(tx) + " " + str(ty) + " " + str(tz) + " " + str(qx) + " " + str(
                qy) + " " + str(qz) + " " + str(qw) + "\n"
            file.write(line)

    if verbose:
        scene.show()
