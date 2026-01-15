import os
import yaml
import shutil
import numpy as np

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile
from path_constants import VSLAMLAB_BENCHMARK

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics

# from Baselines.colmap_to_vslamlab import get_colmap_keyframes
# from Baselines.colmap_to_vslamlab import get_colmap_keyframes
# from Baselines.colmap_to_vslamlab import write_trajectory_tum_format
# from Baselines.colmap_to_vslamlab import get_timestamps
class ETH3D_MVS_DSLR_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('eth3d_mvs_dslr', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):

        # Variables
        compressed_name = f"{sequence_name}_undistorted"
        compressed_name_ext = compressed_name + '.7z'
        decompressed_name = sequence_name
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if os.path.exists(self.dataset_path):
            if not os.listdir(self.dataset_path):
                shutil.rmtree(self.dataset_path)

        if not os.path.exists(decompressed_folder):
            decompressFile(compressed_file, self.dataset_path)
            os.rename(os.path.join(self.dataset_path, sequence_name.replace('_dslr', '')), decompressed_folder)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        images_path = os.path.join(sequence_path, 'images', 'dslr_images_undistorted')
        rgb_path = os.path.join(sequence_path, 'rgb')
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
            for rgb_file in os.listdir(images_path):
                source_file = os.path.join(images_path, rgb_file)
                destination_file = os.path.join(rgb_path, rgb_file)
                shutil.move(source_file, destination_file)
            shutil.rmtree(os.path.join(sequence_path, 'images'))

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                name, ext = os.path.splitext(filename)
                ts = int(name.replace('DSC_', ''))
                file.write(f"{ts:.5f} rgb/{filename}\n")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calib_txt = os.path.join(sequence_path, 'dslr_calibration_undistorted', 'cameras.txt')

        # with open(calib_txt, 'r') as file:
        #     lines = file.read().strip().splitlines()
        #
        # camera_params = lines[-1].split()
        # camera_model = "OPENCV"
        # fx = float(camera_params[4])
        # fy = float(camera_params[4])
        # cx = float(camera_params[5])
        # cy = float(camera_params[6])

        fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('UNKNOWN', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        if os.path.exists(groundtruth_txt):
            return

        images_file = os.path.join(sequence_path, 'dslr_calibration_undistorted', 'images.txt')
        image_id, t_wc, q_wc_xyzw = get_colmap_keyframes(images_file, 4, True)

        image_ts = np.array(get_timestamps(sequence_path, 'rgb.txt'))
        timestamps = []
        for id in image_id:
            timestamps.append(float(image_ts[id-1]))

        timestamps = np.array(timestamps)

        write_trajectory_tum_format(groundtruth_txt, timestamps, t_wc, q_wc_xyzw)

    def remove_unused_files(self, sequence_name):
        calibration_undistorted_folder = os.path.join(self.dataset_path, sequence_name, 'dslr_calibration_undistorted')
        if os.path.exists(calibration_undistorted_folder):
            shutil.rmtree(calibration_undistorted_folder)
