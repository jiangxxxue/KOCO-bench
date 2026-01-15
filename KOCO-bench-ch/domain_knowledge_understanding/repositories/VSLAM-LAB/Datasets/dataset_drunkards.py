import os
import yaml
import pandas as pd
import re
from huggingface_hub import hf_hub_download
from zipfile import ZipFile

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics

class DRUNKARDS_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('drunkards', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [f"{self.dataset_name}_{s[:1]}" for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Variables
        compressed_name = sequence_name
        compressed_name_ext = compressed_name + '.zip'
        repo_id = self.url_download_root

        # Download the compressed file
        file_path = hf_hub_download(repo_id=repo_id, filename=compressed_name_ext, repo_type='dataset')
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_path)

        # Download instrinsics file
        resolution = self.get_sequence_resolution(sequence_name)
        intrinsics_file = f"intrinsics_{resolution}.txt"
        file_path = hf_hub_download(repo_id=repo_id, filename=intrinsics_file, repo_type='dataset')
        intrinsics_txt = os.path.join(sequence_path, intrinsics_file)
        with open(file_path, 'rb') as f_src, open(intrinsics_txt, 'wb') as f_dest:
            f_dest.write(f_src.read())

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        color_path = os.path.join(sequence_path, 'color')
        rgb_path = os.path.join(sequence_path, 'rgb')
        if os.path.exists(color_path) and os.path.isdir(color_path):
            os.rename(color_path, rgb_path)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                name, ext = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                file.write(f"{ts:.16f} rgb/{filename}\n")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        resolution = self.get_sequence_resolution(sequence_name)
        intrinsics_txt = os.path.join(sequence_path, f'intrinsics_{resolution}.txt')
        with open(intrinsics_txt, 'r') as file:
            for line in file:
                if line.startswith('fx, fy, cx, cy:'):
                    values_str = line.split(':')[1].strip()
                    fx, fy, cx, cy = map(float, values_str.split(','))

        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        pose_txt = os.path.join(sequence_path, 'pose.txt')
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        if os.path.exists(pose_txt):
            os.rename(pose_txt, groundtruth_txt)
            data = pd.read_csv(groundtruth_txt, sep=' ', header=None)
            data[0] = data[0] / self.rgb_hz
            data.to_csv(groundtruth_txt, sep=' ', header=False, index=False, float_format='%.16f')

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        resolution = self.get_sequence_resolution(sequence_name)
        intrinsics_txt = os.path.join(sequence_path, f'intrinsics_{resolution}.txt')
        os.remove(intrinsics_txt)


    def get_sequence_resolution(self, sequence_name):
        return int(re.search(r'_(\d+)_', sequence_name).group(1))
